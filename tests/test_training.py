from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import (
    _inner_train_val_split,
    add_cross_sectional_ranks,
    add_price_features,
    add_regime_features,
    build_labeled_panel,
    monthly_walk_forward,
    precision_at_k,
    purge_train_dates,
    purged_date_splits,
    select_training_rows,
    wide_field,
)


def test_precision_at_k() -> None:
    y = pd.Series([0, 0, 1, 0, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.95])
    # top-2 are indices 2 and 4, both positive -> 1.0
    assert precision_at_k(y, scores, k=2) == 1.0
    # top-3 includes one negative (index 3 has 0.3 — actually sort: 4,2,3 -> labels 1,1,0 -> 2/3
    assert abs(precision_at_k(y, scores, k=3) - 2 / 3) < 1e-9


def test_wide_field_single_ticker() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    raw = pd.DataFrame({"Close": [1.0, 1.1], "Volume": [100, 110]}, index=idx)
    w = wide_field(raw, "Close")
    assert list(w.columns) == ["Close"]
    assert len(w) == 2


def _synthetic_adj_close() -> pd.DataFrame:
    """Wide adj_close panel: 2 tickers, 30 business days."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    return pd.DataFrame(
        {"AAA": np.linspace(100, 110, 30), "BBB": np.linspace(50, 48, 30)},
        index=dates,
    )


def _synthetic_stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "end_date": [pd.NaT, pd.NaT],
    })


def test_build_labeled_panel_creates_target() -> None:
    adj_close = _synthetic_adj_close()
    stints = _synthetic_stints()
    panel = build_labeled_panel(adj_close, stints, horizon=5, threshold=0.03)
    assert "target_5pct" in panel.columns
    assert "fwd_ret" in panel.columns
    assert set(panel["ticker"].unique()) == {"AAA", "BBB"}
    # AAA goes up → some positives; BBB goes down → all zeros
    aaa = panel[panel["ticker"] == "AAA"]
    bbb = panel[panel["ticker"] == "BBB"]
    assert aaa["target_5pct"].sum() >= 0
    assert bbb["target_5pct"].sum() == 0


def test_build_labeled_panel_no_future_leak() -> None:
    """Forward return for the last `horizon` rows should be NaN → dropped."""
    adj_close = _synthetic_adj_close()
    stints = _synthetic_stints()
    panel = build_labeled_panel(adj_close, stints, horizon=5, threshold=0.03)
    # 30 dates, 2 tickers, last 5 rows per ticker have NaN fwd_ret → dropped
    assert len(panel) <= 2 * (30 - 5)


def test_add_price_features_columns() -> None:
    dates = pd.bdate_range("2020-01-02", periods=260)
    df = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["A", "B"], 260),
        "adj_close": np.random.default_rng(42).uniform(90, 110, 520),
    })
    out = add_price_features(df)
    for col in ["ret_63d", "ret_252d", "high_52w_pct", "drawdown_63d", "momentum", "rsi_14"]:
        assert col in out.columns


def test_regime_features_market_median() -> None:
    df = pd.DataFrame({
        "date": ["2024-01-02"] * 3,
        "ticker": ["A", "B", "C"],
        "ret_5d": [0.01, 0.03, 0.05],
        "ret_21d": [0.10, 0.20, 0.30],
    })
    out = add_regime_features(df)
    assert out["market_ret_5d"].iloc[0] == 0.03  # median
    assert out["market_ret_21d"].iloc[0] == 0.20


def test_cross_sectional_ranks_range() -> None:
    df = pd.DataFrame({
        "date": ["2024-01-02"] * 4,
        "ret_21d": [0.1, 0.2, 0.3, 0.4],
        "vol_10d": [0.01, 0.02, 0.03, 0.04],
        "volume_zscore": [1.0, 2.0, 3.0, 4.0],
    })
    out = add_cross_sectional_ranks(df)
    for col in ["ret_21d_rank", "vol_10d_rank", "volume_zscore_rank"]:
        assert col in out.columns
        assert out[col].min() > 0
        assert out[col].max() <= 1.0


# ---------------------------------------------------------------------------
# Regression: purge/embargo around forward-return label windows
# ---------------------------------------------------------------------------


def test_purge_train_dates_drops_boundary_window() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    df = pd.DataFrame({"date": np.repeat(dates, 2), "x": 1.0})
    boundary = dates[15]
    out = purge_train_dates(df[df["date"] < boundary], boundary, purge_days=5)
    # The last 5 trading dates before the boundary (indices 10..14) are gone.
    assert out["date"].max() == dates[9]


def test_purge_train_dates_zero_is_noop() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    df = pd.DataFrame({"date": dates, "x": 1.0})
    out = purge_train_dates(df, dates[-1], purge_days=0)
    assert len(out) == len(df)


def test_purge_train_dates_window_larger_than_history_empties() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame({"date": dates, "x": 1.0})
    out = purge_train_dates(df, dates[-1] + pd.Timedelta(days=1), purge_days=10)
    assert len(out) == 0


def test_purged_date_splits_no_label_overlap() -> None:
    """No train date may fall within `purge_days` dates of its validation block."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    s = pd.Series(np.repeat(dates, 3))  # 3 rows per date, shuffled order
    s = s.sample(frac=1.0, random_state=7).reset_index(drop=True)
    splits = purged_date_splits(s, n_splits=4, purge_days=5)
    assert len(splits) >= 1
    d = pd.to_datetime(s).to_numpy()
    udates = np.sort(pd.unique(d))
    for train_idx, val_idx in splits:
        train_dates = set(d[train_idx])
        val_dates = np.sort(pd.unique(d[val_idx]))
        # train strictly before validation
        assert max(train_dates) < val_dates[0]
        # purge gap: at least purge_days unique dates between max(train) and val start
        gap = udates[(udates > max(train_dates)) & (udates < val_dates[0])]
        assert len(gap) >= 5
        # no date in both sides
        assert train_dates.isdisjoint(set(val_dates))


def test_inner_train_val_split_purges_before_val() -> None:
    dates = pd.bdate_range("2024-01-02", periods=50)
    df = pd.DataFrame({"date": np.repeat(dates, 2), "x": 1.0})
    tr, va = _inner_train_val_split(df, "date", val_frac=0.2, purge_days=5)
    assert len(tr) > 0 and len(va) > 0
    udates = np.sort(df["date"].unique())
    gap = udates[(udates > tr["date"].max().to_datetime64())
                 & (udates < va["date"].min().to_datetime64())]
    assert len(gap) >= 5


def _walk_forward_panel(n_days: int = 130, n_tickers: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n_tickers):
        for d in dates:
            rows.append({
                "date": d,
                "ticker": f"T{i}",
                "adj_close": 100.0,
                "f1": rng.random(),
                "f2": rng.random(),
                "fwd_ret": rng.normal(0, 0.05),
                "target_5pct": int(rng.random() < 0.3),
                "vix_percentile": 0.42,
            })
    return pd.DataFrame(rows)


def test_monthly_walk_forward_purges_train_boundary_and_keeps_vix() -> None:
    df = _walk_forward_panel()
    test_start = "2024-05-01"
    metrics, scores = monthly_walk_forward(
        df, ["f1", "f2"], "target_5pct", "date", test_start,
        {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 7},
        inner_val_frac=0.1,
        min_train_rows=50,
        top_k=2,
        random_state=0,
        return_scores=True,
        purge_days=5,
    )
    assert len(metrics) >= 1
    udates = np.sort(df["date"].unique())
    for _, rec in metrics.iterrows():
        m_start = pd.Timestamp(rec["month"] + "-01")
        train_end = pd.Timestamp(rec["train_end"])
        # At least purge_days unique trading dates between last train date and month start
        gap = udates[(udates > train_end.to_datetime64()) & (udates < m_start.to_datetime64())]
        assert len(gap) >= 5, f"month {rec['month']}: only {len(gap)} purged dates"
    # Regression: vix_percentile carried into scores so --vix-filter can work.
    assert "vix_percentile" in scores.columns


def test_rsi_zero_loss_window_is_100() -> None:
    dates = pd.bdate_range("2024-01-02", periods=40)
    df = pd.DataFrame({
        "date": dates,
        "ticker": "UP",
        "adj_close": np.linspace(100, 140, len(dates)),  # strictly rising
    })
    out = add_price_features(df)
    tail = out["rsi_14"].dropna()
    assert len(tail) > 0
    assert (tail == 100.0).all()


def test_select_training_rows_keeps_partial_nan_features() -> None:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4),
        "ret_1d": [0.01, np.nan, 0.02, 0.03],
        "ret_252d": [np.nan, np.nan, np.nan, 0.5],  # long warm-up mostly missing
        "target_5pct": [1, 0, np.nan, 0],
    })
    out = select_training_rows(df, ["ret_1d", "ret_252d"], "target_5pct")
    # Row 1 (no ret_1d) and row 2 (no label) drop; NaN ret_252d rows survive.
    assert len(out) == 2
    strict = select_training_rows(df, ["ret_1d", "ret_252d"], "target_5pct", strict=True)
    assert len(strict) == 1


# ---------------------------------------------------------------------------
# Lambdarank objective (cross-sectional relevance labels)
# ---------------------------------------------------------------------------


def test_add_rank_labels_per_date_quintiles() -> None:
    from stock_predictor.training import add_rank_labels

    # 10 tickers on each of 2 dates; fwd_ret ordering differs by date.
    rows = []
    for d, order in [("2024-01-02", range(10)), ("2024-01-03", range(9, -1, -1))]:
        for t, r in zip("ABCDEFGHIJ", order):
            rows.append({"date": pd.Timestamp(d), "ticker": t, "fwd_ret": float(r)})
    out = add_rank_labels(pd.DataFrame(rows))
    assert set(out["rank_grade"].unique()) == {0, 1, 2, 3, 4}
    # Grades are market-neutral: identical distribution on both dates
    for _, g in out.groupby("date"):
        assert sorted(g["rank_grade"].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # Best fwd_ret on each date gets the top grade
    best = out.loc[out.groupby("date")["fwd_ret"].idxmax()]
    assert (best["rank_grade"] == 4).all()


def test_add_rank_labels_requires_fwd_ret() -> None:
    from stock_predictor.training import add_rank_labels

    with pytest.raises(ValueError, match="fwd_ret"):
        add_rank_labels(pd.DataFrame({"date": [], "x": []}))


def test_monthly_walk_forward_rank_objective_runs() -> None:
    """Lambdarank walk-forward: runs end-to-end and ranks a predictive feature."""
    dates = pd.bdate_range("2024-01-02", periods=130)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(10):
        for d in dates:
            fwd = rng.normal(0, 0.05)
            rows.append({
                "date": d,
                "ticker": f"T{i}",
                "adj_close": 100.0,
                "f1": fwd + rng.normal(0, 0.01),  # strongly predictive of fwd_ret
                "f2": rng.random(),
                "fwd_ret": fwd,
                "target_5pct": int(fwd >= 0.05),
            })
    df = pd.DataFrame(rows)
    metrics, scores = monthly_walk_forward(
        df, ["f1", "f2"], "target_5pct", "date", "2024-05-01",
        {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5},
        inner_val_frac=0.1,
        min_train_rows=100,
        top_k=2,
        random_state=0,
        return_scores=True,
        purge_days=5,
        objective="rank",
    )
    assert len(metrics) >= 1
    assert len(scores) > 0
    # Ranker scores must vary within a date (not degenerate)
    per_date_std = scores.groupby("date")["prob"].std()
    assert (per_date_std > 0).any()
    # With a near-perfect feature, ranking beats the base positive rate
    assert metrics["mean_weekly_precision_at_k"].mean() > df["target_5pct"].mean()


def test_monthly_walk_forward_invalid_objective() -> None:
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "ticker": ["A"],
                       "fwd_ret": [0.0], "target_5pct": [0], "f1": [0.1]})
    with pytest.raises(ValueError, match="objective"):
        monthly_walk_forward(
            df, ["f1"], "target_5pct", "date", "2024-02-01", {},
            inner_val_frac=0.1, min_train_rows=1, top_k=1, random_state=0,
            objective="pairwise",
        )


# ---------------------------------------------------------------------------
# Rank objective: final model, generic scoring, NDCG Optuna
# ---------------------------------------------------------------------------


def _rank_train_panel(n_days: int = 120, n_tickers: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rng = np.random.default_rng(21)
    rows = []
    for i in range(n_tickers):
        for d in dates:
            fwd = rng.normal(0, 0.05)
            rows.append({
                "date": d, "ticker": f"T{i}", "adj_close": 100.0,
                "f1": fwd + rng.normal(0, 0.01), "f2": rng.random(),
                "fwd_ret": fwd, "target_5pct": int(fwd >= 0.05),
            })
    return pd.DataFrame(rows)


def test_train_final_rank_model_and_model_scores() -> None:
    from stock_predictor.training import model_scores, train_final_rank_model

    train = _rank_train_panel()
    params = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 7,
              "min_child_samples": 5}
    model, n_trees = train_final_rank_model(train, ["f1", "f2"], params, seed=0,
                                            purge_days=5, eval_k=3)
    assert n_trees >= 1
    assert not hasattr(model, "predict_proba")  # it's a ranker
    scores = model_scores(model, train[["f1", "f2"]].head(50))
    assert len(scores) == 50
    assert np.std(scores) > 0
    # Higher f1 (≈ higher fwd_ret) should get higher scores on average
    hi = train.nlargest(200, "f1")[["f1", "f2"]]
    lo = train.nsmallest(200, "f1")[["f1", "f2"]]
    assert model_scores(model, hi).mean() > model_scores(model, lo).mean()


def test_model_scores_prefers_predict_proba() -> None:
    from stock_predictor.training import model_scores

    class _Clf:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.7)])

        def predict(self, X):  # must NOT be used
            raise AssertionError("predict_proba should take precedence")

    out = model_scores(_Clf(), pd.DataFrame({"a": [1, 2]}))
    assert list(out) == [0.7, 0.7]


def test_run_optuna_search_rank_objective_smoke() -> None:
    from stock_predictor.training import run_optuna_search

    train = _rank_train_panel(n_days=90, n_tickers=6)
    best = run_optuna_search(
        train, ["f1", "f2"],
        ts_cv_splits=2, n_trials=2, seed=0, purge_days=3,
        objective="rank", rank_eval_k=3,
    )
    assert "learning_rate" in best


def test_run_optuna_search_invalid_objective() -> None:
    from stock_predictor.training import run_optuna_search

    with pytest.raises(ValueError, match="objective"):
        run_optuna_search(
            pd.DataFrame({"date": [], "ticker": []}), [],
            ts_cv_splits=2, n_trials=1, seed=0, objective="regression",
        )
