"""Feature rows, label rows and tradable rows are three different things.

One dropna(fwd_ret) used to decide all three, so a row without a *future*
label was also removed as a feature row and as a tradable row. The cost:
the newest `horizon` sessions — the ones a live model must actually score —
never existed in the panel, and a delisted name could not be traded through
its final quarter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import build_labeled_panel, select_training_rows

DATES = pd.bdate_range("2024-01-01", periods=40)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"ALIVE": np.linspace(100, 130, len(DATES)),
         "ALSO": np.linspace(50, 65, len(DATES))},
        index=DATES,
    )


def test_role_flags_are_emitted() -> None:
    panel = build_labeled_panel(_prices(), None, horizon=10, threshold=0.05)
    assert {"has_label", "is_tradable"} <= set(panel.columns)


def test_the_newest_sessions_survive_as_tradable_rows() -> None:
    """The rows a live model has to score: priced, rankable, no label yet."""
    panel = build_labeled_panel(_prices(), None, horizon=10, threshold=0.05)
    assert panel["date"].max() == DATES[-1], "the panel must reach the data"
    tail = panel[panel["date"] > DATES[-11]]
    assert (~tail["has_label"]).all(), "no forward return exists yet"
    assert tail["is_tradable"].all(), "but they are priced and rankable"


def test_the_cross_section_stays_full_at_the_end() -> None:
    """Regression: the tail collapsed to a couple of delisted names, which is
    not a cross-section a ranker can use."""
    panel = build_labeled_panel(_prices(), None, horizon=10, threshold=0.05)
    per_date = panel.groupby("date")["ticker"].nunique()
    assert per_date.tail(10).min() == 2, "every name must still be present"


def test_training_uses_labelled_rows_only() -> None:
    panel = build_labeled_panel(_prices(), None, horizon=10, threshold=0.05)
    panel["ret_1d"] = 0.01
    train = select_training_rows(panel, ["ret_1d"], "target_5pct")
    assert train["has_label"].all(), "an unlabelled row cannot supervise anything"
    assert train["fwd_ret"].notna().all()
    assert train["date"].max() <= DATES[-11]


def test_unlabelled_rows_carry_no_target() -> None:
    """A 0 label on an unknown future would train the model on a fiction."""
    panel = build_labeled_panel(_prices(), None, horizon=10, threshold=0.05)
    tail = panel[~panel["has_label"]]
    assert tail["fwd_ret"].isna().all()
    assert tail["target_5pct"].isna().all()


def test_untradable_rows_are_excluded_entirely() -> None:
    px = _prices()
    px.loc[DATES[20:], "ALSO"] = np.nan
    panel = build_labeled_panel(px, None, horizon=10, threshold=0.05,
        terminal_fill="assume_delisted")
    also = panel[panel.ticker == "ALSO"]
    assert also["date"].max() == DATES[19], "no price, no row"
    assert also["is_tradable"].all()


def test_a_delisted_name_keeps_a_label_through_its_final_quarter() -> None:
    """Terminal fill still applies: the last sessions before a delisting are
    labelled to the price a holder would realize."""
    px = _prices()
    px.loc[DATES[25:], "ALSO"] = np.nan
    panel = build_labeled_panel(px, None, horizon=10, threshold=0.05,
        terminal_fill="assume_delisted")
    also = panel[panel.ticker == "ALSO"]
    labelled = also[also["has_label"]]
    assert labelled["date"].max() == DATES[23], "final session carries no return"
    assert also[also.date == DATES[20]]["fwd_ret"].iloc[0] == pytest.approx(
        px["ALSO"].iloc[24] / px["ALSO"].iloc[20] - 1
    )


def test_legacy_behaviour_is_still_reachable() -> None:
    panel = build_labeled_panel(
        _prices(), None, horizon=10, threshold=0.05, drop_unlabeled=True,
    )
    assert panel["has_label"].all()
    assert panel["date"].max() == DATES[-11]


# ---------------------------------------------------------------------------
# Walk-forward must score what it can trade, not only what it can grade
# ---------------------------------------------------------------------------


def _wf_panel(n_dates: int = 120, n_names: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_names)), axis=0))
    panel = build_labeled_panel(
        pd.DataFrame(px, index=dates, columns=[f"T{i:02d}" for i in range(n_names)]),
        None, horizon=10, threshold=0.02,
    )
    panel["ret_1d"] = rng.normal(0, 1, len(panel))
    panel["vol_21d"] = 0.02
    return panel


def test_scored_panel_reaches_the_end_of_the_data() -> None:
    """Regression: the newest horizon sessions were never scored, so a live
    model had nothing to trade on the dates that matter most."""
    from stock_predictor.training import monthly_walk_forward

    panel = _wf_panel()
    _, scores = monthly_walk_forward(
        panel, ["ret_1d"], "target_5pct", "date", "2024-03-01",
        {"n_estimators": 40, "learning_rate": 0.1},
        inner_val_frac=0.2, min_train_rows=50, top_k=5, random_state=0,
        return_scores=True, purge_days=1, objective="rank",
    )
    assert scores["date"].max() == panel["date"].max()


def test_the_tail_keeps_a_full_cross_section() -> None:
    from stock_predictor.training import monthly_walk_forward

    panel = _wf_panel()
    _, scores = monthly_walk_forward(
        panel, ["ret_1d"], "target_5pct", "date", "2024-03-01",
        {"n_estimators": 40, "learning_rate": 0.1},
        inner_val_frac=0.2, min_train_rows=50, top_k=5, random_state=0,
        return_scores=True, purge_days=1, objective="rank",
    )
    per_date = scores.groupby("date")["ticker"].nunique()
    assert per_date.tail(10).min() == per_date.median(), (
        "a ragged tail cannot support a cross-sectional ranking"
    )


def test_unlabelled_rows_are_scored_but_not_graded() -> None:
    from stock_predictor.training import monthly_walk_forward

    panel = _wf_panel()
    metrics, scores = monthly_walk_forward(
        panel, ["ret_1d"], "target_5pct", "date", "2024-03-01",
        {"n_estimators": 40, "learning_rate": 0.1},
        inner_val_frac=0.2, min_train_rows=50, top_k=5, random_state=0,
        return_scores=True, purge_days=1, objective="rank",
    )
    tail = scores[scores["date"] > panel["date"].max() - pd.Timedelta(days=14)]
    assert tail["prob"].notna().all(), "every tradable row gets a score"
    assert "has_label" in scores.columns
    # Metrics must be computed only where a label exists.
    assert metrics["pr_auc_pooled"].notna().any()
    assert (metrics["n_test"] > 0).all()


# ---------------------------------------------------------------------------
# The panel handed to the walk-forward
# ---------------------------------------------------------------------------


def _features() -> pd.DataFrame:
    """Two names over ten dates; the last three rows per name are unlabelled."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for t in ("AAA", "BBB"):
        for i, d in enumerate(dates):
            labelled = i < 7
            rows.append({
                "date": d, "ticker": t,
                "adj_close": 100.0 + i,
                "ret_1d": np.nan if i == 0 else 0.01,   # no history on day one
                "mom_21d": 0.5,
                "fwd_ret": 0.02 if labelled else np.nan,
                "target_5pct": 0.0 if labelled else np.nan,
                "has_label": labelled,
                "is_tradable": True,
            })
    return pd.DataFrame(rows)


def test_training_rows_require_a_label_and_a_price_history() -> None:
    from stock_predictor.training import select_training_rows

    out = select_training_rows(_features(), ["ret_1d", "mom_21d"], "target_5pct")
    assert out["has_label"].all(), "an unlabelled row cannot supervise"
    assert out["ret_1d"].notna().all(), "day one has no return to learn from"
    assert len(out) == 2 * 6, "10 dates, minus 3 unlabelled, minus day one"


def test_scoring_rows_require_features_but_not_a_label() -> None:
    """The rows a live model has to rank are exactly the unlabelled newest
    ones. Requiring a label here is what deleted them from the backtest."""
    from stock_predictor.training import select_scoring_rows

    out = select_scoring_rows(_features(), ["ret_1d", "mom_21d"], "target_5pct")
    assert len(out) == 2 * 9, "only day one drops, for want of a return"
    assert not out["has_label"].all(), "unlabelled rows must survive"
    assert out["date"].max() == pd.Timestamp("2024-01-12"), "the newest session is kept"


def test_scoring_rows_are_a_superset_of_training_rows() -> None:
    from stock_predictor.training import select_scoring_rows, select_training_rows

    cols = ["ret_1d", "mom_21d"]
    f = _features()
    tr = select_training_rows(f, cols, "target_5pct")
    sc = select_scoring_rows(f, cols, "target_5pct")
    assert set(tr.index) <= set(sc.index)
    assert len(sc) > len(tr)


def test_strict_mode_still_drops_any_nan_feature() -> None:
    from stock_predictor.training import select_scoring_rows

    f = _features()
    f.loc[f["ticker"] == "BBB", "mom_21d"] = np.nan
    out = select_scoring_rows(f, ["ret_1d", "mom_21d"], "target_5pct", strict=True)
    assert set(out["ticker"]) == {"AAA"}
