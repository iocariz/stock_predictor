"""The tuning objective must measure the rule the strategy actually trades."""

from __future__ import annotations

import numpy as np
import pytest

from stock_predictor.signal_depth import top_n_excess_score

# Three dates, six names each.
DATES = np.repeat(np.arange(3), 6)
FWD = np.array([
    0.10, 0.08, 0.00, 0.00, -0.05, -0.06,   # date 0
    0.12, 0.09, 0.01, 0.00, -0.04, -0.07,   # date 1
    0.11, 0.07, 0.02, 0.00, -0.06, -0.05,   # date 2
])
# Position 0 is the best forward return on every date, 5 the worst.
PERFECT = np.tile(np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), 3)
INVERTED = -PERFECT


def _universe_mean() -> float:
    return float(np.mean([FWD[DATES == d].mean() for d in range(3)]))


def test_perfect_ranking_scores_above_the_universe() -> None:
    score = top_n_excess_score(DATES, PERFECT, FWD, top_n=2)
    expected = np.mean([
        FWD[DATES == d][:2].mean() - FWD[DATES == d].mean() for d in range(3)
    ])
    assert score == pytest.approx(expected)
    assert score > 0


def test_inverted_ranking_scores_below_the_universe() -> None:
    assert top_n_excess_score(DATES, INVERTED, FWD, top_n=2) < 0


def test_perfect_beats_inverted() -> None:
    good = top_n_excess_score(DATES, PERFECT, FWD, top_n=2)
    bad = top_n_excess_score(DATES, INVERTED, FWD, top_n=2)
    assert good > bad


def test_selecting_the_whole_universe_scores_zero() -> None:
    """With top_n >= names per date the basket *is* the universe."""
    assert top_n_excess_score(DATES, PERFECT, FWD, top_n=6) == pytest.approx(0.0)
    assert top_n_excess_score(DATES, INVERTED, FWD, top_n=99) == pytest.approx(0.0)


def test_metric_ignores_ordering_below_the_cutoff() -> None:
    """Two scorers that agree on the top 2 but disagree below must tie — the
    strategy never trades those names, so the objective must not reward them.
    This is exactly what NDCG over the full list does not do."""
    a = np.tile(np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), 3)
    b = np.tile(np.array([6.0, 5.0, 1.0, 2.0, 3.0, 4.0]), 3)
    assert top_n_excess_score(DATES, a, FWD, top_n=2) == pytest.approx(
        top_n_excess_score(DATES, b, FWD, top_n=2)
    )


def test_risk_adjusted_variant_penalises_inconsistency() -> None:
    """Two scorers with the same mean excess; the steadier one must win."""
    dates = np.repeat(np.arange(4), 4)
    fwd = np.array([
        0.04, 0.00, 0.00, 0.00,
        0.04, 0.00, 0.00, 0.00,
        0.16, 0.00, 0.00, 0.00,
        -0.08, 0.00, 0.00, 0.00,
    ])
    steady = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), 4)          # always picks index 0
    # Picks index 0 only on the two extreme dates, index 1 otherwise.
    lumpy = np.concatenate([
        [3.0, 4.0, 2.0, 1.0], [3.0, 4.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0],
    ])
    assert top_n_excess_score(dates, steady, fwd, top_n=1, risk_adjusted=True) > \
           top_n_excess_score(dates, lumpy, fwd, top_n=1, risk_adjusted=True)


def test_nan_forward_returns_are_skipped_not_propagated() -> None:
    fwd = FWD.copy()
    fwd[0] = np.nan
    score = top_n_excess_score(DATES, PERFECT, fwd, top_n=2)
    assert np.isfinite(score)


def test_bad_top_n_rejected() -> None:
    with pytest.raises(ValueError, match="top_n"):
        top_n_excess_score(DATES, PERFECT, FWD, top_n=0)


def test_degenerate_input_returns_a_finite_floor() -> None:
    assert np.isfinite(top_n_excess_score(np.array([0]), np.array([1.0]),
                                          np.array([0.05]), top_n=1))


# ---------------------------------------------------------------------------
# End-to-end through the Optuna search
# ---------------------------------------------------------------------------


def _training_panel(n_dates: int = 40, n_names: int = 30):
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        signal = rng.normal(0, 1, n_names)
        # Forward return genuinely follows the feature, so a tuner that works
        # should find it under any metric.
        # Scaled so the +5% binary label actually fires for the top names.
        fwd = 0.05 * signal + rng.normal(0, 0.02, n_names)
        for i in range(n_names):
            rows.append({
                "date": d, "ticker": f"T{i:02d}",
                "ret_1d": float(signal[i]), "vol_21d": 0.02,
                "fwd_ret": float(fwd[i]),
                "target_5pct": int(fwd[i] >= 0.05),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("metric", ["topn_excess", "topn_ir", "ndcg"])
def test_rank_search_runs_on_each_metric(metric: str) -> None:
    from stock_predictor.training import run_optuna_search

    best = run_optuna_search(
        _training_panel(), ["ret_1d"], ts_cv_splits=2, n_trials=1, seed=0,
        purge_days=1, objective="rank", rank_eval_k=5, optuna_metric=metric,
    )
    assert "n_estimators" in best


@pytest.mark.parametrize("metric", ["topn_excess", "pr_auc"])
def test_binary_search_runs_on_each_metric(metric: str) -> None:
    from stock_predictor.training import run_optuna_search

    best = run_optuna_search(
        _training_panel(), ["ret_1d"], ts_cv_splits=2, n_trials=1, seed=0,
        purge_days=1, objective="binary", rank_eval_k=5, optuna_metric=metric,
    )
    assert "n_estimators" in best


def test_aligned_metric_requires_forward_returns() -> None:
    from stock_predictor.training import run_optuna_search

    panel = _training_panel().drop(columns=["fwd_ret"])
    with pytest.raises(ValueError, match="fwd_ret"):
        run_optuna_search(
            panel, ["ret_1d"], ts_cv_splits=2, n_trials=1, seed=0,
            purge_days=1, objective="binary", optuna_metric="topn_excess",
        )
