"""Fail-fast guards for silently-wrong backtest configurations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    _compute_weights,
    run_backtest,
    run_rank_hold_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=60)


def _panel(*, with_vix: bool = False, probs: np.ndarray | None = None) -> pd.DataFrame:
    tickers = [f"T{i}" for i in range(20)]
    rows = []
    for di, d in enumerate(DATES):
        for ti, t in enumerate(tickers):
            p = 0.5 + 0.01 * ti if probs is None else float(probs[ti % len(probs)])
            row = {
                "date": d,
                "ticker": t,
                "prob": p,
                "adj_close": 100.0 + ti + 0.1 * di,
            }
            if with_vix:
                row["vix_percentile"] = 0.2
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VIX regime options must not silently no-op
# ---------------------------------------------------------------------------


def test_vix_filter_without_column_raises() -> None:
    """Regression: --vix-filter 0.10 on a panel with no vix_percentile column
    produced results byte-identical to no filter, with no warning."""
    cfg = BacktestConfig(benchmark_ticker=None, vix_filter_percentile=0.10)
    with pytest.raises(ValueError, match="vix_percentile"):
        run_backtest(_panel(with_vix=False), cfg)


def test_vix_scale_exposure_without_column_raises() -> None:
    cfg = BacktestConfig(benchmark_ticker=None, vix_scale_exposure=True)
    with pytest.raises(ValueError, match="vix_percentile"):
        run_backtest(_panel(with_vix=False), cfg)


def test_rank_hold_vix_filter_without_column_raises() -> None:
    cfg = BacktestConfig(
        benchmark_ticker=None, vix_filter_percentile=0.10, exit_rank=40,
    )
    with pytest.raises(ValueError, match="vix_percentile"):
        run_rank_hold_backtest(_panel(with_vix=False), cfg)


def test_vix_options_work_when_column_present() -> None:
    cfg = BacktestConfig(benchmark_ticker=None, vix_filter_percentile=0.90)
    result = run_backtest(_panel(with_vix=True), cfg)
    assert result.metrics["n_cohorts"] > 0


def test_no_vix_options_means_no_column_required() -> None:
    result = run_backtest(_panel(with_vix=False), BacktestConfig(benchmark_ticker=None))
    assert result.metrics["n_cohorts"] > 0


# ---------------------------------------------------------------------------
# probability weighting must never short or lever
# ---------------------------------------------------------------------------


def test_probability_weights_reject_negative_scores() -> None:
    """Regression: raw lambdarank scores straddle zero, so p/sum(p) produced
    negative (short) weights in a long-only backtest."""
    with pytest.raises(ValueError, match="negative"):
        _compute_weights(np.array([2.1, 1.4, -0.9, -1.2]), "probability")


def test_probability_weights_reject_near_cancelling_scores() -> None:
    """Regression: scores summing to ~0 produced 50x leverage on one name."""
    with pytest.raises(ValueError, match="negative"):
        _compute_weights(np.array([1.0, -0.99, 0.01]), "probability")


def test_probability_weights_are_a_convex_combination() -> None:
    w = _compute_weights(np.array([0.2, 0.5, 0.3]), "probability")
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all() and (w <= 1).all()


def test_all_zero_scores_fall_back_to_equal_weights() -> None:
    w = _compute_weights(np.zeros(4), "probability")
    np.testing.assert_allclose(w, np.full(4, 0.25))


def test_equal_weighting_is_unaffected_by_negative_scores() -> None:
    w = _compute_weights(np.array([1.0, -1.0, 0.5]), "equal")
    np.testing.assert_allclose(w, np.full(3, 1 / 3))


def test_run_backtest_rejects_probability_weighting_on_ranker_scores() -> None:
    """A ranker panel + --weighting probability should fail up front with an
    actionable message rather than deep inside cohort construction."""
    panel = _panel(probs=np.array([2.0, 0.5, -0.5, -2.0]))
    cfg = BacktestConfig(benchmark_ticker=None, weighting="probability")
    with pytest.raises(ValueError, match="weighting='probability'"):
        run_backtest(panel, cfg)
