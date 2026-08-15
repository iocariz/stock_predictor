"""Score-floor (`min_prob`) gating for both backtest engines."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    run_backtest,
    run_rank_hold_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=80)


def _panel(probs_by_ticker: dict[str, float]) -> pd.DataFrame:
    rows = []
    for di, d in enumerate(DATES):
        for ti, (t, p) in enumerate(probs_by_ticker.items()):
            rows.append({
                "date": d,
                "ticker": t,
                "prob": p,
                "adj_close": 100.0 + ti + 0.05 * di,
            })
    return pd.DataFrame(rows)


def _graded_panel(n: int = 20) -> pd.DataFrame:
    """Tickers with evenly spaced scores from 0.05 to ~1.0."""
    return _panel({f"T{i}": round(0.05 + 0.95 * i / (n - 1), 4) for i in range(n)})


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_min_prob_defaults_to_off() -> None:
    assert BacktestConfig().min_prob is None


def test_min_prob_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="min_prob"):
        BacktestConfig(min_prob=float("nan"))


# ---------------------------------------------------------------------------
# Cohort engine
# ---------------------------------------------------------------------------


def test_cohort_min_prob_excludes_low_scores() -> None:
    panel = _graded_panel(20)
    cfg = BacktestConfig(benchmark_ticker=None, top_n=10, min_prob=0.8)
    result = run_backtest(panel, cfg)
    assert result.metrics["n_cohorts"] > 0
    for cohort in result.cohorts:
        for t in cohort.tickers:
            score = panel.loc[panel["ticker"] == t, "prob"].iloc[0]
            assert score >= 0.8, f"{t} scored {score}, below the floor"


def test_cohort_min_prob_shrinks_the_basket_and_renormalizes() -> None:
    """Only 4 names clear the floor, so a top-10 cohort holds 4 at 25% each."""
    panel = _graded_panel(20)
    cfg = BacktestConfig(benchmark_ticker=None, top_n=10, min_prob=0.86)
    result = run_backtest(panel, cfg)
    eligible = panel.drop_duplicates("ticker")
    n_eligible = int((eligible["prob"] >= 0.86).sum())
    assert 0 < n_eligible < 10
    first = result.cohorts[0]
    assert len(first.tickers) == n_eligible
    assert sum(first.weights) == pytest.approx(1.0)


def test_cohort_min_prob_above_every_score_trades_nothing() -> None:
    cfg = BacktestConfig(benchmark_ticker=None, min_prob=1.5)
    result = run_backtest(_graded_panel(20), cfg)
    assert result.metrics["n_cohorts"] == 0
    # NAV stays flat at the starting capital.
    assert result.daily_nav.nunique() == 1
    assert result.daily_nav.iloc[0] == pytest.approx(cfg.initial_capital)


def test_cohort_min_prob_none_matches_unfiltered_run() -> None:
    panel = _graded_panel(20)
    a = run_backtest(panel, BacktestConfig(benchmark_ticker=None))
    b = run_backtest(panel, BacktestConfig(benchmark_ticker=None, min_prob=None))
    pd.testing.assert_series_equal(a.daily_nav, b.daily_nav)


def test_cohort_min_prob_below_every_score_is_a_noop() -> None:
    panel = _graded_panel(20)
    a = run_backtest(panel, BacktestConfig(benchmark_ticker=None))
    b = run_backtest(panel, BacktestConfig(benchmark_ticker=None, min_prob=-1.0))
    pd.testing.assert_series_equal(a.daily_nav, b.daily_nav)


# ---------------------------------------------------------------------------
# Rank-hold engine
# ---------------------------------------------------------------------------


def test_rank_hold_min_prob_gates_buys_only() -> None:
    panel = _graded_panel(20)
    cfg = BacktestConfig(
        benchmark_ticker=None, top_n=5, exit_rank=15, min_prob=0.8,
    )
    result = run_rank_hold_backtest(panel, cfg)
    for cohort in result.cohorts:
        score = panel.loc[panel["ticker"] == cohort.tickers[0], "prob"].iloc[0]
        assert score >= 0.8


def test_rank_hold_min_prob_above_every_score_never_buys() -> None:
    cfg = BacktestConfig(benchmark_ticker=None, top_n=5, exit_rank=15, min_prob=1.5)
    result = run_rank_hold_backtest(_graded_panel(20), cfg)
    assert result.metrics["n_open_positions"] == 0
    assert result.daily_nav.iloc[-1] == pytest.approx(cfg.initial_capital)


def test_rank_hold_min_prob_none_matches_unfiltered_run() -> None:
    panel = _graded_panel(20)
    base = BacktestConfig(benchmark_ticker=None, top_n=5, exit_rank=15)
    a = run_rank_hold_backtest(panel, base)
    b = run_rank_hold_backtest(
        panel, BacktestConfig(benchmark_ticker=None, top_n=5, exit_rank=15, min_prob=None),
    )
    pd.testing.assert_series_equal(a.daily_nav, b.daily_nav)
