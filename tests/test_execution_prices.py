"""Selection and valuation are different questions.

The backtest priced fills from the *scored* panel, which is point-in-time
filtered, and forward-filled it unconditionally. So when a holding left the
index — or was delisted, halted, or simply had a data gap — its rows stopped
and the last in-index price was carried forward indefinitely. Exits then
executed at that stale price, and rank-hold fell back to the entry price
outright ("exit flat").

Measured on `artifacts/final/wf_control.parquet`: **40 of 840 cohort legs had
no row at their exit date, affecting 11 of 56 cohorts.** DELL exited at a
forward-filled 237.64 against a true 456.79 — a 92% error on one leg. The
row-role fix reduced this to 10 of 840, but did not remove it.

The PIT filter decides what may be *selected*. It does not decide what a
holding is *worth*: you still own a name the index dropped, and it still has a
price. Execution prices therefore come from an unfiltered panel when one is
supplied, and every fill that had to fall back to a carried-forward price is
counted and reported rather than passing silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, run_backtest, run_rank_hold_backtest

DATES = pd.bdate_range("2024-01-01", periods=120)
N = 40
LEAVER = "T00"


def _scored(drop_from: int | None = 60) -> pd.DataFrame:
    """T00 is ranked top, then leaves the index part-way through."""
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (len(DATES), N)), axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            if t == LEAVER and drop_from is not None and di >= drop_from:
                continue          # PIT removal: the row simply stops
            rows.append({"date": d, "ticker": t, "prob": float(N - i),
                         "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


def _truth(scored: pd.DataFrame, leaver_after: float = 3.0) -> pd.DataFrame:
    """Unfiltered prices: the leaver keeps trading, and triples."""
    wide = scored.pivot_table(index="date", columns="ticker",
                              values="adj_close", aggfunc="first")
    wide = wide.reindex(DATES)
    last = wide[LEAVER].ffill()
    gap = wide[LEAVER].isna()
    wide[LEAVER] = wide[LEAVER].where(~gap, last * leaver_after)
    return wide.ffill()


CFG = dict(top_n=5, holding_days=20, max_overlapping_cohorts=2,
           slippage_bps=0.0, benchmark_ticker=None, rebalance_day="Friday")


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_stale_fills_are_counted_not_silent() -> None:
    res = run_backtest(_scored(), BacktestConfig(**CFG))
    assert res.metrics["stale_fills"] > 0
    assert res.metrics["stale_fill_rate"] > 0


def test_a_complete_panel_reports_no_stale_fills() -> None:
    res = run_backtest(_scored(drop_from=None), BacktestConfig(**CFG))
    assert res.metrics["stale_fills"] == 0
    assert res.metrics["stale_fill_rate"] == 0.0


def test_rank_hold_counts_them_too() -> None:
    res = run_rank_hold_backtest(_scored(), BacktestConfig(exit_rank=10, **CFG))
    assert "stale_fills" in res.metrics


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------


def test_execution_prices_are_used_when_supplied() -> None:
    """The leaver triples after removal; pricing from the scored panel misses
    it entirely."""
    scored = _scored()
    stale = run_backtest(scored, BacktestConfig(**CFG))
    real = run_backtest(scored, BacktestConfig(**CFG),
                        execution_prices=_truth(scored))
    assert real.metrics["total_return"] != pytest.approx(stale.metrics["total_return"])
    assert real.metrics["stale_fills"] < stale.metrics["stale_fills"]


def test_supplying_prices_does_not_change_selection() -> None:
    """Valuation must not leak into what gets picked."""
    scored = _scored()
    a = run_backtest(scored, BacktestConfig(**CFG))
    b = run_backtest(scored, BacktestConfig(**CFG), execution_prices=_truth(scored))
    assert [c.tickers for c in a.cohorts] == [c.tickers for c in b.cohorts]


def test_a_complete_execution_panel_removes_the_staleness() -> None:
    scored = _scored()
    res = run_backtest(scored, BacktestConfig(**CFG), execution_prices=_truth(scored))
    assert res.metrics["stale_fills"] == 0


def test_execution_prices_may_cover_only_some_names() -> None:
    """A partial panel must fill what it can rather than being rejected."""
    scored = _scored()
    partial = _truth(scored)[[LEAVER]]
    res = run_backtest(scored, BacktestConfig(**CFG), execution_prices=partial)
    assert res.metrics["stale_fills"] == 0


def test_rank_hold_accepts_execution_prices_too() -> None:
    scored = _scored()
    cfg = BacktestConfig(exit_rank=10, **CFG)
    a = run_rank_hold_backtest(scored, cfg)
    b = run_rank_hold_backtest(scored, cfg, execution_prices=_truth(scored))
    assert b.metrics["stale_fills"] <= a.metrics["stale_fills"]


def test_an_empty_execution_panel_is_ignored_not_fatal() -> None:
    res = run_backtest(_scored(), BacktestConfig(**CFG),
                       execution_prices=pd.DataFrame())
    assert res.metrics["stale_fills"] > 0
