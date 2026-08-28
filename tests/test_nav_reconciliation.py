"""A NAV nobody can reconcile is a NAV nobody can check.

The identity that has to hold for either engine:

    final NAV = initial capital
              + P&L of everything that closed
              + unrealized P&L of whatever is still held

Positions still open at the end never appear in the closed-trade list, so
without the last term the ledger cannot explain the curve. Rank-hold ended a
real run 17% above its closed P&L for exactly that reason -- 15 positions still
open, worth 31,764 unrealized -- and there was no way to tell from the outside
whether that was correct or money appearing from nowhere.

So the engines report the open leg, and this asserts the identity closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    run_backtest,
    run_rank_hold_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=180)
N = 30

TOLERANCE = 1e-9
"""Float arithmetic on the same quantities, not a modelling approximation."""


def _panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (len(DATES), N)), axis=0))
    return pd.DataFrame([
        {"date": d, "ticker": f"T{i:02d}", "prob": float(N - i),
         "adj_close": float(px[di, i])}
        for di, d in enumerate(DATES) for i in range(N)
    ])


def _reconcile(result, config) -> tuple[float, float]:
    """(reported NAV, NAV rebuilt from the ledger)."""
    last = result.daily_nav.index[-1]
    closed = sum(
        c.capital * c.net_return for c in result.cohorts
        if pd.Timestamp(c.exit_date) <= last
    )
    m = result.metrics
    unrealized = (float(m.get("open_position_value", 0.0))
                  - float(m.get("open_position_basis", 0.0)))
    return float(result.daily_nav.iloc[-1]), (
        float(config.initial_capital) + closed + unrealized
    )


def _cfg(**kw) -> BacktestConfig:
    base = dict(top_n=5, holding_days=21, max_overlapping_cohorts=2,
                slippage_bps=5.0, benchmark_ticker=None,
                rebalance_day="Friday", reject_stale_fills=True)
    base.update(kw)
    return BacktestConfig(**base)


# ---------------------------------------------------------------------------


def test_rank_hold_nav_reconciles_with_open_positions_outstanding() -> None:
    cfg = _cfg(exit_rank=20)
    res = run_rank_hold_backtest(_panel(), cfg)
    assert res.metrics["n_open_positions"] > 0, "fixture must leave something open"
    reported, ledger = _reconcile(res, cfg)
    assert reported == pytest.approx(ledger, rel=TOLERANCE)


def test_rank_hold_reports_the_open_leg() -> None:
    res = run_rank_hold_backtest(_panel(), _cfg(exit_rank=20))
    assert res.metrics["open_position_value"] > 0
    assert res.metrics["open_position_basis"] > 0


def test_cohort_nav_reconciles() -> None:
    cfg = _cfg()
    res = run_backtest(_panel(), cfg)
    reported, ledger = _reconcile(res, cfg)
    assert reported == pytest.approx(ledger, rel=TOLERANCE)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_identity_holds_across_panels(seed: int) -> None:
    """One passing case could be a coincidence of a flat market."""
    cfg = _cfg(exit_rank=20)
    res = run_rank_hold_backtest(_panel(seed), cfg)
    reported, ledger = _reconcile(res, cfg)
    assert reported == pytest.approx(ledger, rel=TOLERANCE)


def test_an_open_position_is_marked_not_carried_at_cost() -> None:
    """If the mark were the cost basis the identity would still close, but the
    NAV would be wrong. The two must differ when prices have moved."""
    res = run_rank_hold_backtest(_panel(), _cfg(exit_rank=20))
    m = res.metrics
    assert m["open_position_value"] != pytest.approx(m["open_position_basis"])
