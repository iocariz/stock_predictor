"""Backtest, paper and live must select the same names.

The three paths are allowed to differ in how they *hold* state and in share
granularity. They are not allowed to differ in *what they choose*. Before the
shared core, `--min-prob`, `--rank-offset` and `--min-cross-section` reached
the simulation and never the live path, so a configuration could be measured
and then silently not traded.

These tests compare the two ends directly rather than trusting each side's own
unit tests, because that is exactly the seam a passing suite hid before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, _build_cohort
from stock_predictor.execution import SelectionRules
from stock_predictor.execution_calendar import trading_dates_from_index
from stock_predictor.portfolio import PortfolioState, generate_orders

DATES = pd.bdate_range("2024-01-01", periods=40)
N = 60


def _scored_day() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(N)],
        "prob": np.linspace(0.99, 0.01, N),
        "adj_close": np.linspace(50.0, 200.0, N),
    })


def _price_panel(day: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.tile(day["adj_close"].to_numpy(), (len(DATES), 1)),
        index=DATES, columns=day["ticker"],
    )


def _backtest_picks(day: pd.DataFrame, config: BacktestConfig) -> list[str]:
    built = _build_cohort(
        DATES[0], _price_panel(day), day, config,
        np.array(DATES, dtype="datetime64[ns]"), capital=1_000_000.0,
    )
    # A leg whose exit defers settles separately and becomes its own cohort,
    # so the names entered on this signal are the union across them.
    return sorted({t for c in built for t in c.tickers})


def _live_picks(day: pd.DataFrame, config: BacktestConfig) -> list[str]:
    orders, _ = generate_orders(
        PortfolioState(cash=1_000_000.0),
        day.to_dict("records"),
        dict(zip(day["ticker"], day["adj_close"], strict=True)),
        top_n=config.top_n,
        max_cohorts=config.max_overlapping_cohorts,
        holding_days=config.holding_days,
        slippage_bps=config.slippage_bps,
        as_of=DATES[0].strftime("%Y-%m-%d"),
        trading_dates=trading_dates_from_index(DATES),
        weighting=config.weighting,
        rank_offset=config.rank_offset,
        min_prob=config.min_prob,
        min_cross_section=config.min_cross_section,
        force=True,
    )
    return sorted(o.ticker for o in orders if o.action == "BUY")


CONFIGS = {
    "plain": dict(top_n=10),
    "score floor": dict(top_n=10, min_prob=0.5),
    "rank band": dict(top_n=10, rank_offset=5),
    "band and floor": dict(top_n=10, rank_offset=5, min_prob=0.4),
    "probability weights": dict(top_n=10, weighting="probability"),
    "wide basket": dict(top_n=25, exit_rank=30),
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_backtest_and_the_live_path_pick_the_same_names(name: str) -> None:
    config = BacktestConfig(benchmark_ticker=None, slippage_bps=0.0, **CONFIGS[name])
    day = _scored_day()
    assert _backtest_picks(day, config) == _live_picks(day, config), name


def test_a_score_floor_actually_reaches_the_live_path() -> None:
    """The regression this whole module exists for. A floor above every score
    must stop the live path trading, not be ignored by it."""
    day = _scored_day()
    config = BacktestConfig(benchmark_ticker=None, top_n=10, min_prob=1.5)
    assert _live_picks(day, config) == []
    assert _backtest_picks(day, config) == []


def test_a_rank_offset_actually_reaches_the_live_path() -> None:
    day = _scored_day()
    head = BacktestConfig(benchmark_ticker=None, top_n=5, slippage_bps=0.0)
    band = BacktestConfig(benchmark_ticker=None, top_n=5, rank_offset=10,
                          slippage_bps=0.0)
    assert set(_live_picks(day, head)).isdisjoint(_live_picks(day, band))


def test_a_thin_cross_section_stops_the_live_path_too() -> None:
    """Two names is not a ranking in a simulation and is not one live."""
    thin = _scored_day().head(2)
    config = BacktestConfig(benchmark_ticker=None, top_n=10)
    assert _live_picks(thin, config) == []
    assert _backtest_picks(thin, config) == []


# ---------------------------------------------------------------------------
# Where they are allowed to differ
# ---------------------------------------------------------------------------


def test_live_buys_whole_shares_and_the_backtest_need_not() -> None:
    """The one legitimate divergence: an account cannot hold 25.4 shares."""
    from stock_predictor.execution import CostModel, select_targets, size_targets

    targets = select_targets(_scored_day(), SelectionRules(top_n=7))
    costs = CostModel(slippage_bps=0.0)
    sim = size_targets(targets, 100_000.0, costs, whole_shares=False)
    live = size_targets(targets, 100_000.0, costs, whole_shares=True)

    assert [lot.ticker for lot in sim] == [lot.ticker for lot in live], (
        "same names, only the quantities round"
    )
    assert all(float(lot.shares).is_integer() for lot in live)
    assert any(not float(lot.shares).is_integer() for lot in sim)
    assert sum(lot.cost for lot in live) <= sum(lot.cost for lot in sim) + 1e-6


def test_both_sides_charge_the_same_fill_price() -> None:
    """A backtest that fills cheaper than the account is the whole problem."""
    from stock_predictor.backtest import _apply_slippage
    from stock_predictor.execution import CostModel

    costs = CostModel(slippage_bps=25.0)
    assert _apply_slippage(100.0, 25.0, +1) == pytest.approx(costs.fill_price(100.0, 1))
    assert _apply_slippage(100.0, 25.0, -1) == pytest.approx(costs.fill_price(100.0, -1))


# ---------------------------------------------------------------------------
# The rules are reachable from the live CLI
# ---------------------------------------------------------------------------


def test_the_live_cli_exposes_every_selection_rule() -> None:
    """A rule the backtest can express and the live CLI cannot is a
    configuration you can measure and then fail to trade."""
    import re
    from pathlib import Path

    def flags(mod: str) -> set[str]:
        src = Path("src/stock_predictor") / mod
        return set(re.findall(r'"(--[a-z-]+)"', src.read_text()))

    selection = {"--top-n", "--exit-rank", "--weighting", "--holding-days",
                 "--max-cohorts", "--min-prob", "--rank-offset",
                 "--min-cross-section", "--slippage-bps"}
    missing = (flags("backtest.py") & selection) - flags("predict.py")
    assert not missing, f"backtest-only selection rules: {sorted(missing)}"
