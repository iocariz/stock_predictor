"""A fill needs a real quote on the session it executes.

`_prepare_scored` forward-fills the price panel and execution read straight
from it, so a leg with no quote on its entry or exit session filled at an
earlier price. Counting those after the fact — which is all the previous change
did — does not stop them happening, and `specs.md` is explicit: the scored panel
must not be forward-filled to create execution prices, and every requested fill
must record either Filled or Rejected.

Three gaps the counting missed entirely:

* rank-hold **buys** were never counted, so a name bought at a carried-forward
  price reported nothing;
* the rank-hold denominator was ``2 * len(closed)``, ignoring every entry for a
  position still open;
* the long-short engine ignored execution prices and the diagnostics together.

Forward filling is now confined to *valuation* — marking an open position
between quotes. Execution requires the real thing.
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
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

DATES = pd.bdate_range("2024-01-01", periods=90)
N = 30
GAP = "T00"          # top-ranked, and missing quotes in the middle


def _panel(gap: tuple[int, int] | None = (20, 40)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (len(DATES), N)), axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            if t == GAP and gap and gap[0] <= di <= gap[1]:
                continue
            rows.append({"date": d, "ticker": t, "prob": float(N - i),
                         "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


CFG = dict(top_n=5, holding_days=10, slippage_bps=0.0,
           benchmark_ticker=None, rebalance_day="Friday")


# ---------------------------------------------------------------------------
# Rejection, not just counting
# ---------------------------------------------------------------------------


def test_a_cohort_leg_without_a_real_quote_is_rejected() -> None:
    """T00 is tradable outside its gap, so the property is not "never traded"
    but "never traded on a session where it had no quote"."""
    panel = _panel()
    have = set(zip(panel.date, panel.ticker, strict=True))
    res = run_backtest(panel, BacktestConfig(**CFG))
    for c in res.cohorts:
        for t in c.tickers:
            for when in (c.entry_date, c.exit_date):
                assert (pd.Timestamp(when), t) in have, (
                    f"filled {t} on {when} with no quote on that session"
                )


def test_rank_hold_does_not_buy_what_it_cannot_price() -> None:
    """The reported reproduction: bought at a carried-forward quote and
    reported nothing, because buys were never counted."""
    panel = _panel()
    have = set(zip(panel.date, panel.ticker, strict=True))
    res = run_rank_hold_backtest(panel, BacktestConfig(exit_rank=10, **CFG))
    for c in res.cohorts:
        for t in c.tickers:
            for when in (c.entry_date, c.exit_date):
                assert (pd.Timestamp(when), t) in have, (
                    f"filled {t} on {when} with no quote on that session"
                )


def test_a_clean_panel_fills_normally() -> None:
    """Rejection must not become a general refusal to trade."""
    res = run_backtest(_panel(gap=None), BacktestConfig(**CFG))
    assert res.cohorts
    assert any(GAP in c.tickers for c in res.cohorts)
    assert res.metrics["fills_rejected"] == 0


# ---------------------------------------------------------------------------
# Every requested fill is accounted for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine", "extra"),
    [(run_backtest, {}), (run_rank_hold_backtest, {"exit_rank": 10})],
    ids=["cohort", "rank_hold"],
)
def test_requested_equals_filled_plus_rejected(engine, extra) -> None:
    m = engine(_panel(), BacktestConfig(**CFG, **extra)).metrics
    assert m["fills_requested"] == m["fills_filled"] + m["fills_rejected"]
    assert m["fills_requested"] > 0


@pytest.mark.parametrize(
    ("engine", "extra"),
    [(run_backtest, {}), (run_rank_hold_backtest, {"exit_rank": 10})],
    ids=["cohort", "rank_hold"],
)
def test_rejections_are_reported_when_they_happen(engine, extra) -> None:
    m = engine(_panel(), BacktestConfig(**CFG, **extra)).metrics
    assert m["fills_rejected"] > 0
    assert 0 < m["fill_reject_rate"] <= 1


def test_the_denominator_counts_entries_of_still_open_positions() -> None:
    """It was 2 * len(closed), so every entry for a position still open was
    outside the denominator entirely."""
    res = run_rank_hold_backtest(_panel(), BacktestConfig(exit_rank=10, **CFG))
    opened = len(res.cohorts) + int(res.metrics["n_open_positions"])
    assert res.metrics["fills_requested"] >= opened


# ---------------------------------------------------------------------------
# Execution prices remove the rejections
# ---------------------------------------------------------------------------


def _truth() -> pd.DataFrame:
    full = _panel(gap=None)
    return full.pivot_table(index="date", columns="ticker",
                            values="adj_close", aggfunc="first")


def test_supplying_real_prices_lets_the_fills_through() -> None:
    gapped = _panel()
    without = run_backtest(gapped, BacktestConfig(**CFG))
    with_px = run_backtest(gapped, BacktestConfig(**CFG), execution_prices=_truth())
    assert with_px.metrics["fills_rejected"] < without.metrics["fills_rejected"]
    assert any(GAP in c.tickers for c in with_px.cohorts)


# ---------------------------------------------------------------------------
# The long-short engine was skipped entirely
# ---------------------------------------------------------------------------


def _ls_cfg(**kw) -> LongShortConfig:
    base = dict(rebalance_every=10, slippage_bps=0.0, risk_free_rate=0.0,
                short_borrow_annual=0.0, benchmark_ticker=None,
                min_names_per_side=2, decile=0.2)
    base.update(kw)
    return LongShortConfig(**base)


def test_long_short_reports_fill_accounting() -> None:
    m = run_long_short_backtest(_panel(), _ls_cfg()).metrics
    assert "fills_requested" in m and "fills_rejected" in m
    assert m["fills_requested"] == m["fills_filled"] + m["fills_rejected"]


def test_long_short_accepts_execution_prices() -> None:
    gapped = _panel()
    a = run_long_short_backtest(gapped, _ls_cfg())
    b = run_long_short_backtest(gapped, _ls_cfg(), execution_prices=_truth())
    assert b.metrics["fills_rejected"] <= a.metrics["fills_rejected"]


def test_long_short_does_not_trade_an_unpriceable_name() -> None:
    res = run_long_short_backtest(_panel(), _ls_cfg())
    assert res.metrics["fills_rejected"] > 0


# ---------------------------------------------------------------------------
# Valuation may still be forward-filled
# ---------------------------------------------------------------------------


def test_marking_open_positions_still_uses_a_carried_price() -> None:
    """A position held across a quote gap must still be valued; the NAV series
    cannot go blank. Forward filling is for marking, not for filling."""
    res = run_backtest(_panel(), BacktestConfig(**CFG))
    assert res.daily_nav.notna().all()
    assert (res.daily_nav > 0).all()
