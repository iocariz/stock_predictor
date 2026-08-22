"""--force-rebalance must not override the drawdown kill switch.

    may_open = allow_buys and not (repeat_signal or off_schedule) or force

`and` binds tighter than `or`, so this parses as
`(allow_buys and not blocked) or force` — and `force` alone was enough to open
positions. A portfolio down 50% with the kill switch engaged produced five BUY
orders in both holding modes, and `--confirm` persisted them.

The report made it worse: it printed "KILL-SWITCH ENGAGED — no new positions"
next to "5 buys", suppressing the picks listing while the state gained them.

force is about *timing* — overriding the weekly schedule and the repeat-signal
guard. It was never meant to override risk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.portfolio import (
    PortfolioState,
    generate_orders,
    generate_orders_rank_hold,
)

DATES = pd.bdate_range(end="2024-06-14", periods=200)
AS_OF = "2024-06-14"          # a Friday
PICKS = [{"ticker": f"T{i:02d}", "prob": 0.9 - i * 0.01, "adj_close": 100.0}
         for i in range(30)]
PRICES = {p["ticker"]: p["adj_close"] for p in PICKS}

MODES = [
    ("fixed", generate_orders, dict(max_cohorts=2, holding_days=10)),
    ("rank", generate_orders_rank_hold, dict(exit_rank=30)),
]


def _buys(fn, extra, **kw):
    orders, state = fn(
        PortfolioState(cash=100_000.0), PICKS, PRICES,
        top_n=5, slippage_bps=0.0, as_of=AS_OF,
        trading_dates=DATES.to_numpy(), **extra, **kw,
    )
    return [o for o in orders if o.action == "BUY"], state


# ---------------------------------------------------------------------------
# Risk beats timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_force_does_not_override_the_kill_switch(name, fn, extra) -> None:
    buys, state = _buys(fn, extra, allow_buys=False, force=True)
    assert buys == [], f"{name}: --force-rebalance opened positions while halted"
    assert state.positions == (), f"{name}: halted state gained positions"


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_a_halted_book_opens_nothing_without_force_either(name, fn, extra) -> None:
    assert _buys(fn, extra, allow_buys=False, force=False)[0] == []


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_an_unhalted_book_still_trades(name, fn, extra) -> None:
    """The guard must not block the ordinary case."""
    assert _buys(fn, extra, allow_buys=True, force=True)[0]


# ---------------------------------------------------------------------------
# force still overrides what it was for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_force_still_overrides_the_weekly_schedule(name, fn, extra) -> None:
    """AS_OF is a Friday; asking for Monday makes it off-schedule."""
    blocked = _buys(fn, extra, allow_buys=True, force=False,
                    rebalance_day="Monday")[0]
    forced = _buys(fn, extra, allow_buys=True, force=True,
                   rebalance_day="Monday")[0]
    assert blocked == [], f"{name}: off-schedule should block without force"
    assert forced, f"{name}: force should still override the schedule"


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_force_still_overrides_the_repeat_signal_guard(name, fn, extra) -> None:
    orders, _ = fn(
        PortfolioState(cash=100_000.0, last_signal_date=AS_OF),
        PICKS, PRICES, top_n=5, slippage_bps=0.0, as_of=AS_OF,
        trading_dates=DATES.to_numpy(), allow_buys=True, force=True, **extra,
    )
    assert [o for o in orders if o.action == "BUY"], f"{name}: force ignored"


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_a_repeat_signal_is_blocked_without_force(name, fn, extra) -> None:
    orders, _ = fn(
        PortfolioState(cash=100_000.0, last_signal_date=AS_OF),
        PICKS, PRICES, top_n=5, slippage_bps=0.0, as_of=AS_OF,
        trading_dates=DATES.to_numpy(), allow_buys=True, force=False, **extra,
    )
    assert [o for o in orders if o.action == "BUY"] == []


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_a_halted_book_stays_halted_off_schedule_and_forced(name, fn, extra) -> None:
    """Every combination of the timing guards, with risk still blocking."""
    assert _buys(fn, extra, allow_buys=False, force=True,
                 rebalance_day="Monday")[0] == []


# ---------------------------------------------------------------------------
# The report must not contradict itself
# ---------------------------------------------------------------------------


def test_the_report_never_claims_no_positions_while_listing_buys() -> None:
    """Defence in depth: if the two ever disagree again, it must be visible
    rather than a banner hiding a filled order book."""
    from stock_predictor.portfolio import Order
    from stock_predictor.predict import format_signal_report

    orders = (Order("BUY", "T00", 10, 100.0, "c1", "new_pick"),)
    text = "\n".join(format_signal_report(
        pd.DataFrame(PICKS), orders, PortfolioState(cash=100_000.0),
        nav=50_000.0, drawdown=-0.50, halted=True, top_n=5, max_drawdown=0.15,
    ))
    assert "KILL-SWITCH" in text
    assert "T00" in text, "a buy that exists must be shown, not suppressed"
    assert "no new positions" not in text or "T00" in text
