"""A missing quote is not a fill.

Both live order generators priced exits with `prices.get(ticker, entry_price)`.
When a holding had no quote the exit executed **at the entry price** — not even
at the `last_price` the position already carries — the position was removed, and
the cash was credited.

A deterministic probe: 10 shares, entry $100, last observed $40, no current
quote produced `SELL 10 @ $100.00`, crediting **$1,000** for something last seen
worth $400.

Reachable because the live download covers the model universe intersected with
current index membership, so a holding that *left* the index is not downloaded
and therefore has no quote. That intersection was introduced to fix universe
reproduction; it narrowed the download and widened this hole.

Without a valid quote the exit is now deferred and the position retained. You
still own it; the cash is not real until the fill is.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.portfolio import (
    PortfolioState,
    Position,
    generate_orders,
    generate_orders_rank_hold,
)

DATES = pd.bdate_range(end="2024-06-14", periods=200)
AS_OF = "2024-06-14"
PICKS = [{"ticker": "AAA", "prob": 0.9, "adj_close": 50.0}]
QUOTED = {"AAA": 50.0}

MODES = [
    ("fixed", generate_orders, dict(max_cohorts=2, holding_days=10)),
    ("rank", generate_orders_rank_hold, dict(exit_rank=30)),
]


def _held(**kw) -> Position:
    base = dict(ticker="DEAD", shares=10, entry_price=100.0,
                entry_date="2024-01-02", expiry_date=AS_OF, cohort_id="c1",
                last_price=40.0)
    base.update(kw)
    return Position(**base)


def _run(fn, extra, prices, position=None, allow_buys=True):
    """allow_buys=False isolates the sale: proceeds are otherwise reinvested,
    so final cash would not show what the exit actually credited."""
    state = PortfolioState(cash=0.0, positions=(position or _held(),))
    return fn(state, PICKS, prices, top_n=1, slippage_bps=0.0, as_of=AS_OF,
              trading_dates=DATES.to_numpy(), allow_buys=allow_buys, **extra)


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_no_quote_means_no_sale(name, fn, extra) -> None:
    orders, state = _run(fn, extra, QUOTED)
    assert [o for o in orders if o.ticker == "DEAD"] == [], (
        f"{name}: sold a holding it could not price"
    )
    assert state.cash == 0.0, f"{name}: credited cash for a fill that never happened"


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_the_position_is_retained_not_discarded(name, fn, extra) -> None:
    """You still own it. Dropping it would lose the position entirely."""
    _, state = _run(fn, extra, QUOTED)
    assert [p.ticker for p in state.positions] == ["DEAD"]


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_the_operator_is_told(name, fn, extra, capsys) -> None:
    _run(fn, extra, QUOTED)
    out = capsys.readouterr().out
    assert "DEAD" in out and "defer" in out.lower()


# ---------------------------------------------------------------------------
# A real quote still sells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_a_quoted_holding_exits_at_that_quote(name, fn, extra) -> None:
    orders, state = _run(fn, extra, {**QUOTED, "DEAD": 40.0}, allow_buys=False)
    sells = [o for o in orders if o.ticker == "DEAD"]
    assert len(sells) == 1
    assert sells[0].price == pytest.approx(40.0)
    assert state.cash == pytest.approx(400.0)
    assert "DEAD" not in [p.ticker for p in state.positions]


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_a_quote_far_below_the_entry_is_still_honoured(name, fn, extra) -> None:
    """The point of the fix is not to protect the book from losses."""
    _, state = _run(fn, extra, {**QUOTED, "DEAD": 1.0}, allow_buys=False)
    assert state.cash == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# What counts as a quote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), None])
def test_a_degenerate_quote_is_not_a_quote(name, fn, extra, bad) -> None:
    """Zero is a placeholder, not a price, and NaN divides badly everywhere."""
    orders, state = _run(fn, extra, {**QUOTED, "DEAD": bad})
    assert [o for o in orders if o.ticker == "DEAD"] == []
    assert state.cash == 0.0


@pytest.mark.parametrize(("name", "fn", "extra"), MODES)
def test_last_price_is_not_used_as_a_substitute_for_a_quote(name, fn, extra) -> None:
    """last_price is for *marking*, which is an estimate. A fill is not."""
    _, state = _run(fn, extra, QUOTED, position=_held(last_price=40.0))
    assert state.cash == 0.0, "marked at 40 is not the same as sold at 40"


# ---------------------------------------------------------------------------
# Other holdings are unaffected
# ---------------------------------------------------------------------------


def test_one_unpriceable_holding_does_not_block_the_others() -> None:
    state = PortfolioState(cash=0.0, positions=(
        _held(),
        _held(ticker="LIVE", cohort_id="c2"),
    ))
    orders, new = generate_orders(
        state, PICKS, {**QUOTED, "LIVE": 60.0}, top_n=1, max_cohorts=2,
        holding_days=10, slippage_bps=0.0, as_of=AS_OF,
        trading_dates=DATES.to_numpy(), allow_buys=False,
    )
    sold = {o.ticker for o in orders if o.action == "SELL"}
    assert sold == {"LIVE"}
    assert new.cash == pytest.approx(600.0)
    assert {p.ticker for p in new.positions} == {"DEAD"}
