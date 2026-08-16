"""Parity checks between backtest cohort timing and live order generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import _compute_weights
from stock_predictor.execution_calendar import (
    entry_on_or_after,
    exit_date_iso_after_hold,
    extend_calendar,
    next_trading_day,
    offset_trading_days,
    trading_dates_from_index,
)
from stock_predictor.portfolio import PortfolioState, Position, generate_orders


def test_exit_iso_matches_backtest_offset() -> None:
    td = pd.bdate_range("2024-01-08", periods=40).values
    entry = entry_on_or_after("2024-01-10", td)
    assert entry == pd.Timestamp("2024-01-10")
    ex_iso = exit_date_iso_after_hold(entry, 10, td)
    ex_ts = offset_trading_days(entry, 10, td)
    assert ex_ts is not None
    assert ex_iso == ex_ts.strftime("%Y-%m-%d")


def test_trading_dates_from_index_sorted_unique() -> None:
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-03", "2024-01-05"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    out = trading_dates_from_index(df.index)
    assert len(out) == 2


def test_live_weights_match_backtest_compute_weights() -> None:
    probs = pd.Series([0.2, 0.5, 0.3]).to_numpy()
    w_eq = _compute_weights(probs, "equal")
    w_pr = _compute_weights(probs, "probability")
    assert abs(w_eq.sum() - 1.0) < 1e-9
    assert abs(w_pr.sum() - 1.0) < 1e-9
    np.testing.assert_allclose(w_pr, probs)


def test_extend_calendar_appends_business_days() -> None:
    td = pd.bdate_range("2024-06-03", end="2024-06-14").values  # ends Friday
    cal = extend_calendar(td, 10)
    assert len(cal) == len(td) + 10
    # First appended session is the next business day (Monday 2024-06-17)
    assert pd.Timestamp(cal[len(td)]) == pd.Timestamp("2024-06-17")
    # Known sessions are untouched
    assert (cal[: len(td)] == td).all()


def test_extend_calendar_empty_and_zero() -> None:
    td = pd.bdate_range("2024-06-03", periods=5).values
    assert (extend_calendar(td, 0) == td).all()
    assert len(extend_calendar(np.array([], dtype="datetime64[ns]"), 5)) == 0


def test_live_entry_matches_backtest_next_day_convention() -> None:
    """Live entry (next session strictly after as_of) equals the backtest's
    next_trading_day when the session exists in the historical calendar."""
    td = pd.bdate_range("2024-01-08", periods=40).values
    as_of = pd.Timestamp("2024-01-10")  # Wednesday, a session in td
    cal = extend_calendar(td, 15)
    live_entry = next_trading_day(as_of, cal)
    bt_entry = next_trading_day(as_of, td)
    assert live_entry == bt_entry == pd.Timestamp("2024-01-11")


# ---------------------------------------------------------------------------
# Capital deployment parity (dollars, not just calendars)
# ---------------------------------------------------------------------------

_CAL = pd.bdate_range("2024-01-02", periods=60).to_numpy()
_PRICES = {f"T{i}": 100.0 for i in range(40)}


def _picks(start: int = 0, n: int = 20) -> list[dict]:
    return [
        {"ticker": f"T{i}", "prob": 0.5, "adj_close": 100.0}
        for i in range(start, start + n)
    ]


def _open_cohort(cid: str, tickers: range, expiry: str) -> tuple[Position, ...]:
    return tuple(
        Position(f"T{i}", 50, 100.0, "2024-01-10", expiry, cid) for i in tickers
    )


def _deployed(orders) -> float:
    return sum(o.shares * o.price for o in orders if o.action == "BUY")


@pytest.mark.parametrize(
    ("max_cohorts", "n_active"),
    [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 3)],
)
def test_live_sizes_a_cohort_off_free_slots_like_the_backtest(
    max_cohorts: int, n_active: int,
) -> None:
    """Regression: live divided free cash by max_cohorts while the backtest
    divides by *free* slots, so with 1 of 2 slots open live deployed ~half the
    intended capital and the rest sat idle indefinitely."""
    cash = 60_000.0
    positions: tuple[Position, ...] = ()
    for c in range(n_active):
        positions += _open_cohort(f"coh{c}", range(c * 5, c * 5 + 5), "2124-01-25")
    state = PortfolioState(
        initial_capital=100_000.0, cash=cash,
        high_watermark=100_000.0, positions=positions,
    )

    orders, _ = generate_orders(
        state, _picks(start=30, n=10), _PRICES,
        top_n=10, max_cohorts=max_cohorts, holding_days=10,
        slippage_bps=0.0, as_of="2024-02-01", trading_dates=_CAL,
    )

    free_slots = max_cohorts - n_active
    expected = cash / free_slots  # the backtest's `cash / free_slots`
    # Integer share lots cost a little precision; 1% is well inside one lot.
    assert _deployed(orders) == pytest.approx(expected, rel=0.01)


def test_live_deploys_all_free_cash_when_one_slot_remains() -> None:
    """The concrete case from the audit: 1 of 2 slots free, $50k cash."""
    state = PortfolioState(
        initial_capital=100_000.0, cash=50_000.0, high_watermark=100_000.0,
        positions=_open_cohort("coh1", range(10), "2124-01-25"),
    )
    orders, new_state = generate_orders(
        state, _picks(start=20, n=10), _PRICES,
        top_n=10, max_cohorts=2, holding_days=10,
        slippage_bps=0.0, as_of="2024-02-01", trading_dates=_CAL,
    )
    assert _deployed(orders) == pytest.approx(50_000.0, rel=0.01)
    assert new_state.cash < 600  # not ~26k left stranded


def test_expiring_cohort_frees_its_slot_before_sizing() -> None:
    """Cash from an expiring cohort funds the replacement, and its slot counts
    as free — otherwise the portfolio ratchets down every cycle."""
    expiring = _open_cohort("old", range(10), "2024-01-31")
    state = PortfolioState(
        initial_capital=100_000.0, cash=10_000.0, high_watermark=100_000.0,
        positions=expiring + _open_cohort("live", range(10, 20), "2124-01-25"),
    )
    orders, _ = generate_orders(
        state, _picks(start=20, n=10), _PRICES,
        top_n=10, max_cohorts=2, holding_days=10,
        slippage_bps=0.0, as_of="2024-02-01", trading_dates=_CAL,
    )
    # 10k cash + 10 lots x 50 shares x $100 = 60k free, one slot open.
    assert _deployed(orders) == pytest.approx(60_000.0, rel=0.01)


def test_duplicate_holdings_opt_in_matches_backtest_basket() -> None:
    """The backtest lets a persistent name sit in two overlapping cohorts;
    live excludes held names by default, which under-weights winners."""
    held = _open_cohort("coh1", range(10), "2124-01-25")
    state = PortfolioState(
        initial_capital=100_000.0, cash=50_000.0,
        high_watermark=100_000.0, positions=held,
    )
    common = dict(
        top_n=10, max_cohorts=2, holding_days=10,
        slippage_bps=0.0, as_of="2024-02-01", trading_dates=_CAL,
    )
    # Same 10 names the existing cohort holds are top of the ranking.
    picks = _picks(start=0, n=10)

    excluded, _ = generate_orders(state, picks, _PRICES, **common)
    assert [o.ticker for o in excluded if o.action == "BUY"] == []

    duplicated, _ = generate_orders(
        state, picks, _PRICES, allow_duplicate_holdings=True, **common,
    )
    bought = [o.ticker for o in duplicated if o.action == "BUY"]
    assert bought == [f"T{i}" for i in range(10)]
    assert _deployed(duplicated) == pytest.approx(50_000.0, rel=0.01)
