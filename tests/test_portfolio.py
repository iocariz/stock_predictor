"""Unit tests for portfolio state, orders, and risk management."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_predictor.portfolio import (
    PortfolioState,
    Position,
    active_cohort_ids,
    check_kill_switch,
    find_expiring_positions,
    generate_orders,
    held_tickers,
    init_state,
    load_state,
    portfolio_value,
    save_state,
)

# Long enough for entry-on-or-after + 10-session hold from late Jan 2024
_TD = pd.bdate_range("2024-01-02", periods=80).values


def test_init_state_defaults() -> None:
    s = init_state()
    assert s.cash == 100_000.0
    assert s.initial_capital == 100_000.0
    assert s.positions == ()
    assert s.created_at != ""


def test_save_load_roundtrip(tmp_path: Path) -> None:
    state = init_state(50_000.0)
    pos = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-24", "abc123")
    state = PortfolioState(
        initial_capital=50_000.0,
        cash=48_500.0,
        high_watermark=50_000.0,
        positions=(pos,),
        created_at=state.created_at,
    )
    p = tmp_path / "state.json"
    save_state(state, p)
    loaded = load_state(p)
    assert loaded.cash == 48_500.0
    assert len(loaded.positions) == 1
    assert loaded.positions[0].ticker == "AAPL"
    assert loaded.positions[0].shares == 10


def test_portfolio_value() -> None:
    pos = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-24", "abc")
    state = PortfolioState(cash=5_000.0, positions=(pos,))
    prices = {"AAPL": 160.0}
    assert portfolio_value(state, prices) == 5_000.0 + 10 * 160.0


def test_portfolio_value_missing_price_uses_entry() -> None:
    pos = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-24", "abc")
    state = PortfolioState(cash=5_000.0, positions=(pos,))
    assert portfolio_value(state, {}) == 5_000.0 + 10 * 150.0


def test_kill_switch_not_triggered() -> None:
    state = PortfolioState(cash=100_000.0, high_watermark=100_000.0)
    halted, nav, dd = check_kill_switch(state, {}, max_drawdown=0.15)
    assert not halted
    assert nav == 100_000.0
    assert dd == pytest.approx(0.0)


def test_kill_switch_triggered() -> None:
    state = PortfolioState(cash=80_000.0, high_watermark=100_000.0)
    halted, nav, dd = check_kill_switch(state, {}, max_drawdown=0.15)
    assert halted
    assert dd == pytest.approx(-0.20)


def test_find_expiring_positions() -> None:
    p1 = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-20", "a")
    p2 = Position("MSFT", 5, 400.0, "2024-01-10", "2024-01-25", "b")
    state = PortfolioState(positions=(p1, p2))
    expired = find_expiring_positions(state, "2024-01-20")
    assert len(expired) == 1
    assert expired[0].ticker == "AAPL"


def test_active_cohort_ids() -> None:
    p1 = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-20", "cohort_a")
    p2 = Position("MSFT", 5, 400.0, "2024-01-10", "2024-01-25", "cohort_b")
    state = PortfolioState(positions=(p1, p2))
    active = active_cohort_ids(state, "2024-01-21")
    assert active == {"cohort_b"}


def test_held_tickers() -> None:
    p1 = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-20", "a")
    p2 = Position("MSFT", 5, 400.0, "2024-01-10", "2024-01-25", "b")
    state = PortfolioState(positions=(p1, p2))
    held = held_tickers(state, "2024-01-21")
    assert held == {"MSFT"}


def test_generate_orders_sell_and_buy() -> None:
    p1 = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-20", "old_cohort")
    state = PortfolioState(
        initial_capital=100_000.0,
        cash=98_500.0,
        high_watermark=100_000.0,
        positions=(p1,),
    )
    picks = [
        {"ticker": "NVDA", "adj_close": 100.0, "prob": 0.9},
        {"ticker": "TSLA", "adj_close": 200.0, "prob": 0.8},
    ]
    prices = {"AAPL": 155.0, "NVDA": 100.0, "TSLA": 200.0}
    orders, new_state = generate_orders(
        state, picks, prices,
        top_n=2, max_cohorts=2, holding_days=10, slippage_bps=5.0, as_of="2024-01-20",
        trading_dates=_TD,
    )
    sells = [o for o in orders if o.action == "SELL"]
    buys = [o for o in orders if o.action == "BUY"]
    assert len(sells) == 1
    assert sells[0].ticker == "AAPL"
    assert len(buys) >= 1
    assert new_state.cash >= 0  # shouldn't go negative
    # AAPL should no longer be in positions
    remaining_tickers = {p.ticker for p in new_state.positions}
    assert "AAPL" not in remaining_tickers
    # 2024-01-20 is a Saturday; first session strictly after is 2024-01-22
    # (same next-day entry convention as the backtest)
    assert all(p.entry_date == "2024-01-22" for p in new_state.positions)


def test_generate_orders_no_buys_when_slots_full() -> None:
    p1 = Position("AAPL", 10, 150.0, "2024-01-10", "2024-01-25", "cohort_a")
    p2 = Position("MSFT", 5, 400.0, "2024-01-12", "2024-01-26", "cohort_b")
    state = PortfolioState(
        initial_capital=100_000.0,
        cash=90_000.0,
        high_watermark=100_000.0,
        positions=(p1, p2),
    )
    picks = [{"ticker": "NVDA", "adj_close": 100.0, "prob": 0.9}]
    prices = {"AAPL": 155.0, "MSFT": 410.0, "NVDA": 100.0}
    orders, _ = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=2, holding_days=10, slippage_bps=5.0, as_of="2024-01-20",
        trading_dates=_TD,
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert len(buys) == 0  # both slots occupied, no expirations


def test_generate_orders_excludes_held_tickers() -> None:
    p1 = Position("NVDA", 10, 100.0, "2024-01-15", "2024-01-30", "active_cohort")
    state = PortfolioState(
        initial_capital=100_000.0,
        cash=99_000.0,
        high_watermark=100_000.0,
        positions=(p1,),
    )
    picks = [
        {"ticker": "NVDA", "adj_close": 105.0, "prob": 0.95},  # already held
        {"ticker": "AAPL", "adj_close": 150.0, "prob": 0.85},
    ]
    prices = {"NVDA": 105.0, "AAPL": 150.0}
    orders, _ = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=2, holding_days=10, slippage_bps=5.0, as_of="2024-01-20",
        trading_dates=_TD,
    )
    buys = [o for o in orders if o.action == "BUY"]
    buy_tickers = {o.ticker for o in buys}
    assert "NVDA" not in buy_tickers
    assert "AAPL" in buy_tickers


def test_generate_orders_probability_weighting_dollar_split() -> None:
    """Dollar targets follow normalized probs (same rule as backtest _compute_weights)."""
    state = PortfolioState(
        initial_capital=100_000.0,
        cash=100_000.0,
        high_watermark=100_000.0,
        positions=(),
    )
    picks = [
        {"ticker": "AAA", "adj_close": 100.0, "prob": 0.9},
        {"ticker": "BBB", "adj_close": 100.0, "prob": 0.1},
    ]
    prices = {"AAA": 100.0, "BBB": 100.0}
    orders, _ = generate_orders(
        state, picks, prices,
        top_n=2, max_cohorts=2, holding_days=10, slippage_bps=0.0, as_of="2024-01-10",
        trading_dates=_TD,
        weighting="probability",
    )
    buys = [o for o in orders if o.action == "BUY"]
    by = {o.ticker: o.shares for o in buys}
    assert by["AAA"] / max(by["BBB"], 1) == pytest.approx(9.0, rel=0.05)


def test_generate_orders_commission_reduces_filled_size() -> None:
    """Per-order fees leave less cash for the buy leg → fewer whole shares."""
    p1 = Position("AAPL", 100, 50.0, "2024-01-02", "2024-01-10", "c1")
    state = PortfolioState(
        initial_capital=100_000.0,
        cash=0.0,
        high_watermark=100_000.0,
        positions=(p1,),
    )
    picks = [{"ticker": "MSFT", "adj_close": 100.0, "prob": 0.5}]
    prices = {"AAPL": 60.0, "MSFT": 100.0}
    _, st0 = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=1, holding_days=5, slippage_bps=0.0, as_of="2024-01-10",
        trading_dates=_TD,
        commission_per_share=0.0,
        commission_per_order=0.0,
    )
    _, st1 = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=1, holding_days=5, slippage_bps=0.0, as_of="2024-01-10",
        trading_dates=_TD,
        commission_per_share=0.0,
        commission_per_order=25.0,
    )
    ms0 = sum(p.shares for p in st0.positions if p.ticker == "MSFT")
    ms1 = sum(p.shares for p in st1.positions if p.ticker == "MSFT")
    assert ms0 > 0
    assert ms1 < ms0


def test_generate_orders_allow_buys_false_skips_new_positions() -> None:
    state = init_state(100_000.0)
    picks = [{"ticker": "ZZZ", "adj_close": 10.0, "prob": 0.9}]
    prices = {"ZZZ": 10.0}
    orders, new_st = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=1, holding_days=5, slippage_bps=0.0, as_of="2024-01-10",
        trading_dates=_TD,
        allow_buys=False,
    )
    assert all(o.action != "BUY" for o in orders)
    assert new_st.positions == ()


def test_generate_orders_invalid_weighting() -> None:
    state = init_state()
    with pytest.raises(ValueError, match="weighting"):
        generate_orders(
            state, [], {},
            top_n=1, max_cohorts=1, holding_days=5, slippage_bps=0.0, as_of="2024-01-10",
            trading_dates=_TD,
            weighting="invalid",
        )


# ---------------------------------------------------------------------------
# Regression: live calendar ends at as_of (the real predict-sp500 situation)
# ---------------------------------------------------------------------------


def test_generate_orders_buys_when_calendar_ends_today() -> None:
    """The downloaded price calendar ends at as_of; buys must still be generated.

    Regression: expiry used to be computed on the raw calendar, whose last
    session is as_of itself, so `holding_days` sessions ahead never existed →
    expiry was None → the buy block was silently skipped on every live run.
    """
    as_of = "2024-06-14"  # Friday, last session in the downloaded history
    td = pd.bdate_range(end=as_of, periods=300).values
    state = init_state(100_000.0)
    picks = [
        {"ticker": "NVDA", "adj_close": 100.0, "prob": 0.9},
        {"ticker": "MSFT", "adj_close": 400.0, "prob": 0.8},
    ]
    prices = {"NVDA": 100.0, "MSFT": 400.0}
    orders, new_state = generate_orders(
        state, picks, prices,
        top_n=2, max_cohorts=2, holding_days=10, slippage_bps=5.0, as_of=as_of,
        trading_dates=td,
    )
    buys = [o for o in orders if o.action == "BUY"]
    assert len(buys) == 2
    assert len(new_state.positions) == 2
    # Entry is the next session strictly after as_of (backtest parity):
    # Friday 2024-06-14 → Monday 2024-06-17.
    assert all(p.entry_date == "2024-06-17" for p in new_state.positions)
    # Expiry is holding_days business sessions after entry.
    expected_expiry = pd.bdate_range("2024-06-18", periods=10)[-1].strftime("%Y-%m-%d")
    assert all(p.expiry_date == expected_expiry for p in new_state.positions)


def test_generate_orders_entry_strictly_after_as_of() -> None:
    """When as_of is itself a session, entry must be the NEXT session (not same day)."""
    as_of = "2024-01-10"  # Wednesday, inside _TD
    state = init_state(100_000.0)
    picks = [{"ticker": "AAA", "adj_close": 100.0, "prob": 0.9}]
    prices = {"AAA": 100.0}
    _, new_state = generate_orders(
        state, picks, prices,
        top_n=1, max_cohorts=1, holding_days=5, slippage_bps=0.0, as_of=as_of,
        trading_dates=_TD,
    )
    assert len(new_state.positions) == 1
    assert new_state.positions[0].entry_date == "2024-01-11"


# ---------------------------------------------------------------------------
# Rank-hold live order generation
# ---------------------------------------------------------------------------


def _ranked(*tickers_prices) -> tuple[list[dict], dict[str, float]]:
    picks = [{"ticker": t, "adj_close": px, "prob": 1.0 - i * 0.01}
             for i, (t, px) in enumerate(tickers_prices)]
    prices = {t: px for t, px in tickers_prices}
    return picks, prices


def test_generate_orders_rank_hold_sells_decayed_and_fills_slots() -> None:
    from stock_predictor.portfolio import OPEN_ENDED_EXPIRY, generate_orders_rank_hold

    held_good = Position("AAA", 10, 100.0, "2024-01-02", OPEN_ENDED_EXPIRY, "c1")
    held_bad = Position("ZZZ", 10, 50.0, "2024-01-02", OPEN_ENDED_EXPIRY, "c1")
    state = PortfolioState(
        initial_capital=100_000.0, cash=80_000.0, high_watermark=100_000.0,
        positions=(held_good, held_bad),
    )
    # Full ranking: AAA rank 1, BBB rank 2, CCC rank 3; ZZZ absent -> rank inf.
    picks, prices = _ranked(("AAA", 100.0), ("BBB", 200.0), ("CCC", 300.0))
    prices["ZZZ"] = 55.0
    orders, new_state = generate_orders_rank_hold(
        state, picks, prices,
        top_n=2, exit_rank=2, slippage_bps=0.0, as_of="2024-06-14",
        trading_dates=pd.bdate_range(end="2024-06-14", periods=100).values,
    )
    sells = [o for o in orders if o.action == "SELL"]
    buys = [o for o in orders if o.action == "BUY"]
    assert [o.ticker for o in sells] == ["ZZZ"]
    assert sells[0].reason == "rank_exit"
    assert [o.ticker for o in buys] == ["BBB"]  # one open slot, best unheld name
    held_after = {p.ticker for p in new_state.positions}
    assert held_after == {"AAA", "BBB"}
    assert all(p.expiry_date == OPEN_ENDED_EXPIRY for p in new_state.positions
               if p.ticker == "BBB")


def test_generate_orders_rank_hold_no_churn_when_ranks_stable() -> None:
    from stock_predictor.portfolio import OPEN_ENDED_EXPIRY, generate_orders_rank_hold

    p1 = Position("AAA", 10, 100.0, "2024-01-02", OPEN_ENDED_EXPIRY, "c1")
    p2 = Position("BBB", 5, 200.0, "2024-01-02", OPEN_ENDED_EXPIRY, "c1")
    state = PortfolioState(
        initial_capital=100_000.0, cash=1_000.0, high_watermark=100_000.0,
        positions=(p1, p2),
    )
    picks, prices = _ranked(("AAA", 100.0), ("BBB", 200.0), ("CCC", 300.0))
    orders, new_state = generate_orders_rank_hold(
        state, picks, prices,
        top_n=2, exit_rank=3, slippage_bps=0.0, as_of="2024-06-14",
        trading_dates=pd.bdate_range(end="2024-06-14", periods=100).values,
    )
    assert orders == ()  # both holdings still well-ranked, no free slots
    assert {p.ticker for p in new_state.positions} == {"AAA", "BBB"}


def test_generate_orders_rank_hold_validates_exit_rank() -> None:
    from stock_predictor.portfolio import generate_orders_rank_hold

    with pytest.raises(ValueError, match="exit_rank"):
        generate_orders_rank_hold(
            init_state(), [], {},
            top_n=15, exit_rank=10, slippage_bps=0.0, as_of="2024-06-14",
            trading_dates=pd.bdate_range(end="2024-06-14", periods=10).values,
        )
