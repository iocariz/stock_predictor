"""Live-execution safety: scheduling, idempotency, stale prices, cash (P1-3/4, P2-5).

None of these touch backtest numbers. Together they decide whether
predict-sp500 trades the strategy the backtest simulates, or a different one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.portfolio import (
    PortfolioState,
    Position,
    check_kill_switch,
    generate_orders,
    init_state,
    portfolio_value,
    stale_positions,
)

CAL = pd.bdate_range("2024-01-02", periods=90).to_numpy()
PICKS = [{"ticker": f"T{i}", "prob": 0.9 - i / 100, "adj_close": 100.0} for i in range(4)]
PRICES = {f"T{i}": 100.0 for i in range(4)}
KW = dict(top_n=2, max_cohorts=2, holding_days=10, slippage_bps=0.0,
          trading_dates=CAL)


def _buys(orders):
    return [o for o in orders if o.action == "BUY"]


# ---------------------------------------------------------------------------
# P1-3: idempotency and schedule
# ---------------------------------------------------------------------------


def test_rerunning_the_same_signal_day_opens_nothing_new() -> None:
    """Regression: every confirmed run opened a cohort, so a daily cron built
    five cohorts a week against a backtest that models one."""
    st = init_state(100_000.0)
    o1, st1 = generate_orders(st, PICKS, PRICES, as_of="2024-02-01", **KW)
    o2, st2 = generate_orders(st1, PICKS, PRICES, as_of="2024-02-01", **KW)
    assert len(_buys(o1)) == 2
    assert _buys(o2) == [], "the same signal must not be acted on twice"
    assert len(st2.positions) == len(st1.positions)


def test_the_signal_date_is_recorded_on_the_state() -> None:
    st = init_state(100_000.0)
    _, st1 = generate_orders(st, PICKS, PRICES, as_of="2024-02-01", **KW)
    assert st1.last_signal_date == "2024-02-01"


def test_a_later_signal_day_may_open_a_cohort() -> None:
    st = init_state(100_000.0)
    _, st1 = generate_orders(st, PICKS, PRICES, as_of="2024-02-01", **KW)
    o2, _ = generate_orders(st1, PICKS, PRICES, as_of="2024-02-08", **KW)
    assert len(_buys(o2)) > 0


def test_rebalance_day_gate_matches_the_backtest_schedule() -> None:
    """The backtest trades one signal per ISO week on --rebalance-day."""
    st = init_state(100_000.0)
    # 2024-02-01 is a Thursday.
    o_thu, _ = generate_orders(st, PICKS, PRICES, as_of="2024-02-01",
                               rebalance_day="Friday", **KW)
    o_fri, _ = generate_orders(st, PICKS, PRICES, as_of="2024-02-02",
                               rebalance_day="Friday", **KW)
    assert _buys(o_thu) == [], "off-schedule runs must not open positions"
    assert len(_buys(o_fri)) == 2


def test_force_overrides_both_gates() -> None:
    st = init_state(100_000.0)
    _, st1 = generate_orders(st, PICKS, PRICES, as_of="2024-02-01", **KW)
    o2, _ = generate_orders(st1, PICKS, PRICES, as_of="2024-02-01",
                            rebalance_day="Friday", force=True, **KW)
    assert len(_buys(o2)) == 2


def test_expiries_still_settle_on_an_off_schedule_run() -> None:
    """Sells are date-driven and must never be gated — a halted or
    off-schedule run still has to liquidate what expired."""
    pos = (Position("T0", 10, 100.0, "2024-01-10", "2024-01-30", "old"),)
    st = PortfolioState(initial_capital=100_000.0, cash=50_000.0,
                        high_watermark=100_000.0, positions=pos)
    orders, new = generate_orders(st, PICKS, PRICES, as_of="2024-02-01",
                                  rebalance_day="Friday", **KW)
    assert [o.action for o in orders] == ["SELL"]
    assert new.positions == ()


# ---------------------------------------------------------------------------
# P1-4: stale prices must not be valued at cost
# ---------------------------------------------------------------------------


def test_an_unpriced_holding_is_not_marked_at_cost() -> None:
    """Regression: a delisted or failed-download holding was valued at its
    entry price, so it could never register a loss and the kill-switch could
    never fire on it."""
    pos = (Position("GONE", 100, 50.0, "2024-01-10", "2124-01-30", "c",
                    last_price=5.0),)
    st = PortfolioState(initial_capital=100_000.0, cash=0.0,
                        high_watermark=100_000.0, positions=pos)
    nav = portfolio_value(st, {})
    assert nav == pytest.approx(500.0), "must use the last observed price, not entry"


def test_live_price_beats_last_price_beats_entry() -> None:
    pos = (Position("A", 10, 50.0, "2024-01-10", "2124-01-30", "c", last_price=30.0),)
    st = PortfolioState(initial_capital=1_000.0, cash=0.0,
                        high_watermark=1_000.0, positions=pos)
    assert portfolio_value(st, {"A": 20.0}) == pytest.approx(200.0)
    assert portfolio_value(st, {}) == pytest.approx(300.0)
    never = (Position("A", 10, 50.0, "2024-01-10", "2124-01-30", "c"),)
    st2 = PortfolioState(initial_capital=1_000.0, cash=0.0,
                         high_watermark=1_000.0, positions=never)
    assert portfolio_value(st2, {}) == pytest.approx(500.0), "entry is the last resort"


def test_stale_holdings_are_reported() -> None:
    pos = (Position("A", 10, 50.0, "2024-01-10", "2124-01-30", "c"),
           Position("B", 10, 50.0, "2024-01-10", "2124-01-30", "c"))
    st = PortfolioState(initial_capital=1_000.0, cash=0.0,
                        high_watermark=1_000.0, positions=pos)
    assert stale_positions(st, {"A": 10.0}) == ("B",)
    assert stale_positions(st, {"A": 10.0, "B": 10.0}) == ()


def test_the_kill_switch_sees_a_collapsed_holding() -> None:
    """The point of the fix: a position that fell 90% must trip the switch."""
    pos = (Position("GONE", 1000, 100.0, "2024-01-10", "2124-01-30", "c",
                    last_price=10.0),)
    st = PortfolioState(initial_capital=100_000.0, cash=0.0,
                        high_watermark=100_000.0, positions=pos)
    halted, nav, dd = check_kill_switch(st, {}, 0.15)
    assert nav == pytest.approx(10_000.0)
    assert dd < -0.15
    assert halted, "a 90% loss must halt, not hide behind entry price"


def test_last_price_is_refreshed_when_a_quote_is_available() -> None:
    st = init_state(100_000.0)
    _, st1 = generate_orders(st, PICKS, PRICES, as_of="2024-02-01", **KW)
    _, st2 = generate_orders(st1, PICKS, {"T0": 80.0, "T1": 80.0},
                             as_of="2024-02-08", **KW)
    held = {p.ticker: p.last_price for p in st2.positions}
    assert held["T0"] == pytest.approx(80.0), "state must carry the newest quote"


# ---------------------------------------------------------------------------
# P2-5: commissions must not overdraw cash
# ---------------------------------------------------------------------------


def test_commissions_cannot_drive_cash_negative() -> None:
    """Regression: sizing ignored fees, so $1,000 cash with $100/order fees
    produced -$200."""
    st = PortfolioState(initial_capital=1_000.0, cash=1_000.0, high_watermark=1_000.0)
    orders, new = generate_orders(
        st, PICKS, PRICES, as_of="2024-02-01",
        commission_per_order=100.0,
        **{**KW, "max_cohorts": 1},
    )
    assert new.cash >= 0, f"overdrawn: {new.cash}"
    spend = sum(o.shares * o.price for o in _buys(orders)) + 100.0 * len(_buys(orders))
    assert spend <= 1_000.0 + 1e-6


def test_per_share_commissions_are_also_budgeted() -> None:
    st = PortfolioState(initial_capital=1_000.0, cash=1_000.0, high_watermark=1_000.0)
    _, new = generate_orders(
        st, PICKS, PRICES, as_of="2024-02-01",
        commission_per_share=5.0, **{**KW, "max_cohorts": 1},
    )
    assert new.cash >= 0


# ---------------------------------------------------------------------------
# P1-2: a fundamentals-trained model must be able to score
# ---------------------------------------------------------------------------


def test_inference_panel_accepts_fundamentals() -> None:
    """Regression: training supported fundamental columns but inference never
    passed them, so scoring raised on missing features."""
    import inspect
    from stock_predictor.predict import build_inference_panel, main

    assert "fundamentals" in inspect.signature(build_inference_panel).parameters
    src = inspect.getsource(main)
    assert "fetch_fundamentals" in src, "main must source them"
    assert 'startswith("fund_")' in src, "and only when the model needs them"


def test_predict_cli_exposes_the_schedule_gates() -> None:
    from unittest.mock import patch
    from stock_predictor.predict import parse_args

    with patch("sys.argv", ["predict-sp500", "--model", "m.pkl"]):
        a = parse_args()
    assert a.rebalance_day is None
    assert a.force_rebalance is False
    with patch("sys.argv", ["predict-sp500", "--model", "m.pkl",
                            "--rebalance-day", "Friday", "--force-rebalance"]):
        b = parse_args()
    assert b.rebalance_day == "Friday"
    assert b.force_rebalance is True
