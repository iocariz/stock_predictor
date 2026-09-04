"""The engine with the measurable tilt could not be traded.

``predict-sp500 --hold-mode`` accepted ``fixed`` and ``rank`` — the two
long-only engines, whose alpha is indistinguishable from zero in every draw and
whose measured drawdowns (−45.94% and −51.90%) are three times the 15% kill
switch they run under. The long-short book is the only engine whose alpha
survives its own noise (t +2.44…+2.94 across four rebuilds) and the only one
whose drawdown (−13.03%) fits that switch, and it had no live path at all.

So the live configuration was not merely mis-set; the thing worth setting it to
did not exist.

This adds it. The rules are the backtest's rules, sharing ``_target_book`` and
``CostModel`` rather than reimplementing them, because a live path that costs
less than the simulation is the whole problem this project keeps rediscovering.

Two structural differences from the long-only paths, and the tests below exist
mostly to pin them:

* **Shares go negative.** A short is a liability, not an absent position, and
  ``portfolio_value`` has to fall when a short rises.
* **There is no per-position expiry.** The book rebalances on a calendar, so
  positions carry the open-ended sentinel and are closed by the next
  rebalance's target, never by the fixed-expiry sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.portfolio import (
    OPEN_ENDED_EXPIRY,
    PortfolioState,
    Position,
    init_state,
    portfolio_value,
)
from stock_predictor.portfolio import generate_orders_long_short as gen

DATES = pd.bdate_range("2024-01-01", periods=200)
SESSIONS = DATES.to_numpy()
N = 20
TICKERS = [f"T{i:02d}" for i in range(N)]


def _picks(order: list[str] | None = None) -> list[dict]:
    """Full universe scored best-first."""
    names = order or TICKERS
    return [{"ticker": t, "prob": float(len(names) - i)}
            for i, t in enumerate(names)]


def _prices(px: float = 100.0, **over) -> dict[str, float]:
    p = {t: px for t in TICKERS}
    p.update(over)
    return p


def _cfg(**kw):
    base = dict(decile=0.25, long_weight=0.5, short_weight=0.5,
                rebalance_every=21, slippage_bps=0.0, min_names_per_side=2,
                trading_dates=SESSIONS)
    base.update(kw)
    return base


def _run(state, picks, prices, as_of, **kw):
    return gen(state, picks, prices, as_of=as_of, **_cfg(**kw))


# ---------------------------------------------------------------------------
# Opening a book
# ---------------------------------------------------------------------------


def test_the_first_rebalance_opens_both_sides() -> None:
    orders, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    assert orders, "no orders on an empty book"
    longs = [p for p in st.positions if p.shares > 0]
    shorts = [p for p in st.positions if p.shares < 0]
    assert longs and shorts, f"{len(longs)} long, {len(shorts)} short"


def test_the_best_ranked_names_are_bought_and_the_worst_sold_short() -> None:
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    held = {p.ticker: p.shares for p in st.positions}
    assert held.get("T00", 0) > 0, "top-ranked name is not long"
    assert held.get(f"T{N - 1:02d}", 0) < 0, "worst-ranked name is not short"


def test_gross_exposure_matches_the_configured_weights() -> None:
    _, st = _run(init_state(), _picks(), _prices(),
                 "2024-01-01", long_weight=0.5, short_weight=0.5)
    nav = 100_000.0
    long_notional = sum(p.shares * 100.0 for p in st.positions if p.shares > 0)
    short_notional = -sum(p.shares * 100.0 for p in st.positions if p.shares < 0)
    assert long_notional == pytest.approx(0.5 * nav, rel=0.02)
    assert short_notional == pytest.approx(0.5 * nav, rel=0.02)


def test_positions_carry_the_open_ended_expiry() -> None:
    """The fixed-expiry sweep must never touch this book."""
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    assert all(p.expiry_date == OPEN_ENDED_EXPIRY for p in st.positions)


# ---------------------------------------------------------------------------
# A short is a liability
# ---------------------------------------------------------------------------


def test_nav_falls_when_a_short_rises() -> None:
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    short = next(p for p in st.positions if p.shares < 0)
    before = portfolio_value(st, _prices())
    after = portfolio_value(st, _prices(**{short.ticker: 130.0}))
    assert after < before, "a rising short did not cost anything"


def test_nav_rises_when_a_short_falls() -> None:
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    short = next(p for p in st.positions if p.shares < 0)
    before = portfolio_value(st, _prices())
    after = portfolio_value(st, _prices(**{short.ticker: 70.0}))
    assert after > before


def test_opening_the_book_does_not_invent_money() -> None:
    """Short proceeds raise cash and create an equal liability; NAV must not
    move on the opening trade beyond costs."""
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    assert portfolio_value(st, _prices()) == pytest.approx(100_000.0, rel=1e-9)


def test_short_proceeds_reach_cash() -> None:
    """With balanced weights the long spend and the short credit cancel, so
    cash sitting at its opening figure proves nothing on its own. The invariant
    is that cash moved by *both* legs: buying without crediting the short would
    leave it at 50,000."""
    _, st = _run(init_state(), _picks(), _prices(), "2024-01-01")
    longs = sum(p.shares * 100.0 for p in st.positions if p.shares > 0)
    shorts = -sum(p.shares * 100.0 for p in st.positions if p.shares < 0)
    assert longs > 0 and shorts > 0, "one leg is missing"
    assert st.cash == pytest.approx(100_000.0 - longs + shorts, abs=1.0)
    assert st.cash > 100_000.0 - longs, "short proceeds were not credited"


# ---------------------------------------------------------------------------
# The rebalance calendar
# ---------------------------------------------------------------------------


def test_no_orders_between_rebalances() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    orders, _ = _run(st, _picks(), _prices(), str(DATES[5].date()))
    assert orders == ()


def test_the_book_rebalances_after_the_configured_interval() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    flipped = list(reversed(TICKERS))
    orders, st2 = _run(st, _picks(flipped), _prices(), str(DATES[21].date()))
    assert orders, "no rebalance at the interval"
    held = {p.ticker: p.shares for p in st2.positions}
    assert held.get("T00", 0) < 0, "the newly worst name is not short"


def test_force_overrides_the_calendar() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    orders, _ = _run(st, _picks(list(reversed(TICKERS))), _prices(),
                     str(DATES[3].date()), force=True)
    assert orders


def test_rerunning_one_session_does_not_trade_twice() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    orders, _ = _run(st, _picks(), _prices(), str(DATES[0].date()))
    assert orders == (), "the same signal opened a second book"


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_borrow_accrues_on_the_short_leg() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()),
                 short_borrow_annual=0.10)
    _, later = _run(st, _picks(), _prices(), str(DATES[5].date()),
                    short_borrow_annual=0.10)
    assert later.cash < st.cash, "no borrow charged on an open short book"


def test_no_borrow_when_the_rate_is_zero() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    _, later = _run(st, _picks(), _prices(), str(DATES[5].date()))
    assert later.cash == pytest.approx(st.cash, rel=0, abs=1e-9)


def test_slippage_costs_the_book_money() -> None:
    _, free = _run(init_state(), _picks(), _prices(), "2024-01-01")
    _, charged = _run(init_state(), _picks(), _prices(), "2024-01-01",
                      slippage_bps=25.0)
    assert (portfolio_value(charged, _prices())
            < portfolio_value(free, _prices()))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_name_without_a_quote_is_not_traded() -> None:
    prices = _prices()
    del prices["T00"]
    _, st = _run(init_state(), _picks(), prices, "2024-01-01")
    assert all(p.ticker != "T00" for p in st.positions)


def test_a_zero_price_is_not_a_price() -> None:
    _, st = _run(init_state(), _picks(), _prices(T00=0.0), "2024-01-01")
    assert all(p.ticker != "T00" for p in st.positions)


def test_a_thin_cross_section_opens_nothing() -> None:
    """Fewer names than the minimum per side is not a book."""
    orders, st = _run(init_state(), _picks(TICKERS[:4]), _prices(),
                      "2024-01-01", decile=0.25, min_names_per_side=3)
    assert orders == ()
    assert st.positions == ()


def test_allow_new_false_closes_without_opening() -> None:
    """The halted path: unwind, do not re-enter."""
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    assert st.positions
    orders, flat = _run(st, _picks(), _prices(), str(DATES[21].date()),
                        allow_new=False)
    assert orders, "nothing was unwound"
    assert flat.positions == (), "the book did not go flat"


# ---------------------------------------------------------------------------
# Parity with the backtest
# ---------------------------------------------------------------------------


def test_the_target_book_is_the_backtest_function() -> None:
    """Not a reimplementation. Two copies of a sizing rule is how the live path
    and the simulation drift apart."""
    import inspect

    from stock_predictor import portfolio

    assert "target_book" in inspect.getsource(portfolio.generate_orders_long_short)


def test_a_flat_book_marks_at_cash() -> None:
    st = PortfolioState(initial_capital=100_000.0, cash=100_000.0,
                        high_watermark=100_000.0, positions=())
    assert portfolio_value(st, _prices()) == pytest.approx(100_000.0)


def test_an_existing_short_is_covered_when_it_leaves_the_book() -> None:
    st = PortfolioState(
        initial_capital=100_000.0, cash=110_000.0, high_watermark=100_000.0,
        positions=(Position(ticker="T19", shares=-100, entry_price=100.0,
                            entry_date="2024-01-01",
                            expiry_date=OPEN_ENDED_EXPIRY,
                            cohort_id="ls", last_price=100.0),),
        last_signal_date=str(DATES[0].date()),
    )
    # T19 now ranks best, so it must flip from short to long.
    orders, out = _run(st, _picks(["T19", *TICKERS[:-1]]), _prices(),
                       str(DATES[21].date()))
    held = {p.ticker: p.shares for p in out.positions}
    assert held.get("T19", 0) > 0, f"short not covered and flipped: {held.get('T19')}"
    assert any(o.ticker == "T19" and o.action == "BUY" for o in orders)


def test_integer_shares_only() -> None:
    _, st = _run(init_state(), _picks(), _prices(px=137.0), "2024-01-01")
    assert all(float(p.shares).is_integer() for p in st.positions)
    assert all(isinstance(p.shares, (int, np.integer)) for p in st.positions)


# ---------------------------------------------------------------------------
# Parity with the backtest, on a synthetic panel
# ---------------------------------------------------------------------------


def _panel_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, (len(DATES), N)), axis=0))
    rows = []
    for di, d in enumerate(DATES):
        order = np.random.default_rng(500 + di).permutation(N)
        for i, t in enumerate(TICKERS):
            rows.append({"date": d, "ticker": t, "prob": float(order[i]),
                         "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


def test_the_live_book_tracks_the_backtest() -> None:
    """The check that matters, and the one that found three bugs.

    A live path that does not reproduce its own simulation is measuring a
    different strategy from the one that was validated. Run against the real
    baseline this caught a borrow charge billed as 1+2+...+n instead of n
    (−20 points), share sizing truncated toward zero instead of rounded
    (−9 points), and delisted names carried forever instead of disposed. The
    residual is whole-share sizing against the backtest's fractional book,
    which is a real difference between an account and a simulation rather than
    a defect.
    """
    from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

    panel = _panel_frame()
    execution = panel.pivot_table(index="date", columns="ticker",
                                  values="adj_close", aggfunc="first")
    cfg = LongShortConfig(
        decile=0.25, long_weight=0.5, short_weight=0.5, rebalance_every=21,
        slippage_bps=5.0, benchmark_ticker=None, risk_free_rate=0.0,
        min_names_per_side=2, short_borrow_annual=0.02, reject_stale_fills=True,
    )
    bt = run_long_short_backtest(panel, cfg, execution_prices=execution)

    st = init_state(100_000.0)
    for d in DATES:
        prices = {t: float(v) for t, v in execution.loc[d].items()
                  if pd.notna(v) and v > 0}
        picks = (panel[panel["date"] == d]
                 .sort_values("prob", ascending=False)[["ticker", "prob"]]
                 .to_dict("records"))
        _, st = gen(st, picks, prices, as_of=str(d.date()),
                    decile=0.25, long_weight=0.5, short_weight=0.5,
                    rebalance_every=21, slippage_bps=5.0, min_names_per_side=2,
                    short_borrow_annual=0.02, trading_dates=SESSIONS)

    final = {t: float(v) for t, v in execution.iloc[-1].items()
             if pd.notna(v) and v > 0}
    live = portfolio_value(st, final)
    sim = float(bt.daily_nav.iloc[-1])
    assert live == pytest.approx(sim, rel=0.05), (
        f"live {live:,.0f} vs backtest {sim:,.0f} "
        f"({live / sim - 1:+.2%}) — beyond whole-share sizing")


def test_borrow_is_charged_once_per_session() -> None:
    """The 20-point bug, pinned. Accruing from the last *rebalance* bills
    1+2+...+n over an n-session hold instead of n."""
    _, opened = _run(init_state(), _picks(), _prices(), str(DATES[0].date()),
                     short_borrow_annual=0.10)
    st = opened
    for i in range(1, 11):
        _, st = _run(st, _picks(), _prices(), str(DATES[i].date()),
                     short_borrow_annual=0.10)
    short_notional = -sum(p.shares * 100.0 for p in opened.positions
                          if p.shares < 0)
    expected = short_notional * 0.10 / 252 * 10
    assert (opened.cash - st.cash) == pytest.approx(expected, rel=0.02)


def test_an_unexitable_position_is_disposed_not_carried_forever() -> None:
    """A live book that cannot sell a delisted name must not hold it
    indefinitely. Against the real baseline this accumulated eight names."""
    from stock_predictor.delisting import DelistingPolicy

    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    gone = next(p.ticker for p in st.positions if p.shares < 0)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=2)
    dark = _prices()
    del dark[gone]
    for i in range(1, 30):
        _, st = _run(st, _picks(), dark, str(DATES[i].date()),
                     delisting_policy=policy)
    _, st = _run(st, _picks(), dark, str(DATES[21].date()),
                 delisting_policy=policy, force=True)
    assert all(p.ticker != gone for p in st.positions), "never disposed"
