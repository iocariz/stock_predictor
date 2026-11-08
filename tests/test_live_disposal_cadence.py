"""A delisted name is disposed when its grace expires, not at the next rebalance.

``generate_orders_long_short`` returns early on a quiet session:

    if not due:
        return (), quiet

and the disposal sweep sits *after* that line. So a position with no quote can
only ever be disposed on a rebalance session. With ``rebalance_every`` at 63
that quantises the grace period to 63-session boundaries: a name whose grace
expires one session after a turnover waits another 63, for an effective grace
of up to 126 sessions.

Observed live. WBD last printed 2026-10-05 and stopped; Yahoo returns one row
and nothing after. At 3 sessions unpriced it sits as a -38 share short marked
at a frozen $30.95. The next rebalance is 2026-12-04, roughly 38 sessions out,
by which point it has ~41 sessions unpriced against a 63-session grace -- so it
is deferred, and waits for the rebalance after that. Around five months
carrying an open short in a security that no longer trades.

This is the same lesson the borrow clock already learned in this file: carry
accrues on every session the leg is open, "whether or not today is a
rebalance", because gating a per-session quantity on the turnover calendar
misstates it. Disposal is the same shape -- the grace period is counted in
sessions, so it has to be *checked* in sessions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.delisting import DelistingPolicy
from stock_predictor.portfolio import (
    PortfolioState,
    Position,
    init_state,
)
from stock_predictor.portfolio import generate_orders_long_short as gen

DATES = pd.bdate_range("2024-01-01", periods=200)
SESSIONS = DATES.to_numpy()
N = 20
TICKERS = [f"T{i:02d}" for i in range(N)]


def _picks(order: list[str] | None = None) -> list[dict]:
    names = order or TICKERS
    return [{"ticker": t, "prob": float(len(names) - i)}
            for i, t in enumerate(names)]


def _prices(px: float = 100.0, drop: set[str] | None = None) -> dict[str, float]:
    return {t: px for t in TICKERS if t not in (drop or set())}


def _cfg(**kw):
    base = dict(decile=0.25, long_weight=0.5, short_weight=0.5,
                rebalance_every=63, slippage_bps=0.0, min_names_per_side=2,
                trading_dates=SESSIONS)
    base.update(kw)
    return base


def _run(state, picks, prices, as_of, **kw):
    return gen(state, picks, prices, as_of=as_of, **_cfg(**kw))


def _open_book():
    """A book opened on session 0, so the next rebalance is 63 sessions out."""
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    return st


def _age(state: PortfolioState, ticker: str, sessions: int) -> PortfolioState:
    """Hand-age one position's unpriced counter, as sessions of no quote would."""
    return PortfolioState(
        initial_capital=state.initial_capital,
        cash=state.cash,
        high_watermark=state.high_watermark,
        positions=tuple(
            Position(
                ticker=p.ticker, shares=p.shares, entry_price=p.entry_price,
                entry_date=p.entry_date, expiry_date=p.expiry_date,
                cohort_id=p.cohort_id, last_price=p.last_price,
                sessions_unpriced=sessions if p.ticker == ticker
                else p.sessions_unpriced,
            )
            for p in state.positions
        ),
        created_at=state.created_at,
        updated_at=state.updated_at,
        history=state.history,
        last_signal_date=state.last_signal_date,
        last_accrual_date=state.last_accrual_date,
    )


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_an_expired_grace_is_disposed_on_a_quiet_session() -> None:
    """The live failure. Nothing should wait 63 sessions for a turnover to
    notice a grace period that has already run out."""
    st = _open_book()
    gone = next(p.ticker for p in st.positions)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=5)
    st = _age(st, gone, 6)          # grace already exceeded
    dark = _prices(drop={gone})

    _, after = _run(st, _picks(), dark, str(DATES[3].date()),
                    delisting_policy=policy)
    assert all(p.ticker != gone for p in after.positions), (
        "an expired position survived a quiet session")


def test_a_position_still_inside_grace_is_retained() -> None:
    """Disposal must not become eager. A name that stops printing for a day is
    not delisted, and writing it off early realises a loss that did not
    happen."""
    st = _open_book()
    gone = next(p.ticker for p in st.positions)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=5)
    st = _age(st, gone, 1)
    dark = _prices(drop={gone})

    _, after = _run(st, _picks(), dark, str(DATES[3].date()),
                    delisting_policy=policy)
    assert any(p.ticker == gone for p in after.positions), (
        "a position inside its grace period was disposed")


def test_disposing_on_a_quiet_session_does_not_rebalance_the_book() -> None:
    """Only the dead name moves. A disposal is not a reason to turn the book
    over 60 sessions early."""
    st = _open_book()
    gone = next(p.ticker for p in st.positions)
    before = {p.ticker: p.shares for p in st.positions if p.ticker != gone}
    policy = DelistingPolicy(fallback="write_off", grace_sessions=5)
    st = _age(st, gone, 6)

    orders, after = _run(st, _picks(), _prices(drop={gone}),
                         str(DATES[3].date()), delisting_policy=policy)
    assert all(o.ticker == gone for o in orders), (
        f"the book rebalanced on a disposal: {[o.ticker for o in orders]}")
    assert {p.ticker: p.shares for p in after.positions if p.ticker != gone} == before


def test_a_disposed_short_settles_at_its_mark_not_at_zero() -> None:
    """Covering a short for nothing is the maximum profit, not a conservative
    estimate. WBD is a short, so this is the live case."""
    st = _open_book()
    short = next(p for p in st.positions if p.shares < 0)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=5)
    st = _age(st, short.ticker, 6)

    _, after = _run(st, _picks(), _prices(drop={short.ticker}),
                    str(DATES[3].date()), delisting_policy=policy)
    # Settling a short costs cash: the liability is bought back at its mark.
    assert after.cash < st.cash, "covering the short did not cost anything"
    expected = st.cash + short.shares * short.last_price
    assert after.cash == pytest.approx(expected, rel=1e-6)


def test_the_rebalance_path_still_disposes() -> None:
    """Regression guard: the disposal that already worked on a turnover must
    keep working. Forced so the rebalance branch is the one under test."""
    st = _open_book()
    gone = next(p.ticker for p in st.positions)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=200)
    dark = _prices(drop={gone})
    # Grace far longer than the test, so only the forced rebalance can act.
    st = _age(st, gone, 201)
    _, st = _run(st, _picks(), dark, str(DATES[5].date()),
                 delisting_policy=policy, force=True)
    assert all(p.ticker != gone for p in st.positions)


def test_a_quiet_disposal_is_reported(capsys) -> None:
    """disposed and deferred were collected and never printed, so the live
    book could write a name off in silence. A disposal moves real cash."""
    st = _open_book()
    gone = next(p.ticker for p in st.positions)
    policy = DelistingPolicy(fallback="write_off", grace_sessions=5)
    st = _age(st, gone, 6)
    _run(st, _picks(), _prices(drop={gone}), str(DATES[3].date()),
         delisting_policy=policy)
    out = capsys.readouterr().out
    assert gone in out, f"disposal of {gone} was silent: {out!r}"
