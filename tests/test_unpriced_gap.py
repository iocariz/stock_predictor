"""A gap is a duration, and it has to be measured from the last quote.

The long-short engine counts how long a holding has been unpriced so the
delisting policy can tell a vendor hiccup from a delisting. It built that
counter as::

    _last_priced = actual.mul(_pos, axis=0).where(actual).cummax()

``cummax`` skips NaN but does not fill it, so ``_last_priced`` is NaN on
exactly the rows where a quote is missing -- which is the only situation the
disposal branch ever reads it in. The NaN test therefore always fired, and the
gap always came out as ``i + 1``: sessions since the **start of the backtest**,
not since the last quote.

The consequence is not subtle. A name that printed every session for a year and
then misses one is treated as having been silent for a year, so any grace period
is already exhausted and the position is written off to zero on the spot.
Reproduced below: twelve names, flat prices, one absent session, a three-session
grace period, and NAV falls from $100,000 to $83,333 with
``written_off = 1, deferred = 0``.

``specs.md:181`` is explicit that a gap alone never proves a delisting. The
grace period exists to encode that, and this defect skipped it entirely.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.delisting import DelistingPolicy
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

DATES = pd.bdate_range("2024-01-01", periods=120)
N = 12
DARK = "T00"
"""Ranked best, so it is held on the long side."""

FILL_SESSION = 61
"""A rebalance signals every 20 sessions and fills on the next one, so this is
a session the engine actually tries to trade -- the only time it can discover
that a holding cannot be exited."""


def _panel(missing: set[int]) -> pd.DataFrame:
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            if t == DARK and di in missing:
                continue
            rows.append({"date": d, "ticker": t, "prob": float(N - i),
                         "adj_close": 100.0})
    return pd.DataFrame(rows)


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


def _run(missing: set[int], *, grace: int = 3, fallback: str = "write_off"):
    panel = _panel(missing)
    cfg = LongShortConfig(
        decile=0.25, rebalance_every=20, slippage_bps=0.0,
        short_borrow_annual=0.0, risk_free_rate=0.0, benchmark_ticker=None,
        min_names_per_side=2, reject_stale_fills=True,
        delisting_policy=DelistingPolicy(fallback=fallback,
                                         grace_sessions=grace),
    )
    return run_long_short_backtest(panel, cfg, execution_prices=_exec(panel))


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_one_missing_quote_does_not_write_a_position_off() -> None:
    """The reported bug. Prices are flat at 100 for every name and every
    session bar one, so nothing has lost value and nothing has stopped
    trading."""
    res = _run({FILL_SESSION})
    assert res.metrics.get("disposals_written_off", 0) == 0, (
        "a single absent session wrote the holding off")
    assert res.metrics.get("exits_deferred", 0) >= 1, (
        "the gap was not deferred either")


def test_nav_survives_a_single_absent_session() -> None:
    clean = _run(set())
    gapped = _run({FILL_SESSION})
    assert float(gapped.daily_nav.iloc[-1]) == pytest.approx(
        float(clean.daily_nav.iloc[-1]), rel=0.02), (
        "one missing quote moved NAV materially on an otherwise flat panel")


def test_the_gap_is_counted_from_the_last_quote_not_the_first_session() -> None:
    """The same one-session gap must behave identically early and late in the
    run. Under the defect the gap grew with the session index, so a late gap
    was always fatal and an early one was not."""
    early = _run({21})
    late = _run({FILL_SESSION})
    assert (early.metrics.get("disposals_written_off", 0)
            == late.metrics.get("disposals_written_off", 0) == 0)


# ---------------------------------------------------------------------------
# The grace period still has to end
# ---------------------------------------------------------------------------


def test_a_name_that_stops_printing_is_still_written_off() -> None:
    """Fixing the counter must not make the policy inert: a real delisting
    still has to be disposed of once the grace period lapses."""
    res = _run(set(range(40, len(DATES))))
    assert res.metrics.get("disposals_written_off", 0) >= 1


def test_a_gap_inside_the_grace_period_defers() -> None:
    res = _run({FILL_SESSION - 1, FILL_SESSION}, grace=3)
    assert res.metrics.get("disposals_written_off", 0) == 0
    assert res.metrics.get("exits_deferred", 0) >= 1


def test_a_gap_beyond_the_grace_period_disposes() -> None:
    res = _run(set(range(FILL_SESSION - 10, FILL_SESSION + 1)), grace=3)
    assert res.metrics.get("disposals_written_off", 0) >= 1


def test_the_hold_policy_never_disposes() -> None:
    res = _run(set(range(40, len(DATES))), fallback="hold")
    assert res.metrics.get("disposals_written_off", 0) == 0
    assert res.metrics.get("exits_deferred", 0) >= 1


def test_a_clean_panel_defers_nothing() -> None:
    res = _run(set())
    assert res.metrics.get("exits_deferred", 0) == 0
    assert res.metrics.get("disposals_written_off", 0) == 0


# ---------------------------------------------------------------------------
# The live path counts the same duration
# ---------------------------------------------------------------------------


def test_the_live_counter_resets_on_a_real_quote() -> None:
    """The live book tracks the same gap on the position itself. A quote
    arriving has to clear it, or the count is again 'sessions since start'."""
    from stock_predictor.portfolio import (
        generate_orders_long_short,
        init_state,
    )

    sessions = DATES.to_numpy()
    picks = [{"ticker": f"T{i:02d}", "prob": float(N - i)} for i in range(N)]
    prices = {f"T{i:02d}": 100.0 for i in range(N)}
    dark = {k: v for k, v in prices.items() if k != DARK}

    def step(state, px, day):
        return generate_orders_long_short(
            state, picks, px, decile=0.25, long_weight=0.5, short_weight=0.5,
            rebalance_every=20, slippage_bps=0.0, as_of=str(DATES[day].date()),
            trading_dates=sessions, min_names_per_side=2,
        )[1]

    st = step(init_state(), prices, 0)
    st = step(st, dark, 1)
    st = step(st, dark, 2)
    held = next(p for p in st.positions if p.ticker == DARK)
    assert held.sessions_unpriced == 2

    st = step(st, prices, 3)
    held = next(p for p in st.positions if p.ticker == DARK)
    assert held.sessions_unpriced == 0, "a real quote did not clear the gap"


def test_the_live_gap_is_not_double_counted() -> None:
    """The counter is incremented once when marks are refreshed. Adding one
    again at the disposal check would end the grace period a session early."""
    import inspect

    from stock_predictor import portfolio

    src = inspect.getsource(portfolio.generate_orders_long_short)
    assert "held.sessions_unpriced + 1" not in src, (
        "the gap is incremented twice in one session")
