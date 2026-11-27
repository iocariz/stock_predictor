"""The live book has to be able to say where it has been, and to be restarted.

Two gaps found while reviewing eleven weeks of paper trading.

**No record.** ``portfolio_state.json`` held the current positions and nothing
of the path: ``history`` was only ever written by the long-only engines, on a
cohort closing. The long-short book's NAV series existed in one place, the
cron's log file, as formatted text.

**No way back from the kill switch.** Past the drawdown limit the book unwinds
and stops opening positions, which is the point. But the limit is measured from
a high-water mark that only ever rises, so an unwound book sitting in cash stays
past it forever. The documented remedy was to edit the JSON by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.portfolio import (
    check_kill_switch,
    init_state,
    portfolio_value,
    rearm_kill_switch,
)
from stock_predictor.portfolio import generate_orders_long_short as gen

DATES = pd.bdate_range("2024-01-01", periods=200)
SESSIONS = DATES.to_numpy()
N = 20
TICKERS = [f"T{i:02d}" for i in range(N)]


def _picks() -> list[dict]:
    return [{"ticker": t, "prob": float(N - i)} for i, t in enumerate(TICKERS)]


def _prices(px: float = 100.0, **over: float) -> dict[str, float]:
    prices = {t: px for t in TICKERS}
    prices.update(over)
    return prices


def _run(state, prices, session: int, **kw):
    base = dict(decile=0.25, long_weight=0.5, short_weight=0.5,
                rebalance_every=21, slippage_bps=0.0, min_names_per_side=2,
                trading_dates=SESSIONS)
    base.update(kw)
    return gen(state, _picks(), prices, as_of=str(DATES[session].date()), **base)


def _navs(state) -> list[dict]:
    return [h for h in state.history if h.get("kind") == "nav"]


# --- the record ---------------------------------------------------------------


def test_every_session_leaves_a_nav_record() -> None:
    state = init_state(100_000.0)
    for session in (1, 2, 3):
        _, state = _run(state, _prices(), session)
    assert [h["date"] for h in _navs(state)] == [
        str(DATES[s].date()) for s in (1, 2, 3)]


def test_the_record_is_what_the_book_was_worth() -> None:
    _, state = _run(init_state(100_000.0), _prices(), 1)
    # Every long up 10%, every short flat: the book is long 50,000.
    up = _prices(**{t: 110.0 for t in TICKERS[:5]})
    _, state = _run(state, up, 2)
    last = _navs(state)[-1]
    assert last["nav"] == pytest.approx(portfolio_value(state, up), abs=0.01)
    assert last["nav"] == pytest.approx(105_000.0, rel=1e-3)
    assert last["long"] == pytest.approx(55_000.0, rel=1e-3)
    assert last["short"] == pytest.approx(-50_000.0, rel=1e-3)
    assert last["nav"] == pytest.approx(
        last["cash"] + last["long"] + last["short"], abs=0.01)


def test_running_twice_on_one_session_records_it_once() -> None:
    """The cron retries, and an operator may run by hand the same morning."""
    _, state = _run(init_state(100_000.0), _prices(), 1)
    _, state = _run(state, _prices(), 2)
    _, again = _run(state, _prices(T00=120.0), 2)
    dates = [h["date"] for h in _navs(again)]
    assert dates == sorted(set(dates))
    assert _navs(again)[-1]["nav"] == pytest.approx(
        portfolio_value(again, _prices(T00=120.0)), abs=0.01), "the later mark wins"


def test_a_rebalance_is_recorded_with_what_it_traded() -> None:
    orders, state = _run(init_state(100_000.0), _prices(), 1)
    events = [h for h in state.history if h.get("kind") == "rebalance"]
    assert len(events) == 1
    assert events[0]["date"] == str(DATES[1].date())
    assert events[0]["orders"] == len(orders) > 0
    assert events[0]["turnover"] == pytest.approx(100_000.0, rel=1e-3)


def test_a_quiet_session_records_no_rebalance() -> None:
    _, state = _run(init_state(100_000.0), _prices(), 1)
    _, state = _run(state, _prices(), 2)
    assert len([h for h in state.history if h.get("kind") == "rebalance"]) == 1


def test_the_record_survives_a_save_and_a_load(tmp_path) -> None:
    from stock_predictor.portfolio import load_state, save_state

    _, state = _run(init_state(100_000.0), _prices(), 1)
    path = tmp_path / "state.json"
    save_state(state, path)
    assert load_state(path).history == state.history


# --- the kill switch ----------------------------------------------------------


def _tripped():
    """A book that ran to 130,000, fell back to 100,000 and was unwound."""
    from dataclasses import replace

    return replace(init_state(100_000.0), high_watermark=130_000.0)


def test_an_unwound_book_stays_halted_on_its_own() -> None:
    """The behaviour that needs a way out: nothing the book can do in cash
    moves it back inside the limit."""
    halted, _, dd = check_kill_switch(_tripped(), {}, 0.15)
    assert halted and dd == pytest.approx(-0.2308, abs=1e-3)


def test_rearming_measures_drawdown_from_where_the_book_is_now() -> None:
    state = rearm_kill_switch(_tripped(), {})
    assert state.high_watermark == pytest.approx(100_000.0)
    halted, _, dd = check_kill_switch(state, {}, 0.15)
    assert not halted and dd == pytest.approx(0.0)


def test_rearming_uses_the_marked_value_not_the_cash() -> None:
    _, state = _run(init_state(100_000.0), _prices(), 1)
    from dataclasses import replace

    down = _prices(**{t: 60.0 for t in TICKERS[:5]})      # longs -40%: NAV 80,000
    state = replace(state, high_watermark=100_000.0)
    assert check_kill_switch(state, down, 0.15)[0]
    rearmed = rearm_kill_switch(state, down)
    assert rearmed.high_watermark == pytest.approx(80_000.0, rel=1e-3)
    assert rearmed.positions == state.positions, "re-arming trades nothing"


def test_rearming_is_recorded() -> None:
    """It moves the line the risk limit is measured from. That should not be
    something the state file cannot account for afterwards."""
    state = rearm_kill_switch(_tripped(), {}, when="2024-03-01")
    event = [h for h in state.history if h.get("kind") == "kill_switch_rearmed"]
    assert event == [{"kind": "kill_switch_rearmed", "date": "2024-03-01",
                      "watermark_before": 130_000.0, "watermark_after": 100_000.0}]
