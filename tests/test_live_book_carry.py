"""Six defects in how a long-short book carries itself between sessions.

Five are in the live path and one is in the disposal rule both paths share. All
six are the same shape: a quantity that is correct in memory, for one session,
under one code path, and wrong the moment any of those change.

* **A written-off short manufactured money.** Disposal settles at zero on the
  stated fallback. For a long that is the conservative reading -- the holder's
  claim is worthless. For a *short* it is the opposite: a liability erased for
  free is the maximum possible profit. Unchanged prices and no settlement
  evidence produced a $50,000 gain on $100,000 of capital.
* **Borrow vanished between runs.** ``last_accrual_date`` was never written to
  the state file, so every reload reset it and the accrual guard skipped the
  charge. $1.98 in memory, $0 after a save and a load.
* **The high-water mark never rose.** NAV to $130,000 and back to $100,000
  reported 0% drawdown instead of 23.1%, so the kill switch could not fire.
* **A tripped kill switch waited for the calendar.** The rebalance early return
  runs before ``allow_new`` is read, so a halted book generated no closing
  orders until its next scheduled rebalance -- up to 63 sessions of the
  exposure the switch exists to end.
* **Cash earned nothing.** The backtest credits interest on cash at 4.5% and
  the live path only subtracted borrow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_predictor.delisting import DelistingPolicy, disposal_value
from stock_predictor.portfolio import (
    LONG_SHORT_COHORT,
    OPEN_ENDED_EXPIRY,
    PortfolioState,
    Position,
    generate_orders_long_short,
    init_state,
    load_state,
    portfolio_value,
    save_state,
)

DATES = pd.bdate_range("2024-01-01", periods=200)
SESSIONS = DATES.to_numpy()
N = 20
TICKERS = [f"T{i:02d}" for i in range(N)]


def _picks(order: list[str] | None = None) -> list[dict]:
    names = order or TICKERS
    return [{"ticker": t, "prob": float(len(names) - i)}
            for i, t in enumerate(names)]


def _prices(px: float = 100.0, **over) -> dict[str, float]:
    p = {t: px for t in TICKERS}
    p.update(over)
    return p


def _run(state, picks, prices, as_of, **kw):
    base = dict(decile=0.25, long_weight=0.5, short_weight=0.5,
                rebalance_every=21, slippage_bps=0.0, min_names_per_side=2,
                trading_dates=SESSIONS)
    base.update(kw)
    return generate_orders_long_short(state, picks, prices, as_of=as_of, **base)


# ---------------------------------------------------------------------------
# A written-off short is not free money
# ---------------------------------------------------------------------------


def test_a_short_is_not_settled_at_zero_without_evidence() -> None:
    """Zero is conservative for a long and maximally optimistic for a short.
    ``specs.md:181`` asks for a conservative fallback, so a short with no
    evidence covers at its last mark instead."""
    px, src = disposal_value(
        "AAA", "2024-06-01", evidence={}, sessions_unpriced=99,
        policy=DelistingPolicy(fallback="write_off", grace_sessions=3),
        direction=-1, mark=87.5,
    )
    assert px == pytest.approx(87.5)
    assert src != "write_off"


def test_a_long_is_still_written_off_at_zero() -> None:
    px, src = disposal_value(
        "AAA", "2024-06-01", evidence={}, sessions_unpriced=99,
        policy=DelistingPolicy(fallback="write_off", grace_sessions=3),
        direction=1, mark=87.5,
    )
    assert px == 0.0
    assert src == "write_off"


def test_evidence_settles_either_direction_at_the_stated_price() -> None:
    ev = {"AAA": (pd.Timestamp("2024-05-01"), 12.0)}
    for direction in (1, -1):
        px, src = disposal_value(
            "AAA", "2024-06-01", evidence=ev, sessions_unpriced=99,
            policy=DelistingPolicy(fallback="write_off", grace_sessions=3),
            direction=direction, mark=87.5,
        )
        assert (px, src) == (12.0, "evidence")


def test_covering_a_dark_short_does_not_create_profit() -> None:
    """The reported reproduction: flat prices, no evidence, a short that stops
    printing. NAV must not jump by the size of the liability."""
    _, opened = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    short = next(p for p in opened.positions if p.shares < 0)
    dark = _prices()
    del dark[short.ticker]
    policy = DelistingPolicy(fallback="write_off", grace_sessions=2)

    st = opened
    for i in range(1, 40):
        _, st = _run(st, _picks(), dark, str(DATES[i].date()),
                     delisting_policy=policy)
    before = portfolio_value(opened, _prices())
    after = portfolio_value(st, dark)
    assert after <= before * 1.02, (
        f"erasing the short liability manufactured {after - before:,.0f}")


# ---------------------------------------------------------------------------
# Borrow has to survive a save and a load
# ---------------------------------------------------------------------------


def test_the_accrual_date_round_trips(tmp_path: Path) -> None:
    st = PortfolioState(last_accrual_date="2024-03-05")
    path = tmp_path / "state.json"
    save_state(st, path)
    assert json.loads(path.read_text())["last_accrual_date"] == "2024-03-05"
    assert load_state(path).last_accrual_date == "2024-03-05"


def test_a_state_file_without_the_field_still_loads(tmp_path: Path) -> None:
    """Files written before this existed must not fail to open."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "initial_capital": 100_000.0, "cash": 100_000.0,
        "high_watermark": 100_000.0, "positions": [], "history": [],
    }))
    assert load_state(path).last_accrual_date == ""


def test_borrow_is_charged_across_a_save_and_reload(tmp_path: Path) -> None:
    """The defect: every reload reset the accrual date, so the guard skipped
    the charge and a live book never paid borrow at all."""
    path = tmp_path / "state.json"
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()),
                 short_borrow_annual=0.10)
    save_state(st, path)

    reloaded = load_state(path)
    _, later = _run(reloaded, _picks(), _prices(), str(DATES[5].date()),
                    short_borrow_annual=0.10)
    assert later.cash < st.cash, "borrow was not charged after a reload"


# ---------------------------------------------------------------------------
# The high-water mark
# ---------------------------------------------------------------------------


def test_the_watermark_rises_with_nav() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    rich = _prices()
    for p in st.positions:
        if p.shares > 0:
            rich[p.ticker] = 160.0
    _, marked = _run(st, _picks(), rich, str(DATES[1].date()))
    assert marked.high_watermark > 100_000.0, "the peak was never recorded"
    assert marked.high_watermark == pytest.approx(
        portfolio_value(marked, rich), rel=1e-9)


def test_the_watermark_does_not_fall() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    rich = _prices()
    for p in st.positions:
        if p.shares > 0:
            rich[p.ticker] = 160.0
    _, peak = _run(st, _picks(), rich, str(DATES[1].date()))
    _, back = _run(peak, _picks(), _prices(), str(DATES[2].date()))
    assert back.high_watermark == pytest.approx(peak.high_watermark)


def test_a_drawdown_from_the_peak_is_visible_to_the_kill_switch() -> None:
    """NAV up then back to par must read as a real drawdown, not 0%."""
    from stock_predictor.portfolio import check_kill_switch

    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    rich = _prices()
    for p in st.positions:
        if p.shares > 0:
            rich[p.ticker] = 160.0
    _, peak = _run(st, _picks(), rich, str(DATES[1].date()))
    _, back = _run(peak, _picks(), _prices(), str(DATES[2].date()))

    halted, nav, dd = check_kill_switch(back, _prices(), 0.15)
    assert dd < -0.05, f"drawdown read as {dd:+.2%} against a recorded peak"


# ---------------------------------------------------------------------------
# A tripped switch unwinds now, not at the next rebalance
# ---------------------------------------------------------------------------


def test_a_halted_book_unwinds_off_calendar() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    assert st.positions
    orders, flat = _run(st, _picks(), _prices(), str(DATES[3].date()),
                        allow_new=False)
    assert orders, "a halted book generated no closing orders"
    assert flat.positions == (), "the book did not go flat"


def test_a_halted_empty_book_does_nothing() -> None:
    orders, st = _run(init_state(), _picks(), _prices(), str(DATES[3].date()),
                      allow_new=False)
    assert orders == ()
    assert st.positions == ()


def test_an_unhalted_book_still_respects_the_calendar() -> None:
    """Regression guard: the unwind path must not make every session a
    rebalance."""
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    orders, _ = _run(st, _picks(), _prices(), str(DATES[3].date()))
    assert orders == ()


# ---------------------------------------------------------------------------
# Cash earns what the simulation says it earns
# ---------------------------------------------------------------------------


def test_cash_accrues_interest() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()),
                 risk_free_rate=0.045)
    _, later = _run(st, _picks(), _prices(), str(DATES[10].date()),
                    risk_free_rate=0.045)
    assert later.cash > st.cash, "cash earned no interest over ten sessions"


def test_no_interest_when_the_rate_is_zero() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    _, later = _run(st, _picks(), _prices(), str(DATES[5].date()))
    assert later.cash == pytest.approx(st.cash, abs=1e-9)


def test_interest_and_borrow_are_both_applied() -> None:
    """They net against each other in the backtest and must here too."""
    _, opened = _run(init_state(), _picks(), _prices(), str(DATES[0].date()),
                     risk_free_rate=0.045, short_borrow_annual=0.10)
    _, both = _run(opened, _picks(), _prices(), str(DATES[10].date()),
                   risk_free_rate=0.045, short_borrow_annual=0.10)
    _, borrow_only = _run(opened, _picks(), _prices(), str(DATES[10].date()),
                          risk_free_rate=0.0, short_borrow_annual=0.10)
    assert both.cash > borrow_only.cash


# ---------------------------------------------------------------------------
# The pipeline asks for the objective it was told to use
# ---------------------------------------------------------------------------


def test_binary_objective_is_passed_explicitly() -> None:
    """The CLI now defaults to --objective rank, so omitting a flag for binary
    silently trained a ranker and called it a classifier comparison."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        ["bash", "-c",
         f'cd "{root}" && DRY_RUN=1 OBJECTIVE=binary ./scripts/run_pipeline.sh train-full'],
        capture_output=True, text=True, timeout=180,
    ).stdout
    assert "--objective binary" in out, out[-400:]


def test_rank_objective_is_also_explicit() -> None:
    import subprocess

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        ["bash", "-c",
         f'cd "{root}" && DRY_RUN=1 OBJECTIVE=rank ./scripts/run_pipeline.sh train-full'],
        capture_output=True, text=True, timeout=180,
    ).stdout
    assert "--objective rank" in out, out[-400:]


# ---------------------------------------------------------------------------
# Nothing above may break the book
# ---------------------------------------------------------------------------


def test_a_flat_market_leaves_nav_unchanged() -> None:
    _, st = _run(init_state(), _picks(), _prices(), str(DATES[0].date()))
    assert portfolio_value(st, _prices()) == pytest.approx(100_000.0, rel=1e-9)


def test_an_existing_short_position_still_marks_as_a_liability() -> None:
    st = PortfolioState(
        cash=110_000.0,
        positions=(Position(ticker="T19", shares=-100, entry_price=100.0,
                            entry_date="2024-01-01",
                            expiry_date=OPEN_ENDED_EXPIRY,
                            cohort_id=LONG_SHORT_COHORT, last_price=100.0),),
    )
    assert portfolio_value(st, _prices()) == pytest.approx(100_000.0)
    assert portfolio_value(st, _prices(T19=130.0)) == pytest.approx(97_000.0)
