"""The long-only engines must count a gap from the last quote too.

``tests/test_unpriced_gap.py`` records the defect in the long-short engine: the
unpriced-gap counter was built as::

    _last_priced = actual.mul(_pos, axis=0).where(actual).cummax()

``cummax`` skips NaN without filling it, so the frame is NaN on exactly the
rows where a quote is missing -- the only rows the disposal branch reads. The
gap then comes out as sessions since the **start of the backtest**, any grace
period is already spent, and a holding that misses one session is written off
to zero on the spot.

The long-short engine was fixed with ``.ffill()``. The cohort and rank-hold
engines carried the same line without it. On the recorded baseline that wrote
JNPR and DISCK off at zero: two names that had printed every session for years
and were being acquired, not delisted worthless.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    run_backtest,
    run_rank_hold_backtest,
)
from stock_predictor.delisting import DelistingPolicy

DATES = pd.bdate_range("2024-01-01", periods=120)
N = 12
DARK = "T00"
FALLS_OUT_OF_FAVOUR = 40
"""DARK ranks best until this session and worst after it, so both engines hold
it and both then try to sell it: the cohort engine on expiry, rank-hold on
rank decay. An exit attempt is the only time either can find a missing quote."""
SINGLE_GAPS = range(FALLS_OUT_OF_FAVOUR - 12, FALLS_OUT_OF_FAVOUR + 20)

ENGINES = {"cohort": run_backtest, "rank-hold": run_rank_hold_backtest}


def _panel(missing: set[int]) -> pd.DataFrame:
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            if t == DARK and di in missing:
                continue
            prob = float(N - i)
            if t == DARK and di >= FALLS_OUT_OF_FAVOUR:
                prob = -1.0
            rows.append({"date": d, "ticker": t, "prob": prob, "adj_close": 100.0})
    return pd.DataFrame(rows)


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


def _final_nav(engine: str, missing: set[int], *, grace: int = 3,
               fallback: str = "write_off") -> float:
    panel = _panel(missing)
    cfg = BacktestConfig(
        top_n=3, holding_days=5, max_overlapping_cohorts=2, exit_rank=6,
        slippage_bps=0.0, benchmark_ticker=None, reject_stale_fills=True,
        delisting_policy=DelistingPolicy(fallback=fallback, grace_sessions=grace),
    )
    res = ENGINES[engine](panel, cfg, execution_prices=_exec(panel))
    return float(res.daily_nav.iloc[-1])


@pytest.mark.parametrize("engine", list(ENGINES))
def test_no_single_absent_session_writes_a_holding_off(engine: str) -> None:
    """Prices are flat at 100 for every name on every session but one, so
    nothing has lost value. Whichever session goes missing, NAV must not move."""
    clean = _final_nav(engine, set())
    moved = {
        gap: nav for gap in SINGLE_GAPS
        if (nav := _final_nav(engine, {gap})) != pytest.approx(clean, rel=1e-6)
    }
    assert moved == {}, f"clean NAV {clean:,.0f}; one missing quote gave {moved}"


@pytest.mark.parametrize("engine", list(ENGINES))
def test_a_gap_inside_the_grace_period_costs_nothing(engine: str) -> None:
    clean = _final_nav(engine, set())
    for start in SINGLE_GAPS:
        nav = _final_nav(engine, {start, start + 1}, grace=3)
        assert nav == pytest.approx(clean, rel=1e-6), start


@pytest.mark.parametrize("engine", list(ENGINES))
def test_a_name_that_stops_printing_is_still_written_off(engine: str) -> None:
    """Fixing the counter must not make the policy inert: a holding that never
    prints again has to be disposed of once the grace period lapses."""
    clean = _final_nav(engine, set())
    gone = _final_nav(engine, set(range(FALLS_OUT_OF_FAVOUR - 6, len(DATES))))
    assert gone < clean * 0.95


@pytest.mark.parametrize("engine", list(ENGINES))
def test_the_same_gap_costs_the_same_early_and_late(engine: str) -> None:
    """Under the defect the gap grew with the session index, so the outcome of
    an identical gap depended on how far into the run it fell."""
    clean = _final_nav(engine, set())
    for gap in (SINGLE_GAPS[0], SINGLE_GAPS[-1]):
        assert _final_nav(engine, {gap}) == pytest.approx(clean, rel=1e-6), gap


# ---------------------------------------------------------------------------
# What the fix above uncovered: a stranded leg is not a cohort
# ---------------------------------------------------------------------------
#
# A leg whose exit defers is carried as its own single-ticker Cohort until it
# settles. The slot count treated that fragment as a whole cohort, so one
# stranded $3,500 position held one of the two slots. The next cohort then took
# all the cash instead of half, the one after it opened with what was left --
# nothing -- and the book alternated between a cohort holding everything and a
# cohort holding $0.00000000001 for the rest of the run. On the recorded
# baseline this began with NBL in November 2020.
#
# It was invisible while the counter above was broken, because a deferral never
# happened: the leg was written off on the day.


def _cohorts(missing: set[int]):
    panel = _panel(missing)
    cfg = BacktestConfig(
        top_n=3, holding_days=5, max_overlapping_cohorts=2, exit_rank=6,
        slippage_bps=0.0, benchmark_ticker=None, reject_stale_fills=True,
        delisting_policy=DelistingPolicy(grace_sessions=3),
    )
    return run_backtest(panel, cfg, execution_prices=_exec(panel)).cohorts


def _full_cohort_capitals(cohorts) -> list[float]:
    return [c.capital for c in cohorts if len(c.tickers) > 1]


def test_a_deferred_leg_does_not_take_a_cohort_slot() -> None:
    """Prices are flat, so NAV is 100,000 throughout and each of two
    overlapping cohorts should be about half of it, gap or no gap."""
    clean = _full_cohort_capitals(_cohorts(set()))
    assert max(clean) < 60_000, "fixture: two cohorts should split the capital"
    lopsided = {}
    for start in SINGLE_GAPS:
        capitals = _full_cohort_capitals(_cohorts({start, start + 1}))
        if max(capitals) > 60_000:
            lopsided[start] = round(max(capitals))
    assert lopsided == {}, f"a cohort took more than its share after a deferral: {lopsided}"


def test_no_cohort_is_opened_on_dust() -> None:
    for start in SINGLE_GAPS:
        tiny = [c.capital for c in _cohorts({start, start + 1}) if c.capital < 1.0]
        assert tiny == [], (start, tiny)


def test_the_same_number_of_cohorts_open_with_and_without_a_gap() -> None:
    """A stranded leg must not cost the strategy a rebalance."""
    def signals(cohorts) -> int:
        return len({c.signal_date for c in cohorts})

    clean = signals(_cohorts(set()))
    for start in SINGLE_GAPS:
        assert signals(_cohorts({start, start + 1})) == clean, start
