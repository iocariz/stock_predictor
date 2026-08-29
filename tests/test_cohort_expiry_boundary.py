"""A cohort that sells on the signal session has sold.

Three places decide when a fixed-hold cohort stops being a position, and one of
them disagreed with the other two.

* ``_build_daily_nav`` credits the cohort's capital **on** ``exit_date`` and
  marks it only through ``exit_date - 1``. Settlement happens that session.
* The live path agrees: ``find_expiring_positions`` treats ``expiry_date <=
  as_of`` as expired, and ``active_cohort_ids`` counts a position active only
  while ``expiry_date > as_of``.
* The rebalance loop did not. It credited proceeds only when ``exit_date <
  sig_date`` and counted a cohort active through ``sig_date <= exit_date``.

So on a signal session that fell exactly on a cohort's exit, the simulation
held the position open *and* withheld its cash — the slot was occupied by
something already sold, and the money it had just returned could not fund the
next entry. Entry happens on the session *after* the signal, so by then the
sale is a day old; there is no overlap to protect against.

On the 2019-2026 baseline this hit **13 of 57 cohorts**.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    _get_rebalance_dates,
    _prepare_scored,
    run_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=120)
TICKERS = [f"T{i:02d}" for i in range(12)]
HOLD = 9
"""Entry lands on a Monday and the exit nine sessions later is a Friday, which
is a signal session — the collision this module is about."""


def _panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "ticker": t, "prob": float(len(TICKERS) - i),
         "adj_close": 100.0 + i + 0.01 * di}
        for di, d in enumerate(DATES) for i, t in enumerate(TICKERS)
    ])


def _cfg(**kw) -> BacktestConfig:
    base = dict(top_n=3, holding_days=HOLD, max_overlapping_cohorts=1,
                slippage_bps=0.0, rebalance_day="Friday",
                benchmark_ticker=None, reject_stale_fills=False)
    base.update(kw)
    return BacktestConfig(**base)


def _signal_dates(cfg) -> set[pd.Timestamp]:
    _, td, _, _ = _prepare_scored(_panel(), None)
    return set(pd.DatetimeIndex(_get_rebalance_dates(td, cfg.rebalance_day)))


def test_the_fixture_actually_produces_a_collision() -> None:
    """Guard: if no cohort exits on a signal session this proves nothing."""
    cfg = _cfg()
    res = run_backtest(_panel(), cfg)
    sig = _signal_dates(cfg)
    collisions = [c for c in res.cohorts if pd.Timestamp(c.exit_date) in sig]
    assert collisions, "fixture must exercise the exit-on-signal boundary"


def test_a_cohort_exiting_on_the_signal_does_not_hold_the_slot() -> None:
    """With one slot, a cohort selling on Friday must not block Friday's entry
    — the new position is bought on Monday, after the sale."""
    cfg = _cfg()
    res = run_backtest(_panel(), cfg)
    sig = _signal_dates(cfg)
    by_entry = {pd.Timestamp(c.entry_date) for c in res.cohorts}
    for c in res.cohorts:
        ex = pd.Timestamp(c.exit_date)
        if ex not in sig:
            continue
        if ex > DATES[-1] - pd.Timedelta(days=int(HOLD * 1.6)):
            continue          # no room left in the panel for another hold
        later = [d for d in by_entry if d > ex]
        assert later, f"nothing entered after the exit on {ex.date()}"
        # The next entry should follow that signal session immediately, not
        # skip a week because the slot looked occupied.
        assert min(later) <= ex + pd.Timedelta(days=5), (
            f"the slot freed on {ex.date()} was not reused until {min(later).date()}"
        )


def test_proceeds_are_available_to_the_signal_that_day() -> None:
    """The capital deployed after a collision must reflect the returned cash,
    not a book that is still missing it."""
    cfg = _cfg()
    res = run_backtest(_panel(), cfg)
    sig = _signal_dates(cfg)
    for c in res.cohorts:
        ex = pd.Timestamp(c.exit_date)
        if ex not in sig:
            continue
        nxt = sorted((x for x in res.cohorts if pd.Timestamp(x.entry_date) > ex),
                     key=lambda x: pd.Timestamp(x.entry_date))
        if not nxt:
            continue
        following = nxt[0]
        assert following.capital > 0
        # Prices drift up in this fixture, so a settled cohort returns more
        # than it took. The next cohort should be funded with at least that.
        assert following.capital >= c.capital * (1.0 + c.net_return) * 0.99


def test_nav_still_reconciles_after_the_boundary_change(  ) -> None:
    cfg = _cfg()
    res = run_backtest(_panel(), cfg)
    last = res.daily_nav.index[-1]
    closed = sum(c.capital * c.net_return for c in res.cohorts
                 if pd.Timestamp(c.exit_date) <= last)
    m = res.metrics
    unreal = (float(m.get("open_position_value", 0.0))
              - float(m.get("open_position_basis", 0.0)))
    assert float(res.daily_nav.iloc[-1]) == pytest.approx(
        cfg.initial_capital + closed + unreal, rel=1e-9
    )


def test_a_cohort_still_occupies_its_slot_before_expiry() -> None:
    """The correction must not free slots early — that would let the book hold
    more than max_overlapping_cohorts at once."""
    res = run_backtest(_panel(), _cfg(max_overlapping_cohorts=1))
    spans = sorted((pd.Timestamp(c.entry_date), pd.Timestamp(c.exit_date))
                   for c in res.cohorts)
    for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        assert s2 >= e1, f"cohorts {s1.date()}-{e1.date()} and {s2.date()} overlap"
