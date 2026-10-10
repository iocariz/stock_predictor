"""The rebalance grid has an anchor, and the result depends on it.

``signal_idx = set(range(0, n_days - 1, rebalance_every))`` anchors the first
rebalance at index 0. With ``rebalance_every`` at 63 that picks one of 63
possible schedules, and the other 62 are never measured.

The README already says to "always sweep rebalance-schedule offsets", and the
tool to do it did not exist. The gap showed up as two runs disagreeing on the
same window: the 2025-2026 slice of a 2019-anchored panel took 8 rebalances
and reported alpha +12.9%, while a 2025-anchored run took 7 and reported
+2.2%. Same engine, same universe, different anchor dates -- so they were
trading different books and the difference was read as a finding about
survivorship.

An offset is not a tuning knob. Sweeping it measures how much of a result is
the schedule rather than the signal, which is the only way to know whether a
reported alpha is robust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.long_short import (
    LongShortConfig,
    run_long_short_backtest,
)

N_DAYS = 200
N_TICKERS = 24
DATES = pd.bdate_range("2024-01-01", periods=N_DAYS)
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]


def _scored() -> pd.DataFrame:
    """A panel with a stable ranking and gently drifting prices."""
    rng = np.random.default_rng(7)
    rows = []
    for di, d in enumerate(DATES):
        for ti, t in enumerate(TICKERS):
            rows.append({
                "date": d,
                "ticker": t,
                # Stable ranking so the book is deterministic per session.
                "prob": float(N_TICKERS - ti) + rng.normal(0, 0.01),
                "adj_close": 100.0 * (1.0 + 0.001 * di) + ti,
            })
    return pd.DataFrame(rows)


def _cfg(**kw) -> LongShortConfig:
    base = dict(decile=0.25, long_weight=0.5, short_weight=0.5,
                rebalance_every=21, min_names_per_side=2,
                slippage_bps=0.0, benchmark_ticker=None)
    base.update(kw)
    return LongShortConfig(**base)


def test_the_offset_defaults_to_zero() -> None:
    """Every recorded result in this project was produced at offset 0, so the
    default must reproduce it exactly."""
    assert LongShortConfig().rebalance_offset == 0


def test_an_offset_moves_the_rebalance_dates() -> None:
    scored = _scored()
    a = run_long_short_backtest(scored, _cfg(rebalance_offset=0))
    b = run_long_short_backtest(scored, _cfg(rebalance_offset=5))
    # Turnover is non-zero only on fill sessions, so its support *is* the
    # schedule.
    fills_a = set(a.turnover[a.turnover > 0].index)
    fills_b = set(b.turnover[b.turnover > 0].index)
    assert fills_a != fills_b, "the offset did not move the schedule"


def test_an_offset_shorter_than_the_period_keeps_every_rebalance() -> None:
    """Shifting the anchor must not quietly cost the run a cycle."""
    scored = _scored()
    a = run_long_short_backtest(scored, _cfg(rebalance_offset=0))
    b = run_long_short_backtest(scored, _cfg(rebalance_offset=5))
    assert abs(a.n_rebalances - b.n_rebalances) <= 1, (
        f"offset changed the cycle count: {a.n_rebalances} vs {b.n_rebalances}")


def test_a_full_period_offset_is_rejected() -> None:
    """Offsetting by the whole period is the next schedule, not a new one, and
    silently accepting it would double-count a result in a sweep."""
    with pytest.raises(ValueError, match="rebalance_offset"):
        _cfg(rebalance_offset=21)


def test_a_negative_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="rebalance_offset"):
        _cfg(rebalance_offset=-1)


def test_the_offset_is_recorded_on_the_result() -> None:
    """A sweep is unreadable if a row cannot say which schedule it measured."""
    r = run_long_short_backtest(_scored(), _cfg(rebalance_offset=3))
    assert r.config.rebalance_offset == 3
