"""A ranking needs something to rank.

Separating row roles keeps the newest sessions in the panel, which is right —
they are exactly what a live model scores. But a panel can still end ragged:
a date carrying two names is not a cross-section, and taking "the top 15" from
it means taking everything and calling the result a selection.

Entries are gated on cross-section width; exits never are, so a thin day can
still close a position but cannot open one on a ranking that does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    run_backtest,
    run_rank_hold_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=120)


def _panel(*, wide_until: int, wide: int = 40, thin: int = 2) -> pd.DataFrame:
    """A panel that collapses from *wide* names to *thin* partway through.

    Ranks rotate every five sessions so the rank-hold engine actually churns;
    a frozen ranking would never close a position and the exit assertions
    below would pass vacuously.
    """
    rng = np.random.default_rng(0)
    rows = []
    for di, d in enumerate(DATES):
        n = wide if di < wide_until else thin
        for i in range(n):
            rank = (i + di // 5) % n
            rows.append({
                "date": d,
                "ticker": f"T{i:02d}",
                "prob": float(n - rank) / n,
                "adj_close": float(100 * np.exp(rng.normal(0, 0.01))),
            })
    return pd.DataFrame(rows)


def _cfg(**kw) -> BacktestConfig:
    # exit_rank == top_n so the rank-hold engine closes a name as soon as it
    # rotates out of the book, rather than holding it to the end of the panel.
    base = dict(top_n=10, exit_rank=10, holding_days=5, slippage_bps=0.0,
                benchmark_ticker=None, risk_free_rate=0.0)
    base.update(kw)
    return BacktestConfig(**base)


def test_default_floor_is_the_basket_you_asked_for() -> None:
    """Fewer scored names than positions means the 'ranking' is the universe."""
    cfg = _cfg(top_n=10, rank_offset=5)
    assert cfg.effective_min_cross_section == 15


def test_explicit_floor_wins() -> None:
    assert _cfg(min_cross_section=99).effective_min_cross_section == 99


def test_floor_is_validated() -> None:
    with pytest.raises(ValueError, match="min_cross_section"):
        _cfg(min_cross_section=0)


ENGINES = pytest.mark.parametrize(
    "engine", [run_backtest, run_rank_hold_backtest], ids=["cohort", "rank_hold"],
)


@ENGINES
def test_no_entries_are_opened_on_a_thin_day(engine) -> None:
    """The ragged tail must not be traded as if it were a ranking."""
    res = engine(_panel(wide_until=60), _cfg())
    thin_start = DATES[60]
    late = [c for c in res.cohorts if pd.Timestamp(c.entry_date) > thin_start]
    assert not late, f"opened {len(late)} cohorts on a 2-name day"


@ENGINES
def test_a_wide_panel_is_unaffected(engine) -> None:
    """The guard must be invisible when there is a real cross-section."""
    res = engine(_panel(wide_until=len(DATES)), _cfg())
    assert len(res.cohorts) > 0


@ENGINES
def test_positions_still_exit_on_a_thin_day(engine) -> None:
    """Gating entries must never strand a holding. A position opened while the
    panel was wide has to be closable after it narrows."""
    res = engine(_panel(wide_until=60), _cfg(holding_days=3))
    closed_after = [c for c in res.cohorts
                    if pd.Timestamp(c.exit_date) >= DATES[60]]
    assert closed_after, "a thin cross-section must not trap open positions"
