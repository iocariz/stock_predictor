"""What happens to a position that cannot be sold.

Rejecting unpriceable fills was correct, but it left capital locked in
positions that can never exit. specs.md is specific about the way out:

  :181  Historical delisting treatment MUST use explicit evidence or an
        explicit conservative fallback. A ticker's last available row alone
        MUST NOT prove delisting.
  :249  Missing exits, halts, and delistings MUST follow a documented
        configurable policy and appear in the result diagnostics.
  :587  Missing terminal vendor data is not automatically labeled a delisting.

So: a gap is never itself proof. Evidence wins when supplied. Absent evidence,
a named policy disposes of the position after a grace period, and the default
fallback is zero because a stock you cannot sell is not worth its last quote.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.delisting import (
    DelistingPolicy,
    disposal_value,
    load_proceeds,
)

AS_OF = pd.Timestamp("2024-06-14")


def _evidence() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["ACQ", "BUST"],
        "date": [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-05-01")],
        "proceeds": [58.50, 0.0],
    })


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_explicit_proceeds_win() -> None:
    """A cash acquisition at $58.50 is a fact, not a guess."""
    out = disposal_value("ACQ", AS_OF, evidence=load_proceeds(_evidence()),
                         sessions_unpriced=1, policy=DelistingPolicy())
    assert out == (58.50, "evidence")


def test_evidence_of_a_total_loss_is_still_evidence() -> None:
    out = disposal_value("BUST", AS_OF, evidence=load_proceeds(_evidence()),
                         sessions_unpriced=1, policy=DelistingPolicy())
    assert out == (0.0, "evidence")


def test_evidence_dated_after_the_run_is_not_yet_known() -> None:
    """Point-in-time: a deal that settles next month cannot pay today."""
    ev = load_proceeds(_evidence())
    out = disposal_value("ACQ", pd.Timestamp("2024-01-02"), evidence=ev,
                         sessions_unpriced=200, policy=DelistingPolicy())
    assert out != (58.50, "evidence")


def test_no_evidence_frame_is_not_an_error() -> None:
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=1,
                          policy=DelistingPolicy()) is None


# ---------------------------------------------------------------------------
# The grace period — a gap is not a delisting
# ---------------------------------------------------------------------------


def test_a_short_gap_is_held_not_written_off() -> None:
    """A halt or a vendor outage is not a delisting (specs.md:587)."""
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=5,
                          policy=DelistingPolicy()) is None


def test_a_gap_past_the_grace_period_triggers_the_fallback() -> None:
    out = disposal_value("X", AS_OF, evidence={}, sessions_unpriced=64,
                         policy=DelistingPolicy(grace_sessions=63))
    assert out == (0.0, "write_off")


def test_the_boundary_is_inclusive_of_the_grace_period() -> None:
    p = DelistingPolicy(grace_sessions=63)
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=63,
                          policy=p) is None
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=64,
                          policy=p) is not None


def test_hold_forever_is_an_available_policy() -> None:
    """Explicit, named, and never disposes — for anyone who wants the capital
    left visibly stuck rather than written off."""
    p = DelistingPolicy(fallback="hold")
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=9999,
                          policy=p) is None


def test_the_fallback_is_conservative_by_default() -> None:
    """A stock you cannot sell is not worth its last quote."""
    assert DelistingPolicy().fallback == "write_off"
    assert disposal_value("X", AS_OF, evidence={}, sessions_unpriced=1000,
                          policy=DelistingPolicy())[0] == 0.0


# ---------------------------------------------------------------------------
# Loading evidence
# ---------------------------------------------------------------------------


def test_proceeds_load_keyed_by_ticker() -> None:
    ev = load_proceeds(_evidence())
    assert set(ev) == {"ACQ", "BUST"}


def test_an_empty_frame_loads_empty() -> None:
    assert load_proceeds(pd.DataFrame(columns=["ticker", "date", "proceeds"])) == {}


def test_none_loads_empty() -> None:
    assert load_proceeds(None) == {}


def test_a_frame_missing_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="proceeds"):
        load_proceeds(pd.DataFrame({"ticker": ["A"], "date": [AS_OF]}))


def test_negative_proceeds_are_rejected() -> None:
    bad = pd.DataFrame({"ticker": ["A"], "date": [AS_OF], "proceeds": [-1.0]})
    with pytest.raises(ValueError, match="negative"):
        load_proceeds(bad)


def test_the_earliest_evidence_for_a_ticker_wins() -> None:
    """Two rows for one name: the first settlement is the one that happened."""
    dup = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-03-01")],
        "proceeds": [10.0, 7.0],
    })
    out = disposal_value("A", AS_OF, evidence=load_proceeds(dup),
                         sessions_unpriced=1, policy=DelistingPolicy())
    assert out == (7.0, "evidence")


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


def test_policy_rejects_nonsense() -> None:
    for bad in (dict(grace_sessions=-1), dict(fallback="invent_a_price")):
        with pytest.raises(ValueError):
            DelistingPolicy(**bad)


# ---------------------------------------------------------------------------
# Reached through the engine
# ---------------------------------------------------------------------------


def _panel(dies_at: int = 40, n_days: int = 200, n: int = 30) -> pd.DataFrame:
    """T00 is top-ranked and stops being quoted part-way through."""
    import numpy as np

    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, (n_days, n)), axis=0))
    rows = []
    for di, d in enumerate(dates):
        for i in range(n):
            t = f"T{i:02d}"
            if t == "T00" and di >= dies_at:
                continue
            rows.append({"date": d, "ticker": t, "prob": float(n - i),
                         "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


def _cfg(**kw):
    from stock_predictor.backtest import BacktestConfig

    base = dict(top_n=5, exit_rank=10, holding_days=10, slippage_bps=0.0,
                benchmark_ticker=None, rebalance_day="Friday")
    base.update(kw)
    return BacktestConfig(**base)


def test_a_stuck_position_is_written_off_after_the_grace_period() -> None:
    from stock_predictor.backtest import run_rank_hold_backtest

    res = run_rank_hold_backtest(_panel(), _cfg())
    assert res.metrics["disposals_written_off"] > 0
    assert res.metrics["disposal_proceeds"] == 0.0


def test_holding_forever_leaves_it_deferred_instead() -> None:
    from stock_predictor.backtest import run_rank_hold_backtest

    res = run_rank_hold_backtest(
        _panel(), _cfg(delisting_policy=DelistingPolicy(fallback="hold")),
    )
    assert res.metrics["disposals_written_off"] == 0
    assert res.metrics["exits_deferred"] > 0


def test_evidence_pays_real_proceeds_through_the_engine() -> None:
    from stock_predictor.backtest import run_rank_hold_backtest

    ev = pd.DataFrame({"ticker": ["T00"],
                       "date": [pd.Timestamp("2024-03-15")],
                       "proceeds": [250.0]})
    res = run_rank_hold_backtest(_panel(), _cfg(), delisting_proceeds=ev)
    assert res.metrics["disposals_by_evidence"] > 0
    assert res.metrics["disposal_proceeds"] > 0


def test_a_clean_panel_disposes_of_nothing() -> None:
    from stock_predictor.backtest import run_rank_hold_backtest

    res = run_rank_hold_backtest(_panel(dies_at=10_000), _cfg())
    assert res.metrics["disposals_written_off"] == 0
    assert res.metrics["disposals_by_evidence"] == 0
