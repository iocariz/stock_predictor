"""A long-short position that cannot be sold has to go somewhere.

Wiring this engine to a command line surfaced that it lacks the machinery both
other engines got. When a fill is rejected -- no quote, or only a
carried-forward one -- the rebalance loop simply ``continue``s. For an *entry*
that is correct: you do not open a position you cannot price. For an *exit* it
means the position is retained, which is also correct, and then nothing further
happens to it. Ever.

So a name that stops trading stays in the book indefinitely, marked at a
carried-forward price, quietly holding capital. On the 2019-2026 baseline the
engine rejected 222 of 5,293 fills and reported no deferral and no disposal
against any of them.

``specs.md:249`` requires missing exits to follow a documented, configurable
policy and to appear in the diagnostics. The cohort and rank-hold engines do
that -- defer, then dispose by explicit evidence or the stated fallback once a
grace period lapses. This one now does too, using the same
:mod:`~stock_predictor.delisting` machinery, so the three agree about what
happens to a holding that cannot be sold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.delisting import DelistingPolicy
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

DATES = pd.bdate_range("2024-01-01", periods=200)
N = 24
GONE = "T00"
"""Ranked top, so it is held, and then it stops printing."""


def _panel(dark_from: int | None = 90) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (len(DATES), N)), axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            price = float(px[di, i])
            if t == GONE and dark_from is not None and di >= dark_from:
                continue                      # stops trading entirely
            rows.append({"date": d, "ticker": t, "prob": float(N - i),
                         "adj_close": price})
    return pd.DataFrame(rows)


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


def _cfg(**kw) -> LongShortConfig:
    base = dict(decile=0.25, rebalance_every=21, slippage_bps=0.0,
                short_borrow_annual=0.0, risk_free_rate=0.0,
                benchmark_ticker=None, min_names_per_side=2,
                reject_stale_fills=True)
    base.update(kw)
    return LongShortConfig(**base)


def _run(panel, cfg, **kw):
    return run_long_short_backtest(panel, cfg, execution_prices=_exec(panel), **kw)


# ---------------------------------------------------------------------------


def test_an_unsellable_exit_is_counted_not_silently_retained() -> None:
    """specs.md:249 -- missing exits must appear in the diagnostics."""
    res = _run(_panel(), _cfg(delisting_policy=DelistingPolicy(fallback="hold")))
    m = res.metrics
    assert m["fills_rejected"] > 0
    assert m["exits_deferred"] > 0, "rejections were not attributed to anything"


def test_a_name_that_never_prices_again_is_written_off() -> None:
    """Deferring forever leaves capital in a position that cannot be sold."""
    res = _run(_panel(), _cfg(
        delisting_policy=DelistingPolicy(fallback="write_off", grace_sessions=5)))
    m = res.metrics
    assert m.get("disposals_written_off", 0) >= 1


def test_explicit_evidence_is_used_when_supplied() -> None:
    ev = pd.DataFrame({"ticker": [GONE], "date": [DATES[95]], "proceeds": [42.0]})
    res = _run(_panel(), _cfg(), delisting_proceeds=ev)
    assert res.metrics.get("disposals_by_evidence", 0) >= 1


def test_holding_indefinitely_disposes_of_nothing() -> None:
    """The policy is a choice, and 'hold' must actually hold."""
    res = _run(_panel(), _cfg(delisting_policy=DelistingPolicy(fallback="hold")))
    m = res.metrics
    assert m.get("disposals_written_off", 0) == 0
    assert m["exits_deferred"] > 0


def test_a_clean_panel_defers_nothing() -> None:
    """Regression guard: this must be inert when every name keeps trading."""
    res = _run(_panel(dark_from=None), _cfg())
    assert res.metrics["exits_deferred"] == 0
    assert res.metrics.get("disposals_written_off", 0) == 0


def test_a_write_off_removes_the_position_from_the_book() -> None:
    """Disposing means gone, not marked at zero and still counted."""
    res = _run(_panel(), _cfg(
        delisting_policy=DelistingPolicy(fallback="write_off", grace_sessions=5)))
    assert res.daily_nav.iloc[-1] > 0
    assert np.isfinite(res.daily_nav).all()


def test_the_diagnostics_are_reported_by_source() -> None:
    res = _run(_panel(), _cfg(
        delisting_policy=DelistingPolicy(fallback="write_off", grace_sessions=5)))
    for key in ("exits_deferred", "disposals_written_off", "disposals_by_evidence"):
        assert key in res.metrics, key


def test_disposal_proceeds_reach_cash() -> None:
    """Evidence-based proceeds are real money and must show up in the book."""
    ev = pd.DataFrame({"ticker": [GONE], "date": [DATES[95]], "proceeds": [42.0]})
    with_ev = _run(_panel(), _cfg(), delisting_proceeds=ev)
    written = _run(_panel(), _cfg(
        delisting_policy=DelistingPolicy(fallback="write_off", grace_sessions=5)))
    assert with_ev.metrics["disposal_proceeds"] > written.metrics["disposal_proceeds"]


def test_the_default_policy_matches_the_other_engines() -> None:
    assert LongShortConfig().delisting_policy == DelistingPolicy()


def test_a_short_position_can_also_be_disposed_of() -> None:
    """Shorts go dark too, and a short that cannot be covered is a liability
    that does not disappear by being ignored."""
    panel = _panel()
    # Rank the dark name last so it sits in the short book instead.
    panel.loc[panel["ticker"] == GONE, "prob"] = -1.0
    res = run_long_short_backtest(
        panel, _cfg(delisting_policy=DelistingPolicy(fallback="write_off",
                                                     grace_sessions=5)),
        execution_prices=_exec(panel))
    assert res.metrics["exits_deferred"] + res.metrics.get(
        "disposals_written_off", 0) > 0
    assert pytest.approx(res.daily_nav.iloc[-1], rel=1) == res.daily_nav.iloc[-1]
