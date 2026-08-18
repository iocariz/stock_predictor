"""Per-name short borrow.

A flat rate is optimistic for this model specifically: it partly ranks
volatility, so its short book is drawn from the names that are expensive to
borrow. These tests pin the mechanism and, most importantly, the property
that makes it worth having — the short book must cost more than the universe.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_predictor.borrow import (
    DEFAULT_BORROW_SCHEDULE,
    GENERAL_COLLATERAL,
    borrow_concentration,
    borrow_from_percentile,
    estimate_borrow_rates,
    realized_volatility,
    resolve_borrow_rates,
)
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

DATES = pd.bdate_range("2024-01-01", periods=200)
N = 60


def _panel(*, seed: int = 0, vol_ranked: bool = True) -> pd.DataFrame:
    """Panel where ticker index sets volatility, and score ranks *inversely*
    to volatility — so the short book is the high-volatility tail, exactly the
    situation a flat borrow rate flatters."""
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.05, N) if vol_ranked else np.full(N, 0.02)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 1, (len(DATES), N)) * vols, axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            rows.append({
                "date": d, "ticker": f"T{i:02d}",
                "prob": float(-i),                  # T00 best, T59 worst
                "adj_close": float(px[di, i]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rate construction
# ---------------------------------------------------------------------------


def test_schedule_is_monotone_and_starts_at_general_collateral() -> None:
    bounds = [b for b, _ in DEFAULT_BORROW_SCHEDULE]
    rates = [r for _, r in DEFAULT_BORROW_SCHEDULE]
    assert bounds == sorted(bounds)
    assert rates == sorted(rates), "borrow must rise with scarcity"
    assert rates[0] == GENERAL_COLLATERAL
    assert bounds[-1] == 1.0


def test_percentile_maps_through_the_schedule() -> None:
    pct = pd.Series([0.10, 0.85, 0.93, 0.975, 0.999])
    out = borrow_from_percentile(pct)
    assert out.tolist() == [0.005, 0.015, 0.03, 0.08, 0.20]


def test_unknown_percentile_defaults_to_general_collateral() -> None:
    out = borrow_from_percentile(pd.Series([np.nan, 0.5]))
    assert out.iloc[0] == GENERAL_COLLATERAL


def test_volatility_is_trailing_only() -> None:
    """A rate charged today may not use tomorrow's prices."""
    panel = _panel()
    vol = realized_volatility(panel)
    early = panel["date"] <= DATES[3]
    assert vol[early].isna().all(), "no volatility before enough history exists"


def test_estimated_rates_track_actual_volatility() -> None:
    rates = estimate_borrow_rates(_panel())
    panel = _panel()
    vol = realized_volatility(panel)
    merged = rates.assign(vol=vol.to_numpy()).dropna()
    corr = merged["borrow_rate"].corr(merged["vol"], method="spearman")
    assert corr > 0.5, f"rate should rise with volatility, got {corr:.2f}"


def test_flat_universe_gets_a_full_schedule_spread_anyway() -> None:
    """Ranks are cross-sectional, so *some* name is always the top percentile.
    That is intended: the schedule describes shape, not absolute scarcity."""
    rates = estimate_borrow_rates(_panel(vol_ranked=False))
    assert rates["borrow_rate"].nunique() > 1


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_real_rates_on_the_panel_beat_the_proxy() -> None:
    panel = _panel()
    panel["borrow_rate"] = 0.123
    out = resolve_borrow_rates(panel, flat_rate=0.005, per_name=True)
    assert out is not None
    assert (out["borrow_rate"] == 0.123).all(), "supplied data must win"


def test_supplied_frame_beats_everything() -> None:
    panel = _panel()
    frame = panel[["date", "ticker"]].assign(borrow_rate=0.077)
    out = resolve_borrow_rates(panel, flat_rate=0.005, per_name=frame)
    assert (out["borrow_rate"] == 0.077).all()


def test_supplied_frame_is_validated() -> None:
    with pytest.raises(ValueError, match="borrow_rate"):
        resolve_borrow_rates(
            _panel(), flat_rate=0.005,
            per_name=pd.DataFrame({"date": [], "ticker": []}),
        )


def test_flat_rate_path_returns_none() -> None:
    assert resolve_borrow_rates(_panel(), flat_rate=0.005, per_name=False) is None


# ---------------------------------------------------------------------------
# The property that justifies the module
# ---------------------------------------------------------------------------


def test_the_short_book_is_dearer_than_the_universe() -> None:
    """The whole reason for per-name borrow: this model shorts the volatile
    tail, so a flat general-collateral rate understates its cost."""
    panel = _panel()
    rates = estimate_borrow_rates(panel)
    ranked = panel.sort_values(["date", "prob"], ascending=[True, False])
    ranked["rk"] = ranked.groupby("date").cumcount()
    shorts = ranked[ranked["rk"] >= N - 6][["date", "ticker"]]

    conc = borrow_concentration(shorts, rates)
    assert conc["concentration_ratio"] > 1.5, conc
    assert conc["short_book_mean_rate"] > conc["universe_mean_rate"]


def test_per_name_borrow_costs_more_than_the_flat_equivalent() -> None:
    panel = _panel()
    flat = LongShortConfig(
        rebalance_every=20, slippage_bps=0.0, risk_free_rate=0.0,
        short_borrow_annual=GENERAL_COLLATERAL, benchmark_ticker=None,
    )
    per_name = replace(flat, per_name_borrow=True)
    a = run_long_short_backtest(panel, flat)
    b = run_long_short_backtest(panel, per_name)
    assert b.costs["borrow"] > a.costs["borrow"]
    assert b.metrics["total_return"] < a.metrics["total_return"]
    assert b.metrics["effective_borrow_rate"] > GENERAL_COLLATERAL


def test_effective_rate_is_reported_and_flat_when_not_per_name() -> None:
    res = run_long_short_backtest(
        _panel(),
        LongShortConfig(rebalance_every=20, short_borrow_annual=0.02,
                        benchmark_ticker=None),
    )
    assert res.metrics["effective_borrow_rate"] == pytest.approx(0.02)


def test_panel_supplied_rates_flow_through_the_engine() -> None:
    panel = _panel()
    panel["borrow_rate"] = 0.30
    res = run_long_short_backtest(
        panel,
        LongShortConfig(rebalance_every=20, slippage_bps=0.0, risk_free_rate=0.0,
                        short_borrow_annual=0.005, per_name_borrow=True,
                        benchmark_ticker=None),
    )
    assert res.metrics["effective_borrow_rate"] == pytest.approx(0.30, rel=0.02)
