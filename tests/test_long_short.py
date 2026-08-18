"""Dollar-neutral long-short engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_predictor.long_short import (
    LongShortConfig,
    borrow_sensitivity,
    run_long_short_backtest,
)

DATES = pd.bdate_range("2024-01-01", periods=260)
N = 60  # a decile is 6 names, above the default min_names_per_side
FREE = LongShortConfig(
    slippage_bps=0.0, short_borrow_annual=0.0, risk_free_rate=0.0,
    rebalance_every=20, benchmark_ticker=None,
)


def _panel(edge: float, *, seed: int = 0, inverted: bool = False) -> pd.DataFrame:
    """Panel where score predicts forward drift with strength *edge*.

    Ticker i has a fixed score; its price drifts at a rate proportional to
    that score, so a decile book should capture exactly the spread.
    """
    rng = np.random.default_rng(seed)
    rank = np.arange(N) / (N - 1) - 0.5           # -0.5 .. +0.5
    drift = (-rank if inverted else rank) * edge
    noise = rng.normal(0, 0.002, (len(DATES), N))
    px = 100 * np.exp(np.cumsum(drift + noise, axis=0))
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            rows.append({"date": d, "ticker": f"T{i:02d}",
                         "prob": float(rank[i]), "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_config_rejects_nonsense() -> None:
    for bad in (dict(decile=0.0), dict(decile=0.9), dict(rebalance_every=0),
                dict(short_borrow_annual=-0.01), dict(long_weight=-1.0)):
        with pytest.raises(ValueError):
            LongShortConfig(**bad)


def test_book_is_dollar_neutral_and_gross_matches_weights() -> None:
    """Equal notional per side, and gross exposure equal to the configured sum."""
    from stock_predictor.long_short import _target_book

    day = pd.DataFrame({"ticker": [f"T{i:02d}" for i in range(60)],
                        "prob": np.linspace(1, 0, 60)})
    book = _target_book(day, LongShortConfig(), capital=100_000.0)
    longs = sum(v for v in book.values() if v > 0)
    shorts = -sum(v for v in book.values() if v < 0)
    assert longs == pytest.approx(50_000.0)
    assert shorts == pytest.approx(50_000.0)
    assert longs - shorts == pytest.approx(0.0), "book must be dollar-neutral"
    assert longs + shorts == pytest.approx(100_000.0), "1.0x gross by default"


def test_thin_universe_produces_no_book() -> None:
    from stock_predictor.long_short import _target_book

    day = pd.DataFrame({"ticker": ["A", "B"], "prob": [1.0, 0.0]})
    assert _target_book(day, LongShortConfig(), 100_000.0) == {}


# ---------------------------------------------------------------------------
# Does it capture the signal?
# ---------------------------------------------------------------------------


def test_a_real_edge_makes_money() -> None:
    res = run_long_short_backtest(_panel(edge=0.0006), FREE)
    assert res.metrics["total_return"] > 0.05
    assert res.n_rebalances > 5


def test_an_inverted_signal_loses() -> None:
    res = run_long_short_backtest(_panel(edge=0.0006, inverted=True), FREE)
    assert res.metrics["total_return"] < 0


def test_no_edge_is_roughly_flat() -> None:
    res = run_long_short_backtest(_panel(edge=0.0), FREE)
    assert abs(res.metrics["total_return"]) < 0.05


def test_costless_run_tracks_the_gross_decile_spread() -> None:
    """The engine must reproduce the measure it was built to make tradable.

    At 2.0x gross (1.0 per side) the book *is* the decile spread, so a
    frictionless run should land close to compounding that spread.
    """
    from stock_predictor.signal_depth import daily_bucket_returns

    panel = _panel(edge=0.0006)
    cfg = replace(FREE, long_weight=1.0, short_weight=1.0)
    res = run_long_short_backtest(panel, cfg)

    n = int(panel.groupby("date").size().median() * cfg.decile)
    fwd = panel.copy()
    fwd["fwd_ret"] = fwd.groupby("ticker")["adj_close"].transform(
        lambda s: s.shift(-cfg.rebalance_every) / s - 1.0
    )
    spread = (daily_bucket_returns(fwd, n) -
              daily_bucket_returns(fwd, n, from_bottom=True)).dropna()
    periods = len(DATES) / cfg.rebalance_every
    implied = (1 + spread.mean()) ** periods - 1

    assert res.metrics["total_return"] == pytest.approx(implied, rel=0.5), (
        res.metrics["total_return"], implied,
    )


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_borrow_cost_reduces_return_and_scales_with_the_rate() -> None:
    panel = _panel(edge=0.0006)
    free = run_long_short_backtest(panel, FREE)
    cheap = run_long_short_backtest(panel, replace(FREE, short_borrow_annual=0.02))
    dear = run_long_short_backtest(panel, replace(FREE, short_borrow_annual=0.20))
    assert free.metrics["total_return"] > cheap.metrics["total_return"]
    assert cheap.metrics["total_return"] > dear.metrics["total_return"]
    assert dear.costs["borrow"] > cheap.costs["borrow"] > 0


def test_borrow_is_charged_on_short_notional_not_capital() -> None:
    """Doubling short size roughly doubles the borrow bill."""
    panel = _panel(edge=0.0)
    small = run_long_short_backtest(panel, replace(FREE, short_borrow_annual=0.05))
    big = run_long_short_backtest(
        panel, replace(FREE, short_borrow_annual=0.05, short_weight=1.0),
    )
    assert big.costs["borrow"] == pytest.approx(2 * small.costs["borrow"], rel=0.25)


def test_slippage_and_commission_are_charged_on_both_sides() -> None:
    panel = _panel(edge=0.0006)
    free = run_long_short_backtest(panel, FREE)
    costly = run_long_short_backtest(
        panel, replace(FREE, slippage_bps=25.0, commission_per_order=1.0),
    )
    assert costly.costs["slippage"] > 0
    assert costly.costs["commission"] > 0
    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_turnover_only_charges_the_change_in_the_book() -> None:
    """A ranking that never moves should trade once and then stop."""
    panel = _panel(edge=0.0)
    res = run_long_short_backtest(panel, replace(FREE, slippage_bps=10.0))
    traded = res.turnover[res.turnover > 0]
    assert len(traded) >= 1
    # The opening trade must dominate; later rebalances only drift-correct.
    assert traded.iloc[0] > 5 * traded.iloc[1:].max()


def test_financing_credits_cash_at_the_risk_free_rate() -> None:
    panel = _panel(edge=0.0)
    unfunded = run_long_short_backtest(panel, FREE)
    funded = run_long_short_backtest(panel, replace(FREE, risk_free_rate=0.05))
    assert funded.costs["financing_earned"] > 0
    assert unfunded.costs["financing_earned"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_nav_identity_holds_and_series_are_aligned() -> None:
    res = run_long_short_backtest(_panel(edge=0.0006), FREE)
    assert len(res.daily_nav) == len(DATES)
    assert res.daily_nav.notna().all()
    assert res.daily_nav.iloc[0] == pytest.approx(FREE.initial_capital, rel=0.02)


def test_borrow_sensitivity_is_monotone_in_the_rate() -> None:
    out = borrow_sensitivity(_panel(edge=0.0006), FREE, rates=(0.0, 0.02, 0.10))
    assert list(out["short_borrow_annual"]) == [0.0, 0.02, 0.10]
    assert out["total_return"].is_monotonic_decreasing
    assert out["borrow_cost"].is_monotonic_increasing
