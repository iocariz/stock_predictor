"""Risk-metric correctness: HAC alpha t-stats and a non-zero risk-free rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, _compute_metrics
from stock_predictor.backtest_reporting import _nav_only_metrics, relative_metrics

TRADING_DAYS = 252


def _nav_from_returns(returns: np.ndarray, start: float = 100_000.0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.Series(start * np.cumprod(np.r_[1.0, 1.0 + returns]), index=idx)


def _overlapping_returns(
    n: int, hold: int, seed: int, drift: float = 0.0,
) -> np.ndarray:
    """Daily returns of a book holding `hold`-day overlapping baskets.

    Each day's return is the average of the last `hold` independent shocks —
    exactly the autocorrelation a 10-day cohort strategy induces, and the
    reason an i.i.d. t-stat overstates significance.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, 0.01, n + hold)
    return np.array([shocks[i : i + hold].mean() for i in range(n)])


# ---------------------------------------------------------------------------
# Item 6: HAC (Newey-West) alpha t-statistic
# ---------------------------------------------------------------------------


def test_hac_lags_are_reported() -> None:
    rng = np.random.default_rng(0)
    strat = _nav_from_returns(rng.normal(0.0004, 0.01, 400))
    bench = _nav_from_returns(rng.normal(0.0003, 0.01, 400))
    rm = relative_metrics(strat, bench, overlap_days=10)
    assert rm["hac_lags"] >= 9
    assert np.isfinite(rm["alpha_t"])
    assert "alpha_t_iid" in rm


def test_hac_tstat_is_more_conservative_under_overlapping_holds() -> None:
    """Regression: alpha_t assumed i.i.d. daily returns, which a 10-day
    overlapping-hold book does not produce. Positively autocorrelated
    residuals inflate the naive t-stat."""
    strat = _nav_from_returns(_overlapping_returns(750, 10, seed=1, drift=0.0004))
    bench = _nav_from_returns(_overlapping_returns(750, 10, seed=2, drift=0.0003))

    rm = relative_metrics(strat, bench, overlap_days=10)
    assert abs(rm["alpha_t"]) < abs(rm["alpha_t_iid"]), (
        "HAC t-stat should shrink when residuals are positively autocorrelated"
    )


def test_hac_and_iid_agree_when_residuals_are_independent() -> None:
    """With genuinely i.i.d. residuals the correction should be small."""
    rng = np.random.default_rng(7)
    bench_r = rng.normal(0.0003, 0.01, 1500)
    strat_r = 1.0 * bench_r + rng.normal(0.0002, 0.004, 1500)
    rm = relative_metrics(
        _nav_from_returns(strat_r), _nav_from_returns(bench_r), overlap_days=1,
    )
    assert rm["alpha_t"] == pytest.approx(rm["alpha_t_iid"], rel=0.35)


def test_alpha_point_estimate_is_unchanged_by_the_hac_correction() -> None:
    """HAC changes the standard error, never the alpha itself."""
    rng = np.random.default_rng(3)
    strat = _nav_from_returns(rng.normal(0.0005, 0.01, 500))
    bench = _nav_from_returns(rng.normal(0.0002, 0.01, 500))
    a = relative_metrics(strat, bench, overlap_days=1)
    b = relative_metrics(strat, bench, overlap_days=20)
    assert a["alpha_ann"] == pytest.approx(b["alpha_ann"])
    assert a["beta"] == pytest.approx(b["beta"])
    assert a["hac_lags"] < b["hac_lags"]


def test_identical_series_still_yield_no_spurious_tstat() -> None:
    rng = np.random.default_rng(11)
    nav = _nav_from_returns(rng.normal(0.0003, 0.01, 300))
    rm = relative_metrics(nav, nav.copy(), overlap_days=10)
    assert not np.isfinite(rm["alpha_t"]) or abs(rm["alpha_t"]) < 1e-6


# ---------------------------------------------------------------------------
# Item 7: risk-free rate in Sharpe / Sortino
# ---------------------------------------------------------------------------


def test_positive_risk_free_rate_lowers_sharpe() -> None:
    rng = np.random.default_rng(5)
    nav = _nav_from_returns(rng.normal(0.0006, 0.01, 800))
    zero = _compute_metrics(nav, [], risk_free_rate=0.0)
    funded = _compute_metrics(nav, [], risk_free_rate=0.045)
    assert funded["sharpe"] < zero["sharpe"]
    assert funded["sortino"] < zero["sortino"]
    # Total return and drawdown are unaffected by the funding assumption.
    assert funded["total_return"] == pytest.approx(zero["total_return"])
    assert funded["max_drawdown"] == pytest.approx(zero["max_drawdown"])


def test_sharpe_is_zero_when_return_exactly_matches_cash() -> None:
    """A NAV compounding at the risk-free rate has no excess return."""
    rf = 0.05
    daily = (1 + rf) ** (1 / TRADING_DAYS) - 1
    nav = _nav_from_returns(np.full(600, daily))
    m = _compute_metrics(nav, [], risk_free_rate=rf)
    assert m["sharpe"] == pytest.approx(0.0, abs=1e-6) or np.isnan(m["sharpe"])


def test_nav_only_metrics_accepts_risk_free_rate() -> None:
    rng = np.random.default_rng(9)
    nav = _nav_from_returns(rng.normal(0.0006, 0.01, 600))
    assert (
        _nav_only_metrics(nav, risk_free_rate=0.045)["sharpe"]
        < _nav_only_metrics(nav, risk_free_rate=0.0)["sharpe"]
    )


def test_config_rejects_absurd_risk_free_rate() -> None:
    with pytest.raises(ValueError, match="risk_free_rate"):
        BacktestConfig(risk_free_rate=-0.5)


# ---------------------------------------------------------------------------
# Defaults: funding costs are on, and live matches the backtest
# ---------------------------------------------------------------------------


def test_funding_costs_are_charged_by_default() -> None:
    """A zero funding assumption flatters every risk-adjusted metric, so a
    panel with no realized rate still gets charged the cash proxy."""
    from stock_predictor.backtest import DEFAULT_RISK_FREE_RATE, run_backtest

    dates = pd.bdate_range("2024-01-01", periods=90)
    panel = pd.DataFrame([
        {"date": d, "ticker": f"T{i:02d}", "prob": 0.5 + i / 100,
         "adj_close": 100.0 + i + 0.05 * di}
        for di, d in enumerate(dates) for i in range(20)
    ])
    assert "irx_yield" not in panel.columns

    default = run_backtest(panel, BacktestConfig(benchmark_ticker=None))
    unfunded = run_backtest(
        panel, BacktestConfig(benchmark_ticker=None, risk_free_rate=0.0),
    )
    assert default.metrics["risk_free_rate_used"] == pytest.approx(DEFAULT_RISK_FREE_RATE)
    assert unfunded.metrics["risk_free_rate_used"] == 0.0
    assert default.metrics["sharpe"] < unfunded.metrics["sharpe"]


def test_panel_rate_overrides_the_scalar_default() -> None:
    """When the scored panel carries `irx_yield` (13-week T-bill, percent),
    the realized per-date series is used instead of the constant — correct
    across regimes rather than only the window the default was chosen for."""
    from stock_predictor.backtest import run_backtest

    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    for di, d in enumerate(dates):
        for i in range(20):
            rows.append({
                "date": d, "ticker": f"T{i:02d}", "prob": 0.5 + i / 100,
                "adj_close": 100.0 + i + 0.05 * di,
                "irx_yield": 0.10,  # 0.10% annual: near-ZIRP, unlike the 4.5% default
            })
    panel = pd.DataFrame(rows)

    with_panel_rate = run_backtest(panel, BacktestConfig(benchmark_ticker=None))
    pinned_high = run_backtest(
        panel, BacktestConfig(benchmark_ticker=None, risk_free_rate=0.045),
    )
    # Near-zero funding leaves more excess return than a 4.5% charge.
    assert with_panel_rate.metrics["sharpe"] > pinned_high.metrics["sharpe"]
    assert with_panel_rate.metrics["risk_free_rate_used"] == pytest.approx(0.001, abs=2e-4)


def test_explicit_zero_still_disables_the_funding_charge() -> None:
    rng = np.random.default_rng(5)
    nav = _nav_from_returns(rng.normal(0.0006, 0.01, 400))
    assert (
        _compute_metrics(nav, [], risk_free_rate=0.0)["sharpe"]
        > _compute_metrics(nav, [], risk_free_rate=0.045)["sharpe"]
    )
