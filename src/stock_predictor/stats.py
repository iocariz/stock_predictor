"""Heteroskedasticity- and autocorrelation-consistent (HAC) statistics.

Every return series this project reports is autocorrelated by construction:
labels are `horizon`-session forward returns sampled daily, and the portfolio
holds positions for `holding_days` sessions. Ordinary standard errors assume
i.i.d. observations and therefore overstate significance on exactly the
series we care about. These helpers apply a Bartlett-kernel Newey-West
correction instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def auto_hac_lags(n: int) -> int:
    """Newey–West's rule of thumb: floor(4 * (n/100)^(2/9))."""
    if n < 2:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients with a Newey–West HAC covariance matrix.

    *X* must already include an intercept column. The Bartlett kernel weights
    autocovariance j by ``1 - j/(lags+1)``, which keeps the estimate positive
    semi-definite.
    """
    n = X.shape[0]
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    xe = X * resid[:, None]
    s = xe.T @ xe
    for j in range(1, min(lags, n - 1) + 1):
        gamma = xe[j:].T @ xe[:-j]
        s = s + (1.0 - j / (lags + 1.0)) * (gamma + gamma.T)
    return beta, xtx_inv @ s @ xtx_inv


def hac_mean_tstat(x: np.ndarray, *, overlap: int = 1) -> tuple[float, float, int]:
    """t-statistic for ``mean(x) == 0`` with a HAC standard error.

    Regressing *x* on a constant makes the intercept the sample mean and its
    HAC variance the autocorrelation-corrected variance of that mean.

    *overlap* is the number of sessions successive observations share (e.g.
    the label horizon); the lag window is at least ``overlap - 1``.

    Returns ``(mean, t_stat, lags_used)``.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 3:
        return (float(v.mean()) if n else float("nan"), float("nan"), 0)
    lags = max(auto_hac_lags(n), max(int(overlap), 1) - 1)
    coef, cov = hac_ols(v, np.ones((n, 1)), lags)
    mean = float(coef[0])
    var = float(cov[0, 0])
    if not np.isfinite(var) or var <= 0:
        return mean, float("nan"), lags
    return mean, mean / float(np.sqrt(var)), lags


def downside_deviation(returns, target: float = 0.0) -> float:
    """Root-mean-square shortfall below *target*, over **all** observations.

    Sortino's denominator. Two things it is not:

    * Not the standard deviation of the negative returns. ``std`` demeans,
      so it measures dispersion around the average shortfall rather than
      distance from the target — a series of identical losses has zero
      dispersion and would give a denominator of 0 despite real downside.
    * Not restricted to the losing observations. Periods without a shortfall
      contribute zero to the sum but still count in the average, which is what
      makes a strategy with few losses score better than one with many of the
      same size.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    shortfall = np.minimum(arr - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def market_exposure(
    portfolio: "pd.Series | np.ndarray",
    benchmark: "pd.Series | np.ndarray",
    *,
    overlap: int = 1,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Beta and annualised alpha of *portfolio* against *benchmark*.

    Dollar-neutral is not market-neutral. Equalising notional equalises
    dollars, not exposure, and a book whose long leg holds higher-beta names
    than its short leg keeps a market position that a notional-based
    description hides. Measuring it is the only way to know.

    Standard errors are Newey--West, so overlapping holding periods do not
    inflate significance. Pass *risk_free_rate* to specify CAPM correctly; the
    default of 0 reproduces a raw-return regression.
    """
    y = np.asarray(getattr(portfolio, "to_numpy", lambda: portfolio)(), dtype=float)
    x = np.asarray(getattr(benchmark, "to_numpy", lambda: benchmark)(), dtype=float)
    n = min(len(y), len(x))
    y, x = y[:n], x[:n]
    nan = {"beta": float("nan"), "beta_t": float("nan"),
           "alpha_ann": float("nan"), "alpha_t": float("nan")}
    if n < 3 or not np.isfinite(y).all() or not np.isfinite(x).all():
        return nan
    # CAPM is specified on *excess* returns: (r_s - r_f) = a + b (r_b - r_f).
    # Regressing raw on raw leaves the intercept absorbing r_f * (1 - beta) --
    # on the real panel, +3.19%/yr of alpha that is really just the cash rate
    # on the un-invested fraction. It vanishes at beta 1, which is why a
    # long-only book barely showed it.
    rf_daily = risk_free_rate / 252.0
    y = y - rf_daily
    x = x - rf_daily
    design = np.column_stack([np.ones(n), x])
    lags = max(auto_hac_lags(n), max(0, overlap - 1))
    try:
        coef, cov = hac_ols(y, design, lags)
    except np.linalg.LinAlgError:
        return nan
    se = np.sqrt(np.diag(cov))
    return {
        "beta": float(coef[1]),
        "beta_t": float(coef[1] / se[1]) if se[1] > 0 else float("nan"),
        "alpha_ann": float(coef[0] * 252),
        "alpha_t": float(coef[0] / se[0]) if se[0] > 0 else float("nan"),
    }
