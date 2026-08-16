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
