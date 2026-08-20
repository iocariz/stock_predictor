"""Sortino's denominator (P3).

Downside deviation is the root-mean-square shortfall *below the target*,
averaged over every observation. The previous code took the standard
deviation of the negative returns only, which is wrong twice: it restricts
the sample to the losses, and `std` demeans them — measuring dispersion
around the average shortfall rather than distance from the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import _compute_metrics
from stock_predictor.stats import downside_deviation


def _nav(returns: np.ndarray, start: float = 100_000.0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    return pd.Series(start * np.cumprod(np.r_[1.0, 1.0 + returns]), index=idx)


def test_matches_the_textbook_definition() -> None:
    r = pd.Series([-0.02, -0.01, 0.03, 0.04])
    assert downside_deviation(r) == pytest.approx(
        np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    )


def test_it_averages_over_every_observation_not_just_the_losses() -> None:
    """Adding a gain must dilute downside deviation; it did not before,
    because the gains were excluded from the denominator entirely."""
    losses_only = pd.Series([-0.02, -0.01])
    with_gain = pd.Series([-0.02, -0.01, 0.03, 0.04])
    assert downside_deviation(with_gain) < downside_deviation(losses_only)


def test_constant_losses_do_not_collapse_the_denominator() -> None:
    """The sharpest symptom: identical losses have zero *dispersion*, so the
    old denominator was 0 and Sortino blew up — despite real downside."""
    r = pd.Series([-0.01, -0.01, 0.05])
    assert r[r < 0].std(ddof=1) == pytest.approx(0.0), "the old denominator"
    assert downside_deviation(r) == pytest.approx(0.008165, rel=1e-3)


def test_no_downside_means_no_denominator() -> None:
    assert downside_deviation(pd.Series([0.01, 0.02, 0.0])) == pytest.approx(0.0)


def test_a_custom_target_shifts_the_shortfall() -> None:
    r = pd.Series([0.01, 0.02, 0.03])
    assert downside_deviation(r, target=0.0) == pytest.approx(0.0)
    assert downside_deviation(r, target=0.025) > 0


def test_symmetric_returns_give_roughly_volatility_over_root_two() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 200_000))
    assert downside_deviation(r) == pytest.approx(r.std() / np.sqrt(2), rel=0.02)


# ---------------------------------------------------------------------------
# Through the metrics
# ---------------------------------------------------------------------------


def test_sortino_exceeds_sharpe_when_downside_is_the_smaller_half() -> None:
    """Right-skewed returns: losses are small and frequent, gains large."""
    r = np.array([-0.002] * 180 + [0.05] * 20)
    m = _compute_metrics(_nav(r), [], risk_free_rate=0.0)
    assert m["sortino"] > m["sharpe"]


def test_sortino_is_finite_when_every_loss_is_the_same_size() -> None:
    r = np.array([-0.01, -0.01, 0.05] * 60)
    m = _compute_metrics(_nav(r), [], risk_free_rate=0.0)
    assert np.isfinite(m["sortino"]), "the old denominator was exactly zero here"
    assert m["sortino"] > 0


def test_a_downside_free_series_has_no_sortino() -> None:
    m = _compute_metrics(_nav(np.full(100, 0.001)), [], risk_free_rate=0.0)
    assert np.isnan(m["sortino"]), "no downside, no ratio"


def test_reporting_path_uses_the_same_definition() -> None:
    from stock_predictor.backtest_reporting import _nav_only_metrics

    r = np.array([-0.002] * 180 + [0.05] * 20)
    nav = _nav(r)
    assert _nav_only_metrics(nav)["sortino"] == pytest.approx(
        _compute_metrics(nav, [], risk_free_rate=0.0)["sortino"]
    )
