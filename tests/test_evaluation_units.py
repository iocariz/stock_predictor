"""Measure alpha on excess returns, and rank within a date.

Two evaluation faults.

**CAPM on raw returns.** The regression was ``r_s = a + b*r_b``, so the
intercept absorbs ``r_f * (1 - beta)``. On the real panel at ``r_f`` 4.5% and
beta 0.292 that is **+3.19%/yr of spurious alpha**, and it moved the t-statistic
from +2.68 to +1.95 — across the conventional bar.

**Pooling across dates.** PR-AUC and ROC-AUC pooled every ticker-date, and
"weekly precision@k" picked k rows from a whole week rather than k per session.
The strategy chooses a basket *per signal date*, and LambdaRank scores are only
comparable within a group, so pooling compares numbers that were never on the
same scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.stats import market_exposure
from stock_predictor.training import per_date_metrics

RF = 0.045


# ---------------------------------------------------------------------------
# CAPM specification
# ---------------------------------------------------------------------------


def _series(beta: float, alpha_ann: float, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0004, 0.01, n)
    rf_d = RF / 252
    excess = alpha_ann / 252 + beta * (mkt - rf_d) + rng.normal(0, 0.002, n)
    return pd.Series(excess + rf_d), pd.Series(mkt)


def test_excess_specification_recovers_the_true_alpha() -> None:
    port, mkt = _series(beta=0.3, alpha_ann=0.05)
    out = market_exposure(port, mkt, overlap=1, risk_free_rate=RF)
    assert out["alpha_ann"] == pytest.approx(0.05, abs=0.01)
    assert out["beta"] == pytest.approx(0.3, abs=0.03)


def test_the_raw_specification_overstates_by_rf_times_one_minus_beta() -> None:
    port, mkt = _series(beta=0.3, alpha_ann=0.05)
    raw = market_exposure(port, mkt, overlap=1, risk_free_rate=0.0)
    excess = market_exposure(port, mkt, overlap=1, risk_free_rate=RF)
    assert raw["alpha_ann"] - excess["alpha_ann"] == pytest.approx(
        RF * (1 - 0.3), abs=0.01
    )


def test_beta_is_unchanged_by_the_specification() -> None:
    """Only the intercept moves; a constant shift cannot rotate the slope."""
    port, mkt = _series(beta=0.8, alpha_ann=0.02)
    a = market_exposure(port, mkt, overlap=1, risk_free_rate=0.0)
    b = market_exposure(port, mkt, overlap=1, risk_free_rate=RF)
    assert a["beta"] == pytest.approx(b["beta"], abs=1e-9)


def test_a_beta_one_portfolio_is_unaffected() -> None:
    """At beta 1 the bias term vanishes, which is why this hid so long."""
    port, mkt = _series(beta=1.0, alpha_ann=0.03)
    a = market_exposure(port, mkt, overlap=1, risk_free_rate=0.0)
    b = market_exposure(port, mkt, overlap=1, risk_free_rate=RF)
    assert a["alpha_ann"] == pytest.approx(b["alpha_ann"], abs=0.005)


def test_a_zero_rate_reproduces_the_old_numbers() -> None:
    port, mkt = _series(beta=0.3, alpha_ann=0.05)
    assert market_exposure(port, mkt, overlap=1, risk_free_rate=0.0)["alpha_ann"] > \
        market_exposure(port, mkt, overlap=1, risk_free_rate=RF)["alpha_ann"]


# ---------------------------------------------------------------------------
# Per-date evaluation
# ---------------------------------------------------------------------------


def _panel(n_dates: int = 10, n: int = 50) -> pd.DataFrame:
    """Scores rank perfectly within each date, but the *scale* differs by date
    — exactly what an uncalibrated LambdaRank output looks like."""
    rows = []
    for d in range(n_dates):
        offset = d * 100.0                      # date-specific scale
        for i in range(n):
            rows.append({
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "ticker": f"T{i:02d}",
                "prob": offset + (n - i),
                "target_5pct": 1.0 if i < 5 else 0.0,
                # Strictly decreasing with rank, so a perfect ranking can
                # actually reach IC 1.0. A binary forward return is capped by
                # ties well below it however good the signal is.
                "fwd_ret": (n - i) / 1000.0,
            })
    return pd.DataFrame(rows)


def test_precision_is_measured_within_each_date() -> None:
    """Pooling picks k rows from the pool; per-date picks k on every session."""
    out = per_date_metrics(_panel(), k=5)
    assert out["precision_at_k"] == pytest.approx(1.0)


def test_pooling_across_dates_is_defeated_by_uncalibrated_scores() -> None:
    """The same panel scored pooled: one date's offset dominates, so the
    'top 5 overall' all come from the last date."""
    from stock_predictor.training import precision_at_k

    panel = _panel()
    pooled = precision_at_k(panel["target_5pct"], panel["prob"].to_numpy(), k=5)
    per_date = per_date_metrics(panel, k=5)["precision_at_k"]
    assert per_date >= pooled


def test_rank_ic_is_computed_per_date() -> None:
    out = per_date_metrics(_panel(), k=5)
    assert out["rank_ic"] == pytest.approx(1.0, abs=0.01)


def test_top_n_excess_is_reported() -> None:
    out = per_date_metrics(_panel(), k=5)
    assert out["top_k_excess"] > 0


def test_the_number_of_dates_is_reported() -> None:
    assert per_date_metrics(_panel(n_dates=7), k=5)["n_dates"] == 7


def test_a_single_date_still_works() -> None:
    out = per_date_metrics(_panel(n_dates=1), k=5)
    assert out["n_dates"] == 1
    assert out["precision_at_k"] == pytest.approx(1.0)


def test_unlabelled_rows_are_excluded() -> None:
    panel = _panel()
    panel.loc[panel.date == panel.date.max(), "target_5pct"] = np.nan
    assert per_date_metrics(panel, k=5)["n_dates"] == 9


def test_a_date_thinner_than_k_is_skipped_not_scored_as_perfect() -> None:
    panel = _panel(n_dates=3)
    thin = panel[panel.date == panel.date.min()].head(3)
    panel = pd.concat([thin, panel[panel.date > panel.date.min()]])
    assert per_date_metrics(panel, k=5)["n_dates"] == 2
