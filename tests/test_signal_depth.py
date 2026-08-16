"""Selection-depth diagnostics: does the signal work where the strategy trades?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.signal_depth import (
    daily_bucket_returns,
    depth_frame,
    depth_table,
    format_depth_table,
    is_signal_monotone,
    rank_ic,
)
from stock_predictor.stats import hac_mean_tstat

DATES = pd.bdate_range("2024-01-01", periods=200)
N_TICKERS = 60


def _panel(fwd_of_rank, *, seed: int = 0) -> pd.DataFrame:
    """Panel whose forward return is a chosen function of within-date rank."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in DATES:
        order = rng.permutation(N_TICKERS)
        for rank, ticker_idx in enumerate(order, start=1):
            rows.append({
                "date": d,
                "ticker": f"T{ticker_idx:03d}",
                # Higher score == better rank (rank 1 is the top pick).
                "prob": 1.0 - rank / (N_TICKERS + 1),
                "fwd_ret": fwd_of_rank(rank) + rng.normal(0, 0.002),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bucket mechanics
# ---------------------------------------------------------------------------


def test_top_bucket_selects_highest_scores() -> None:
    panel = _panel(lambda r: 0.01 if r <= 5 else 0.0)
    top5 = daily_bucket_returns(panel, 5)
    assert len(top5) == len(DATES)
    assert top5.mean() == pytest.approx(0.01, abs=5e-4)


def test_bottom_bucket_selects_lowest_scores() -> None:
    panel = _panel(lambda r: -0.02 if r > N_TICKERS - 5 else 0.0)
    bottom5 = daily_bucket_returns(panel, 5, from_bottom=True)
    assert bottom5.mean() == pytest.approx(-0.02, abs=5e-4)


def test_missing_column_raises() -> None:
    panel = _panel(lambda r: 0.0).drop(columns=["fwd_ret"])
    with pytest.raises(ValueError, match="fwd_ret"):
        depth_table(panel)


# ---------------------------------------------------------------------------
# The shape diagnostic
# ---------------------------------------------------------------------------


def test_skilful_ranker_is_monotone_and_top_heavy() -> None:
    """Forward return decreasing in rank: the tightest bucket should win."""
    panel = _panel(lambda r: 0.02 * (1.0 - r / N_TICKERS))
    rows = depth_table(panel, buckets=(5, 15, 50))
    by = {r.label: r for r in rows}
    assert by["top 5"].mean_fwd_ret > by["top 15"].mean_fwd_ret > by["top 50"].mean_fwd_ret
    assert by["top 5"].excess_vs_universe > 0
    assert is_signal_monotone(rows)


def test_hollow_top_is_detected() -> None:
    """Regression for the real finding: the mid band carries a mild edge while
    the extreme top underperforms the universe. A backtest can still look
    acceptable; the ranking is not usable where it is traded."""
    def fwd(r: int) -> float:
        if r <= 5:
            return -0.002       # extreme top is actively bad
        if r <= 40:
            return 0.010        # mid band carries the edge
        return 0.002            # weak tail

    panel = _panel(fwd, seed=3)
    rows = depth_table(panel, buckets=(5, 15, 50))
    by = {r.label: r for r in rows}
    assert by["top 5"].excess_vs_universe < 0
    assert by["top 50"].excess_vs_universe > 0
    # The traded end is the weak end, even though deeper selection helps.
    assert by["top 5"].mean_fwd_ret < by["top 50"].mean_fwd_ret
    assert not is_signal_monotone(rows), "hollow top should fail the shape check"


def test_universe_row_is_the_baseline() -> None:
    panel = _panel(lambda r: 0.02 * (1.0 - r / N_TICKERS))
    rows = depth_table(panel, buckets=(5, 15))
    uni = next(r for r in rows if r.label == "universe")
    assert uni.excess_vs_universe == 0.0
    assert uni.mean_fwd_ret == pytest.approx(
        panel.groupby("date")["fwd_ret"].mean().mean()
    )


def test_bottom_bucket_included_by_default() -> None:
    rows = depth_table(_panel(lambda r: 0.0), buckets=(5, 20))
    assert any(r.label == "bottom 20" for r in rows)
    rows_off = depth_table(_panel(lambda r: 0.0), buckets=(5, 20), include_bottom=False)
    assert not any(r.label.startswith("bottom") for r in rows_off)


# ---------------------------------------------------------------------------
# Rank IC
# ---------------------------------------------------------------------------


def test_rank_ic_positive_for_a_good_signal() -> None:
    ic = rank_ic(_panel(lambda r: 0.02 * (1.0 - r / N_TICKERS)))
    assert ic["mean"] > 0.5
    assert ic["t"] > 2
    assert ic["n_days"] == len(DATES)


def test_rank_ic_near_zero_for_noise() -> None:
    rng = np.random.default_rng(11)
    rows = [
        {"date": d, "ticker": f"T{i:03d}", "prob": rng.random(), "fwd_ret": rng.normal(0, 0.02)}
        for d in DATES for i in range(N_TICKERS)
    ]
    ic = rank_ic(pd.DataFrame(rows))
    assert abs(ic["mean"]) < 0.05
    assert abs(ic["t"]) < 2


# ---------------------------------------------------------------------------
# HAC correction and rendering
# ---------------------------------------------------------------------------


def test_hac_tstat_shrinks_under_overlap() -> None:
    """Overlapping windows inflate a naive t-stat; the HAC lag floor is what
    keeps the depth table honest."""
    rng = np.random.default_rng(4)
    shocks = rng.normal(0.0006, 0.01, 800 + 10)
    overlapping = np.array([shocks[i : i + 10].mean() for i in range(800)])
    _, t_hac, lags = hac_mean_tstat(overlapping, overlap=10)
    t_naive = overlapping.mean() / overlapping.std(ddof=1) * np.sqrt(len(overlapping))
    assert lags >= 9
    assert abs(t_hac) < abs(t_naive)


def test_table_renders_and_exports() -> None:
    rows = depth_table(_panel(lambda r: 0.02 * (1.0 - r / N_TICKERS)), buckets=(5, 15))
    text = format_depth_table(rows)
    assert "top 5" in text and "universe" in text and "HAC t" in text
    df = depth_frame(rows)
    assert list(df.columns) == [
        "bucket", "n_names", "mean_fwd_ret", "excess_vs_universe", "excess_hac_t", "n_days",
    ]
    assert len(df) == len(rows)
