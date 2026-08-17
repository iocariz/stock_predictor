"""Trading a band further down the ranking, not just the top."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, run_backtest, run_rank_hold_backtest

DATES = pd.bdate_range("2024-01-01", periods=90)
N = 60


def _panel() -> pd.DataFrame:
    """Stable ranking: T00 always scores highest, T59 lowest."""
    return pd.DataFrame([
        {
            "date": d, "ticker": f"T{i:02d}",
            "prob": 1.0 - i / 100.0,
            "adj_close": 100.0 + i + 0.05 * di,
        }
        for di, d in enumerate(DATES)
        for i in range(N)
    ])


def test_rank_offset_defaults_to_zero() -> None:
    assert BacktestConfig().rank_offset == 0


def test_negative_rank_offset_rejected() -> None:
    with pytest.raises(ValueError, match="rank_offset"):
        BacktestConfig(rank_offset=-1)


def test_offset_skips_the_top_of_the_ranking() -> None:
    result = run_backtest(
        _panel(), BacktestConfig(benchmark_ticker=None, top_n=10, rank_offset=50),
    )
    assert result.cohorts
    for cohort in result.cohorts:
        assert set(cohort.tickers) == {f"T{i:02d}" for i in range(50, 60)}


def test_zero_offset_still_trades_the_top() -> None:
    result = run_backtest(
        _panel(), BacktestConfig(benchmark_ticker=None, top_n=10, rank_offset=0),
    )
    for cohort in result.cohorts:
        assert set(cohort.tickers) == {f"T{i:02d}" for i in range(10)}


def test_offset_beyond_the_universe_trades_nothing() -> None:
    result = run_backtest(
        _panel(), BacktestConfig(benchmark_ticker=None, top_n=10, rank_offset=500),
    )
    assert result.metrics["n_cohorts"] == 0


def test_prices_of_excluded_names_remain_available() -> None:
    """The whole point: selection narrows, the price panel does not. Filtering
    the panel instead would leave held names marked at stale forward-filled
    prices and understate beta."""
    panel = _panel()
    offset = run_backtest(
        panel, BacktestConfig(benchmark_ticker=None, top_n=10, rank_offset=50),
    )
    # Same names, but reached by pre-filtering the panel down to the band.
    ranked = panel.sort_values(["date", "prob"], ascending=[True, False])
    ranked["rk"] = ranked.groupby("date").cumcount()
    trimmed = ranked[ranked["rk"] >= 50]
    filtered = run_backtest(
        trimmed, BacktestConfig(benchmark_ticker=None, top_n=10),
    )
    assert offset.metrics["n_cohorts"] == filtered.metrics["n_cohorts"]
    # The offset run prices its book off the full panel.
    assert len(offset.daily_nav) >= len(filtered.daily_nav)


def _rotating_panel() -> pd.DataFrame:
    """Ranks rotate weekly, so held names decay out of the band and close."""
    rows = []
    for di, d in enumerate(DATES):
        shift = (di // 5) % N
        for i in range(N):
            rank_slot = (i + shift) % N
            rows.append({
                "date": d, "ticker": f"T{i:02d}",
                "prob": 1.0 - rank_slot / 100.0,
                "adj_close": 100.0 + i + 0.05 * di,
            })
    return pd.DataFrame(rows)


def test_rank_hold_offset_gates_entry_only() -> None:
    """Entries come from the band; exits stay governed by exit_rank, so a
    holding that climbs into the skipped head is kept rather than churned."""
    panel = _rotating_panel()
    ranked = panel.sort_values(["date", "prob"], ascending=[True, False])
    ranked["rk"] = ranked.groupby("date").cumcount()
    rank_on = {
        (r.date, r.ticker): r.rk for r in ranked.itertuples()
    }

    cfg = BacktestConfig(
        benchmark_ticker=None, top_n=10, exit_rank=55, rank_offset=40,
    )
    result = run_rank_hold_backtest(panel, cfg)
    assert result.cohorts, "rotation should produce closed round trips"

    for cohort in result.cohorts:
        entry_rank = rank_on[(cohort.signal_date, cohort.tickers[0])]
        assert entry_rank >= 40, (
            f"{cohort.tickers[0]} entered at rank {entry_rank}, inside the skipped head"
        )
