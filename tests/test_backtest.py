"""Unit tests for portfolio backtest (no live benchmark download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import (
    BacktestConfig,
    _build_cohort,
    _build_daily_nav,
    _compute_metrics,
    _compute_weights,
    _get_rebalance_dates,
    run_backtest,
)
from stock_predictor.execution_calendar import next_trading_day, offset_trading_days
from stock_predictor.backtest_reporting import (
    _nav_normalized_overlap,
    _nav_only_metrics,
    print_report,
    print_strategy_comparison,
)


def test_get_rebalance_dates_friday_unique_weeks() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    out = _get_rebalance_dates(dates.values, "Friday")
    weeks = {(d.isocalendar().year, d.isocalendar().week) for d in out}
    assert len(out) == len(weeks)
    assert all(d.dayofweek == 4 for d in out)


def test_get_rebalance_dates_last_trading_day_per_week() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    out = _get_rebalance_dates(dates.values, "last")
    by_week: dict[tuple[int, int], pd.Timestamp] = {}
    for d in dates:
        k = (d.isocalendar().year, d.isocalendar().week)
        by_week[k] = pd.Timestamp(d)
    assert out == sorted(by_week.values())


def test_compute_weights_equal_and_probability() -> None:
    w = _compute_weights(np.array([0.2, 0.8]), "equal")
    np.testing.assert_allclose(w, [0.5, 0.5])
    w2 = _compute_weights(np.array([0.2, 0.8]), "probability")
    np.testing.assert_allclose(w2, [0.2, 0.8])


def test_run_backtest_requires_columns() -> None:
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "ticker": ["AAA"]})
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(df, BacktestConfig(benchmark_ticker=None))


def test_run_backtest_requires_prob_column() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "ticker": ["AAA", "AAA"],
            "adj_close": [100.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="prob"):
        run_backtest(df, BacktestConfig(benchmark_ticker=None))


def test_run_backtest_empty() -> None:
    df = pd.DataFrame(columns=["date", "ticker", "prob", "adj_close"])
    with pytest.raises(ValueError, match="empty"):
        run_backtest(df, BacktestConfig(benchmark_ticker=None))


def test_run_backtest_probability_alias() -> None:
    days = pd.bdate_range("2024-01-08", periods=25)
    rows = []
    for d in days:
        for t in ("AAA", "BBB"):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "probability": 0.99 if t == "AAA" else 0.01,
                    "adj_close": 100.0,
                }
            )
    df = pd.DataFrame(rows)
    cfg = BacktestConfig(
        top_n=1,
        max_overlapping_cohorts=1,
        benchmark_ticker=None,
        slippage_bps=25.0,
        rebalance_day="Friday",
    )
    r = run_backtest(df, cfg)
    assert r.metrics["n_cohorts"] >= 1
    assert any(c.net_return < 0 for c in r.cohorts)
    assert len(r.spy_daily_nav) == 0


def test_backtest_config_rebalance_validation() -> None:
    with pytest.raises(ValueError, match="rebalance_day"):
        BacktestConfig(rebalance_day="Caturday")


def _two_ticker_panel(
    days: pd.DatetimeIndex,
    *,
    prefer: str,
) -> pd.DataFrame:
    rows = []
    for d in days:
        for t in ("AAA", "BBB"):
            p = 0.99 if t == prefer else 0.01
            rows.append({"date": d, "ticker": t, "prob": p, "adj_close": 100.0})
    return pd.DataFrame(rows)


def test_strategy_comparison_overlap() -> None:
    days = pd.bdate_range("2024-01-08", periods=30)
    cfg = BacktestConfig(
        top_n=1,
        max_overlapping_cohorts=1,
        benchmark_ticker=None,
        rebalance_day="Friday",
    )
    r_a = run_backtest(_two_ticker_panel(days, prefer="AAA"), cfg)
    r_b = run_backtest(_two_ticker_panel(days, prefer="BBB"), cfg)
    a_n, b_n, idx = _nav_normalized_overlap(r_a.daily_nav, r_b.daily_nav)
    assert len(idx) >= 2
    assert float(a_n.iloc[0]) == pytest.approx(1.0)
    assert float(b_n.iloc[0]) == pytest.approx(1.0)
    print_strategy_comparison(r_a, r_b, label_a="A", label_b="B")


# ---------------------------------------------------------------------------
# _build_cohort edge cases
# ---------------------------------------------------------------------------


def _make_price_panel(
    dates: pd.DatetimeIndex, tickers: list[str], price: float = 100.0
) -> pd.DataFrame:
    """Price panel where every cell is `price` (or NaN for missing)."""
    return pd.DataFrame(price, index=dates, columns=tickers)


def _make_scored_day(tickers: list[str], probs: list[float], date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "prob": probs, "date": date})


def test_build_cohort_missing_exit_price() -> None:
    """Ticker with NaN exit price is excluded; remaining tickers still form a cohort."""
    dates = pd.bdate_range("2024-01-08", periods=20)
    panel = _make_price_panel(dates, ["AAA", "BBB"])
    # Entry = dates[1], exit = dates[1+10] = dates[11]. Make BBB's exit price NaN.
    panel.loc[dates[11], "BBB"] = np.nan
    scored = _make_scored_day(["AAA", "BBB"], [0.9, 0.8], dates[0])
    cfg = BacktestConfig(top_n=2, holding_days=10, benchmark_ticker=None)
    trading = dates.values
    cohort = _build_cohort(dates[0], panel, scored, cfg, trading, 50_000.0)
    assert cohort is not None
    assert "BBB" not in cohort.tickers
    assert "AAA" in cohort.tickers
    assert len(cohort.tickers) == 1


def test_build_cohort_all_prices_missing_returns_none() -> None:
    dates = pd.bdate_range("2024-01-08", periods=20)
    panel = _make_price_panel(dates, ["AAA"], price=np.nan)
    scored = _make_scored_day(["AAA"], [0.9], dates[0])
    cfg = BacktestConfig(top_n=1, holding_days=10, benchmark_ticker=None)
    cohort = _build_cohort(dates[0], panel, scored, cfg, dates.values, 50_000.0)
    assert cohort is None


def test_build_cohort_exit_beyond_data_returns_none() -> None:
    dates = pd.bdate_range("2024-01-08", periods=5)  # too short for 10-day hold
    panel = _make_price_panel(dates, ["AAA"])
    scored = _make_scored_day(["AAA"], [0.9], dates[0])
    cfg = BacktestConfig(top_n=1, holding_days=10, benchmark_ticker=None)
    cohort = _build_cohort(dates[0], panel, scored, cfg, dates.values, 50_000.0)
    assert cohort is None


# ---------------------------------------------------------------------------
# _build_daily_nav correctness
# ---------------------------------------------------------------------------


def test_build_daily_nav_no_cohorts() -> None:
    """With no cohorts, NAV should be constant at initial capital."""
    dates = pd.bdate_range("2024-01-08", periods=10)
    panel = _make_price_panel(dates, ["AAA"])
    cfg = BacktestConfig(initial_capital=100_000.0, benchmark_ticker=None)
    nav = _build_daily_nav([], dates.values, panel, cfg)
    assert len(nav) == 10
    assert all(v == pytest.approx(100_000.0) for v in nav.values)


def test_build_daily_nav_known_return() -> None:
    """One cohort, one ticker, price doubles → NAV reflects the gain."""
    dates = pd.bdate_range("2024-01-08", periods=15)
    panel = _make_price_panel(dates, ["AAA"], price=100.0)
    # Entry on day 1, exit on day 11. Price goes from 100 to 200 at exit.
    panel.loc[dates[11], "AAA"] = 200.0
    from stock_predictor.backtest import Cohort

    cohort = Cohort(
        signal_date=dates[0],
        entry_date=dates[1],
        exit_date=dates[11],
        tickers=("AAA",),
        weights=(1.0,),
        entry_prices=(100.0,),  # no slippage for this test
        exit_prices=(200.0,),
        capital=50_000.0,
        gross_return=1.0,
        cost=0.0,
        net_return=1.0,
    )
    cfg = BacktestConfig(initial_capital=100_000.0, max_overlapping_cohorts=2, benchmark_ticker=None)
    nav = _build_daily_nav([cohort], dates.values, panel, cfg)
    # On exit day: cohort value = 50k * (200/100) = 100k. Cash = 50k. Total = 150k.
    assert nav[dates[11]] == pytest.approx(150_000.0)
    # Before entry: all cash = 100k
    assert nav[dates[0]] == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# VIX filter
# ---------------------------------------------------------------------------


def test_vix_filter_skips_high_vix_rebalance() -> None:
    days = pd.bdate_range("2024-01-08", periods=25)
    rows = []
    for d in days:
        for t in ("AAA", "BBB"):
            rows.append({
                "date": d,
                "ticker": t,
                "prob": 0.5,
                "adj_close": 100.0,
                "vix_percentile": 0.95,  # above filter
            })
    df = pd.DataFrame(rows)
    cfg = BacktestConfig(
        top_n=1,
        max_overlapping_cohorts=1,
        benchmark_ticker=None,
        vix_filter_percentile=0.8,
    )
    result = run_backtest(df, cfg)
    assert result.metrics["n_cohorts"] == 0


# ---------------------------------------------------------------------------
# _compute_metrics edge cases
# ---------------------------------------------------------------------------


def test_compute_metrics_flat_returns() -> None:
    nav = pd.Series([100.0] * 10, index=pd.bdate_range("2024-01-08", periods=10))
    m = _compute_metrics(nav, [])
    assert m["total_return"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert np.isnan(m["sharpe"])


# ---------------------------------------------------------------------------
# _nav_only_metrics
# ---------------------------------------------------------------------------


def test_nav_only_metrics_growing_nav() -> None:
    nav = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.bdate_range("2024-01-08", periods=4))
    m = _nav_only_metrics(nav)
    assert m["total_return"] > 0
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# print_report smoke
# ---------------------------------------------------------------------------


def test_print_report_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    days = pd.bdate_range("2024-01-08", periods=25)
    df = _two_ticker_panel(days, prefer="AAA")
    cfg = BacktestConfig(top_n=1, max_overlapping_cohorts=1, benchmark_ticker=None)
    result = run_backtest(df, cfg)
    print_report(result)
    captured = capsys.readouterr()
    assert "BACKTEST REPORT" in captured.out


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_next_trading_day() -> None:
    dates = pd.bdate_range("2024-01-08", periods=5).values
    # Monday → next is Tuesday
    nxt = next_trading_day(pd.Timestamp("2024-01-08"), dates)
    assert nxt == pd.Timestamp("2024-01-09")


def test_next_trading_day_beyond_end() -> None:
    dates = pd.bdate_range("2024-01-08", periods=3).values
    nxt = next_trading_day(pd.Timestamp("2024-01-10"), dates)
    assert nxt is None


def test_offset_trading_days() -> None:
    dates = pd.bdate_range("2024-01-08", periods=20).values
    result = offset_trading_days(pd.Timestamp("2024-01-08"), 5, dates)
    assert result == pd.Timestamp("2024-01-15")  # 5 bdays later


def test_build_cohort_commission_reduces_net_return() -> None:
    dates = pd.bdate_range("2024-01-08", periods=25)
    panel = _make_price_panel(dates, ["AAA", "BBB"])
    scored = _make_scored_day(["AAA", "BBB"], [0.5, 0.5], dates[0])
    cfg0 = BacktestConfig(top_n=2, holding_days=10, benchmark_ticker=None, commission_per_share=0.0)
    cfg1 = BacktestConfig(top_n=2, holding_days=10, benchmark_ticker=None, commission_per_share=1.0)
    c0 = _build_cohort(dates[0], panel, scored, cfg0, dates.values, 50_000.0)
    c1 = _build_cohort(dates[0], panel, scored, cfg1, dates.values, 50_000.0)
    assert c0 is not None and c1 is not None
    assert c1.net_return < c0.net_return
    assert c1.cost > c0.cost


def test_backtest_config_commission_validation() -> None:
    with pytest.raises(ValueError, match="commission"):
        BacktestConfig(commission_per_share=-0.01)


# ---------------------------------------------------------------------------
# Regression: NAV must compound realized P&L (cash-ledger accounting)
# ---------------------------------------------------------------------------


def _rising_market_panel(n_days: int = 120, n_tickers: int = 12) -> pd.DataFrame:
    """Every stock rises 1%/day → every cohort should be profitable."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_tickers):
        px = 100 * 1.01 ** np.arange(n_days)
        for d_i, d in enumerate(dates):
            rows.append({
                "date": d, "ticker": f"T{i}", "adj_close": px[d_i],
                "prob": rng.random(),
            })
    return pd.DataFrame(rows)


def test_nav_compounds_realized_pnl() -> None:
    """Profitable cohorts must accumulate in the equity curve.

    Regression: NAV used to reset to the fixed capital slice when a cohort
    exited, discarding all realized P&L (a string of +10% cohorts reported
    0.00% total return).
    """
    cfg = BacktestConfig(
        top_n=5, holding_days=10, benchmark_ticker=None, slippage_bps=0.0,
    )
    r = run_backtest(_rising_market_panel(), cfg)
    assert len(r.cohorts) >= 5
    assert all(c.net_return > 0 for c in r.cohorts)
    assert r.metrics["total_return"] > 0.3
    # Later cohorts are funded from compounded proceeds → larger capital.
    assert r.cohorts[-1].capital > r.cohorts[0].capital


def test_nav_holds_realized_value_after_exit() -> None:
    """NAV the day after a profitable exit must keep the realized gain."""
    dates = pd.bdate_range("2024-01-08", periods=15)
    panel = _make_price_panel(dates, ["AAA"], price=100.0)
    panel.loc[dates[11], "AAA"] = 200.0
    from stock_predictor.backtest import Cohort

    cohort = Cohort(
        signal_date=dates[0],
        entry_date=dates[1],
        exit_date=dates[11],
        tickers=("AAA",),
        weights=(1.0,),
        entry_prices=(100.0,),
        exit_prices=(200.0,),
        capital=50_000.0,
        gross_return=1.0,
        cost=0.0,
        net_return=1.0,
    )
    cfg = BacktestConfig(
        initial_capital=100_000.0, max_overlapping_cohorts=2, benchmark_ticker=None,
    )
    nav = _build_daily_nav([cohort], dates.values, panel, cfg)
    # Exit day realizes 50k * (1 + 1.0) = 100k; plus 50k never deployed.
    assert nav[dates[11]] == pytest.approx(150_000.0)
    # Regression: the gain must persist after the cohort closes.
    assert nav[dates[12]] == pytest.approx(150_000.0)
    assert nav[dates[-1]] == pytest.approx(150_000.0)


def test_exit_costs_hit_nav() -> None:
    """Commissions (realized at exit) must lower the final equity curve."""
    panel = _rising_market_panel(n_days=60, n_tickers=6)
    cfg0 = BacktestConfig(
        top_n=3, holding_days=10, benchmark_ticker=None, commission_per_share=0.0,
    )
    cfg1 = BacktestConfig(
        top_n=3, holding_days=10, benchmark_ticker=None, commission_per_share=1.0,
    )
    nav0 = run_backtest(panel, cfg0).daily_nav
    nav1 = run_backtest(panel, cfg1).daily_nav
    assert nav1.iloc[-1] < nav0.iloc[-1]
