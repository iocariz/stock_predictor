"""Portfolio backtest for the S&P 500 forward-return strategy.

Simulates a weekly-rebalance, long-only strategy that buys top-N stocks
ranked by predicted probability.  Tracks overlapping cohorts, models
next-day entry with configurable slippage, and compares against SPY
buy-and-hold.

Example:
  uv run python backtest.py artifacts/wf_scored.parquet --plots-dir artifacts/plots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.execution_calendar import next_trading_day, offset_trading_days


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestConfig:
    top_n: int = 15
    holding_days: int = 10
    rebalance_day: str = "Friday"  # or "last" for last trading day of ISO week
    weighting: str = "equal"  # "equal" or "probability"
    slippage_bps: float = 5.0  # per side
    initial_capital: float = 100_000.0
    max_overlapping_cohorts: int = 2
    vix_filter_percentile: float | None = None
    vix_scale_exposure: bool = False
    """Scale each cohort's capital by the VIX regime instead of (or on top of)
    the hard skip: full size while vix_percentile <= 0.5, then linearly down
    to zero at vix_percentile = 1.0 (scale = clip(2 * (1 - pct), 0, 1)).
    Unallocated capital stays in cash. Requires a vix_percentile column."""
    benchmark_ticker: str | None = "SPY"
    """Download this ticker for buy-and-hold comparison; None skips benchmark (no network)."""
    commission_per_share: float = 0.0
    """Dollars per share per buy or sell leg (round-trip = 2× per name)."""
    commission_per_order: float = 0.0
    """Flat dollars per ticker per buy or sell order (round-trip = 2× per name)."""
    exit_rank: int = 40
    """Rank-hold mode only (:func:`run_rank_hold_backtest`): sell a held name
    when its cross-sectional rank decays beyond this (or it leaves the scored
    universe). Must be >= top_n; larger values -> lower turnover."""
    risk_free_rate: float = 0.0
    """Annualized risk-free rate used for Sharpe and Sortino (0.045 = 4.5%).

    Default 0.0 preserves historical numbers, but a zero funding assumption
    flatters risk-adjusted metrics over any period when cash actually paid —
    2023-2026 cash yielded 4-5%. Total return, CAGR and drawdown are
    unaffected."""
    min_prob: float | None = None
    """Score floor: never buy a name scoring below this, even if it is in the
    top_n. Baskets shrink (weights renormalize over survivors) and a date with
    no eligible name simply does not trade. ``None`` disables the floor.

    The floor is compared against the panel's raw ``prob`` column. For a
    classifier that is a probability, but ``scale_pos_weight`` leaves it
    uncalibrated, and for ``--rank-objective`` it is an unbounded lambdarank
    score — so a threshold is only comparable across runs of the same model
    family and configuration."""

    def __post_init__(self) -> None:
        if self.weighting not in ("equal", "probability"):
            raise ValueError(f"weighting must be 'equal' or 'probability', got {self.weighting!r}")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.holding_days < 1:
            raise ValueError("holding_days must be >= 1")
        if self.exit_rank < self.top_n:
            raise ValueError(
                f"exit_rank ({self.exit_rank}) must be >= top_n ({self.top_n})"
            )
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "last"}
        if self.rebalance_day not in valid_days:
            raise ValueError(f"rebalance_day must be one of {valid_days}, got {self.rebalance_day!r}")
        if self.commission_per_share < 0 or self.commission_per_order < 0:
            raise ValueError("commission_per_share and commission_per_order must be >= 0")
        if not np.isfinite(self.risk_free_rate) or not -0.01 <= self.risk_free_rate <= 1.0:
            raise ValueError(
                f"risk_free_rate must be an annual rate in [-0.01, 1.0], got {self.risk_free_rate!r}"
            )
        if self.min_prob is not None and not np.isfinite(self.min_prob):
            raise ValueError(f"min_prob must be a finite number or None, got {self.min_prob!r}")


@dataclass(frozen=True)
class Cohort:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    entry_prices: tuple[float, ...]
    exit_prices: tuple[float, ...]
    capital: float
    gross_return: float
    cost: float
    net_return: float


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    daily_nav: pd.Series
    daily_returns: pd.Series
    cohorts: tuple[Cohort, ...]
    spy_daily_nav: pd.Series
    spy_daily_returns: pd.Series
    metrics: dict[str, float]
    spy_metrics: dict[str, float]
    start_date: pd.Timestamp
    end_date: pd.Timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_rebalance_dates(
    trading_dates: np.ndarray, rebalance_day: str,
) -> list[pd.Timestamp]:
    """Return one date per ISO week: the specified weekday, or last trading day in data."""
    dates = pd.DatetimeIndex(trading_dates)
    weeks = dates.isocalendar()
    week_key = list(zip(weeks.year, weeks.week))  # (iso_year, iso_week)
    if rebalance_day == "last":
        groups: dict[tuple[int, int], pd.Timestamp] = {}
        for d, k in zip(dates, week_key):
            groups[k] = d
        return sorted(groups.values())
    target_dow = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
    }[rebalance_day]
    result: list[pd.Timestamp] = []
    seen: set[tuple[int, int]] = set()
    for d, k in zip(dates, week_key):
        if d.dayofweek == target_dow and k not in seen:
            result.append(d)
            seen.add(k)
    return result


def _compute_weights(probs: np.ndarray, weighting: str) -> np.ndarray:
    """Long-only portfolio weights summing to 1.

    ``probability`` weighting normalizes scores by their sum, which is only
    meaningful for non-negative scores. Raw lambdarank output straddles zero:
    ``p / sum(p)`` then yields negative weights (an implicit short in a
    long-only book) and, when the scores nearly cancel, unbounded leverage —
    e.g. ``[1.0, -0.99, 0.01]`` produced ``[50.0, -49.5, 0.5]``. Reject those
    inputs instead of silently trading them.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0)
    if weighting != "probability":
        return np.ones(n) / n
    if np.any(probs < 0):
        raise ValueError(
            "weighting='probability' requires non-negative scores, got "
            f"min={float(np.min(probs)):.6g}. Raw lambdarank scores are not "
            "probabilities — use weighting='equal' with --rank-objective "
            "models, or a classifier whose scores are in [0, 1]."
        )
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(n) / n
    return probs / total


def _apply_slippage(price: float, slippage_bps: float, direction: int) -> float:
    """direction: +1 for buy, -1 for sell."""
    return price * (1 + direction * slippage_bps / 10_000)


def _cohort_commission_dollars(
    weights: np.ndarray,
    capital: float,
    entry_slipped: list[float],
    commission_per_share: float,
    commission_per_order: float,
) -> float:
    """Estimated round-trip commission for a cohort (fractional shares)."""
    total = 0.0
    for w, ep in zip(weights, entry_slipped, strict=True):
        shares = w * capital / ep
        total += 2.0 * shares * commission_per_share
    n = len(entry_slipped)
    total += 2.0 * n * commission_per_order
    return total


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------


def _build_cohort(
    signal_date: pd.Timestamp,
    price_panel: pd.DataFrame,
    scored_day: pd.DataFrame,
    config: BacktestConfig,
    trading_dates: np.ndarray,
    capital: float,
) -> Cohort | None:
    entry_date = next_trading_day(signal_date, trading_dates)
    if entry_date is None:
        return None
    exit_date = offset_trading_days(entry_date, config.holding_days, trading_dates)
    if exit_date is None:
        return None

    if config.min_prob is not None:
        scored_day = scored_day[scored_day["prob"] >= config.min_prob]
    top = scored_day.nlargest(config.top_n, "prob")
    if len(top) == 0:
        return None

    tickers: list[str] = []
    entry_prices: list[float] = []
    exit_prices: list[float] = []
    probs: list[float] = []

    for _, row in top.iterrows():
        t = row["ticker"]
        try:
            ep = price_panel.at[entry_date, t]
            xp = price_panel.at[exit_date, t]
        except KeyError:
            continue
        if np.isnan(ep) or np.isnan(xp):
            continue
        tickers.append(t)
        entry_prices.append(_apply_slippage(ep, config.slippage_bps, +1))
        exit_prices.append(_apply_slippage(xp, config.slippage_bps, -1))
        probs.append(row["prob"])

    if len(tickers) == 0:
        return None

    weights = _compute_weights(np.array(probs), config.weighting)
    gross_ret = sum(w * (xp / ep - 1) for w, ep, xp in zip(
        weights,
        [price_panel.at[entry_date, t] for t in tickers],
        [price_panel.at[exit_date, t] for t in tickers],
    ))
    net_ret = sum(w * (xp / ep - 1) for w, ep, xp in zip(weights, entry_prices, exit_prices))
    slip_cost = capital * sum(w * 2 * config.slippage_bps / 10_000 for w in weights)
    comm = _cohort_commission_dollars(
        weights, capital, list(entry_prices),
        config.commission_per_share, config.commission_per_order,
    )
    net_ret = net_ret - comm / capital
    cost = slip_cost + comm

    return Cohort(
        signal_date=signal_date,
        entry_date=entry_date,
        exit_date=exit_date,
        tickers=tuple(tickers),
        weights=tuple(weights.tolist()),
        entry_prices=tuple(entry_prices),
        exit_prices=tuple(exit_prices),
        capital=capital,
        gross_return=gross_ret,
        cost=cost,
        net_return=net_ret,
    )


# ---------------------------------------------------------------------------
# Daily NAV
# ---------------------------------------------------------------------------


def _build_daily_nav(
    cohorts: list[Cohort],
    trading_dates: np.ndarray,
    price_panel: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    """Cash-ledger NAV: capital leaves cash at entry and returns at exit.

    Each cohort's capital is debited from cash on its entry date and credited
    back on its exit date as ``capital * (1 + net_return)`` — i.e. realized
    P&L (including exit slippage and commissions) compounds into cash instead
    of being discarded.  While a cohort is open (entry day through the day
    before exit) it is marked to market against its slipped entry prices.
    """
    nav_index = pd.DatetimeIndex(trading_dates)
    n_days = len(trading_dates)

    if not cohorts:
        return pd.Series(config.initial_capital, index=nav_index, dtype=float)

    cash_flow = np.zeros(n_days)
    invested = np.zeros(n_days)

    for c in cohorts:
        i0 = int(np.searchsorted(trading_dates, c.entry_date, side="left"))
        if i0 >= n_days:
            continue
        i_exit = int(np.searchsorted(trading_dates, c.exit_date, side="left"))

        cash_flow[i0] -= c.capital
        if i_exit < n_days:
            cash_flow[i_exit] += c.capital * (1.0 + c.net_return)
        # Mark-to-market window: entry day through the day before exit; if the
        # exit falls beyond the calendar, hold the mark through the last day.
        i1 = min(i_exit - 1, n_days - 1)
        if i0 > i1:
            continue
        sl = slice(i0, i1 + 1)

        weights = np.array(c.weights)
        entry_prices = np.array(c.entry_prices)
        tickers = list(c.tickers)

        # Vectorized price extraction for the active window
        present = [t for t in tickers if t in price_panel.columns]
        if not present:
            invested[sl] += c.capital
            continue

        prices = price_panel.loc[nav_index[sl], present].values.copy()
        rf = np.ones((i1 - i0 + 1, len(tickers)))
        for ti, t in enumerate(tickers):
            if t in present:
                col = prices[:, present.index(t)]
                nans = np.isnan(col)
                col[nans] = entry_prices[ti]  # flat if missing
                rf[:, ti] = col / entry_prices[ti]

        invested[sl] += c.capital * (rf @ weights)

    cash = config.initial_capital + np.cumsum(cash_flow)
    return pd.Series(cash + invested, index=nav_index, dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


# Below this, a return series is constant to floating-point precision and
# risk-adjusted ratios are noise amplification rather than signal.
_FLAT_EPS = 1e-12


def daily_risk_free(annual_rate: float) -> float:
    """Compounded daily equivalent of an annualized rate."""
    if not annual_rate:
        return 0.0
    return (1.0 + annual_rate) ** (1.0 / 252.0) - 1.0


def _compute_metrics(
    nav: pd.Series,
    cohorts: list[Cohort],
    *,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days == 0:
        return {}
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n_days, 1)) - 1
    # Sharpe and Sortino are excess-return statistics; total return, CAGR and
    # drawdown deliberately stay on raw returns.
    excess = daily_ret - daily_risk_free(risk_free_rate)
    mean_r = excess.mean()
    std_r = excess.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > _FLAT_EPS else float("nan")
    downside = excess[excess < 0].std()
    sortino = (mean_r / downside * np.sqrt(252)) if downside > _FLAT_EPS else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = drawdown.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    total_costs = sum(c.cost for c in cohorts)
    wins = sum(1 for c in cohorts if c.net_return > 0)
    n_cohorts = len(cohorts)
    win_rate = wins / n_cohorts if n_cohorts > 0 else float("nan")

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "total_costs": total_costs,
        "win_rate": win_rate,
        "n_cohorts": n_cohorts,
    }


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------


def _download_benchmark(
    start: pd.Timestamp, end: pd.Timestamp, ticker: str, initial_capital: float,
    *, provider: object | None = None,
) -> tuple[pd.Series, pd.Series]:
    try:
        if provider is None:
            # Default to the yfinance provider (includes rate-limit retry).
            from stock_predictor.providers.yfinance_provider import YFinanceProvider

            provider = YFinanceProvider()
        close = provider.download_benchmark(
            ticker, str(start.date()), str(end.date()),
        )
        if close.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        nav = initial_capital * close / float(close.iloc[0])
        daily_ret = nav.pct_change().dropna()
        return nav, daily_ret
    except Exception as exc:
        print(f"Benchmark download failed ({ticker}): {exc}")
        empty = pd.Series(dtype=float)
        return empty, empty


def _align_benchmark_to_nav(
    bench_nav: pd.Series, strategy_index: pd.DatetimeIndex, initial_capital: float,
) -> pd.Series:
    """Reindex benchmark closes to strategy trading days and rescale to initial capital."""
    if bench_nav.empty or len(strategy_index) == 0:
        return pd.Series(dtype=float, index=strategy_index)
    strat_idx = pd.DatetimeIndex(strategy_index).normalize()
    b = bench_nav.copy()
    b.index = pd.DatetimeIndex(b.index).normalize()
    aligned = b.reindex(strat_idx).ffill()
    if aligned.isna().all():
        return pd.Series(dtype=float, index=strat_idx)
    aligned = aligned.bfill()
    v0 = float(aligned.iloc[0])
    if v0 == 0 or np.isnan(v0):
        return pd.Series(dtype=float, index=strat_idx)
    return initial_capital * aligned.astype(float) / v0


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


def _prepare_scored(
    scored_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Validate a scored panel and return (df, trading_dates, ffilled price panel)."""
    if scored_df is None or len(scored_df) == 0:
        raise ValueError("scored_df is empty")

    required = {"date", "ticker", "adj_close"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing columns: {missing}")
    if "prob" not in scored_df.columns and "probability" not in scored_df.columns:
        raise ValueError("scored_df needs a 'prob' or 'probability' column")

    df = scored_df.copy()
    if "prob" not in df.columns:
        df = df.rename(columns={"probability": "prob"})
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    trading_dates = np.sort(df["date"].unique())
    if len(trading_dates) < 2:
        raise ValueError("Need at least two distinct dates in scored_df")

    price_panel = df.pivot_table(
        index="date", columns="ticker", values="adj_close", aggfunc="first",
    ).sort_index().ffill()
    return df, trading_dates, price_panel


def _vix_percentile_by_date(df: pd.DataFrame, config: BacktestConfig) -> pd.Series | None:
    """Per-date VIX percentile when the config uses regime filtering/scaling.

    Raises when a regime option is requested but the panel cannot support it.
    Returning ``None`` here used to make ``--vix-filter`` and
    ``vix_scale_exposure`` silent no-ops: a sweep over VIX thresholds printed
    a table of identical rows, inviting the conclusion that regime filtering
    does not help when in fact it never ran.
    """
    wants_vix = config.vix_filter_percentile is not None or config.vix_scale_exposure
    if not wants_vix:
        return None
    if "vix_percentile" not in df.columns:
        requested = []
        if config.vix_filter_percentile is not None:
            requested.append(f"vix_filter_percentile={config.vix_filter_percentile}")
        if config.vix_scale_exposure:
            requested.append("vix_scale_exposure=True")
        raise ValueError(
            f"{' and '.join(requested)} requires a 'vix_percentile' column, but "
            f"the scored panel has none (columns: {sorted(df.columns)}). "
            "Re-export the walk-forward panel from a training run whose macro "
            "features loaded, or drop the VIX options."
        )
    by_date = df.groupby("date", sort=False)["vix_percentile"].first()
    if by_date.isna().all():
        raise ValueError(
            "'vix_percentile' column is present but entirely NaN, so the VIX "
            "regime options would have no effect. Check the macro merge in the "
            "training run that produced this panel."
        )
    return by_date


def _validate_weighting(df: pd.DataFrame, config: BacktestConfig) -> None:
    """Reject probability weighting on a panel of signed (ranker) scores."""
    if config.weighting != "probability":
        return
    worst = float(df["prob"].min())
    if worst < 0:
        raise ValueError(
            f"weighting='probability' requires non-negative scores, but the "
            f"panel's 'prob' column reaches {worst:.6g}. This looks like a "
            "lambdarank model (--rank-objective), whose raw scores are not "
            "probabilities. Use weighting='equal' instead."
        )


def _vix_exposure_scale(
    config: BacktestConfig, vix_by_date: pd.Series | None, sig_date: pd.Timestamp,
) -> float:
    """Capital multiplier for the VIX-scaled exposure mode (1.0 when off/unknown)."""
    if not config.vix_scale_exposure or vix_by_date is None or sig_date not in vix_by_date.index:
        return 1.0
    v = float(vix_by_date.loc[sig_date])
    if v != v:  # NaN-safe: unknown regime -> full size
        return 1.0
    return min(1.0, max(0.0, 2.0 * (1.0 - v)))


def _benchmark_leg(
    config: BacktestConfig,
    daily_nav: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    provider: object | None,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Download, align, and score the buy-and-hold benchmark leg."""
    spy_nav = pd.Series(dtype=float)
    spy_ret = pd.Series(dtype=float)
    spy_metrics: dict[str, float] = {}
    if not config.benchmark_ticker:
        return spy_nav, spy_ret, spy_metrics
    raw_nav, _ = _download_benchmark(
        start_date, end_date, config.benchmark_ticker, config.initial_capital,
        provider=provider,
    )
    if raw_nav.empty:
        print(
            f"Note: no benchmark bars returned for {config.benchmark_ticker!r} "
            f"({start_date.date()} → {end_date.date()}). "
            "Check network, yfinance limits, or date range."
        )
    spy_nav = _align_benchmark_to_nav(raw_nav, daily_nav.index, config.initial_capital)
    spy_ret = spy_nav.pct_change().dropna()
    spy_metrics = (
        _compute_metrics(spy_nav, [], risk_free_rate=config.risk_free_rate)
        if len(spy_nav) > 1 else {}
    )
    if not spy_metrics:
        usable = int(spy_nav.notna().sum()) if len(spy_nav) else 0
        if not raw_nav.empty and usable == 0:
            print(
                "Note: benchmark loaded but no prices aligned to strategy trading days "
                "(index/calendar mismatch or strategy dates outside benchmark range). "
                "Benchmark column will show N/A."
            )
        elif not raw_nav.empty:
            print(
                "Note: benchmark aligned series has insufficient returns for metrics; "
                "benchmark column will show N/A."
            )
    return spy_nav, spy_ret, spy_metrics


def run_backtest(
    scored_df: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    *,
    provider: object | None = None,
) -> BacktestResult:
    """Run the weekly-rebalance cohort backtest on walk-forward scored data."""
    df, trading_dates, price_panel = _prepare_scored(scored_df)
    _validate_weighting(df, config)

    rebalance_dates = _get_rebalance_dates(trading_dates, config.rebalance_day)
    vix_by_date = _vix_percentile_by_date(df, config)

    # VIX filter: skip rebalances entirely above the threshold
    if config.vix_filter_percentile is not None and vix_by_date is not None:
        rebalance_dates = [
            d for d in rebalance_dates
            if d not in vix_by_date.index or float(vix_by_date.loc[d]) <= config.vix_filter_percentile
        ]

    # Build cohorts sequentially, compounding realized P&L back into cash.
    cohorts: list[Cohort] = []
    by_date = df.groupby("date")
    cash = config.initial_capital
    settled = 0  # cohorts are settled in entry order (fixed holding period)

    for sig_date in rebalance_dates:
        # Credit realized proceeds from cohorts that exited before this signal.
        while settled < len(cohorts) and cohorts[settled].exit_date < sig_date:
            c = cohorts[settled]
            cash += c.capital * (1.0 + c.net_return)
            settled += 1
        # Check overlapping cohort limit
        active = sum(
            1 for c in cohorts if c.entry_date <= sig_date <= c.exit_date
        )
        free_slots = config.max_overlapping_cohorts - active
        if free_slots <= 0:
            continue
        if sig_date not in by_date.groups:
            continue
        capital = (cash / free_slots) * _vix_exposure_scale(config, vix_by_date, sig_date)
        if capital <= 0:
            continue
        scored_day = by_date.get_group(sig_date)
        cohort = _build_cohort(
            sig_date, price_panel, scored_day, config, trading_dates, capital,
        )
        if cohort is not None:
            cohorts.append(cohort)
            cash -= capital

    # Daily NAV
    daily_nav = _build_daily_nav(cohorts, trading_dates, price_panel, config)
    daily_returns = daily_nav.pct_change().dropna()

    # Metrics
    metrics = _compute_metrics(daily_nav, cohorts, risk_free_rate=config.risk_free_rate)

    start_date = pd.Timestamp(trading_dates[0])
    end_date = pd.Timestamp(trading_dates[-1])
    spy_nav, spy_ret, spy_metrics = _benchmark_leg(
        config, daily_nav, start_date, end_date, provider,
    )

    return BacktestResult(
        config=config,
        daily_nav=daily_nav,
        daily_returns=daily_returns,
        cohorts=tuple(cohorts),
        spy_daily_nav=spy_nav,
        spy_daily_returns=spy_ret,
        metrics=metrics,
        spy_metrics=spy_metrics,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# Rank-hold backtest: one continuous portfolio, sell on rank decay
# ---------------------------------------------------------------------------


def run_rank_hold_backtest(
    scored_df: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    *,
    provider: object | None = None,
) -> BacktestResult:
    """Rank-based holding: buy the top-N, sell only when a name's rank decays.

    One continuously managed portfolio instead of fixed-period cohorts. Each
    rebalance day the universe is ranked by score; held names whose rank has
    fallen beyond ``config.exit_rank`` (or that left the scored universe) are
    sold at the next session's close, and freed capital is deployed equally
    across the open slots into the best-ranked names not yet held, up to
    ``top_n`` positions. Turnover — and its cost — is driven by signal decay
    instead of the calendar, which is the point: the fixed-holding cohort
    engine liquidates winners every ``holding_days`` sessions only to re-buy
    many of them a week later.

    The VIX filter blocks *buys* on stressed dates (sells always execute);
    ``vix_scale_exposure`` scales new-buy budgets. Each closed round trip is
    recorded as a single-ticker :class:`Cohort` so win-rate/cost reporting
    and comparisons work unchanged; ``metrics["total_costs"]`` additionally
    includes entry costs of positions still open at the end.
    """
    df, trading_dates, price_panel = _prepare_scored(scored_df)
    _validate_weighting(df, config)
    n_days = len(trading_dates)
    nav_index = pd.DatetimeIndex(trading_dates)
    rebalance_dates = _get_rebalance_dates(trading_dates, config.rebalance_day)
    vix_by_date = _vix_percentile_by_date(df, config)
    by_date = df.groupby("date")
    slip = config.slippage_bps / 10_000

    cash = config.initial_capital
    open_pos: dict[str, dict] = {}
    closed: list[Cohort] = []
    cash_flow = np.zeros(n_days)
    # (entry_idx, exit_idx or None while open, ticker, shares)
    intervals: list[list] = []
    total_costs = 0.0

    for sig_date in rebalance_dates:
        if sig_date not in by_date.groups:
            continue
        entry_date = next_trading_day(sig_date, trading_dates)
        if entry_date is None:
            break
        e_idx = int(np.searchsorted(trading_dates, np.datetime64(entry_date)))
        scored_day = by_date.get_group(sig_date).sort_values("prob", ascending=False)
        ranked_tickers = list(scored_day["ticker"])
        ranks = {t: i + 1 for i, t in enumerate(ranked_tickers)}

        # Sells: rank decayed beyond exit_rank, or name left the universe.
        for t in list(open_pos):
            if ranks.get(t, config.exit_rank + 1) <= config.exit_rank:
                continue
            pos = open_pos.pop(t)
            px = float(price_panel.at[entry_date, t]) if t in price_panel.columns else float("nan")
            if px != px or px <= 0:
                px = pos["raw_entry"]  # last resort: exit flat
            sell_px = _apply_slippage(px, config.slippage_bps, -1)
            sell_comm = pos["shares"] * config.commission_per_share + config.commission_per_order
            proceeds = pos["shares"] * sell_px - sell_comm
            cash += proceeds
            cash_flow[e_idx] += proceeds
            exit_cost = pos["shares"] * px * slip + sell_comm
            total_costs += exit_cost
            closed.append(Cohort(
                signal_date=pos["signal_date"],
                entry_date=pos["entry_date"],
                exit_date=entry_date,
                tickers=(t,),
                weights=(1.0,),
                entry_prices=(pos["entry_price"],),
                exit_prices=(sell_px,),
                capital=pos["basis"],
                gross_return=px / pos["raw_entry"] - 1.0,
                cost=pos["entry_cost"] + exit_cost,
                net_return=proceeds / pos["basis"] - 1.0,
            ))
            for iv in intervals:
                if iv[2] == t and iv[1] is None:
                    iv[1] = e_idx
                    break

        # Buys (regime filter blocks new exposure, never the sells above)
        if (
            config.vix_filter_percentile is not None
            and vix_by_date is not None
            and sig_date in vix_by_date.index
            and float(vix_by_date.loc[sig_date]) > config.vix_filter_percentile
        ):
            continue
        slots = config.top_n - len(open_pos)
        if slots <= 0:
            continue
        budget = cash * _vix_exposure_scale(config, vix_by_date, sig_date)
        if budget <= 0:
            continue
        per = budget / slots
        bought = 0
        # The score floor gates *buys* only; exits stay governed by exit_rank
        # so a held name is never stranded by a threshold change.
        buy_candidates = ranked_tickers
        if config.min_prob is not None:
            eligible = set(
                scored_day.loc[scored_day["prob"] >= config.min_prob, "ticker"]
            )
            buy_candidates = [t for t in ranked_tickers if t in eligible]
        for t in buy_candidates:
            if bought >= slots:
                break
            if t in open_pos or t not in price_panel.columns:
                continue
            px = float(price_panel.at[entry_date, t])
            if px != px or px <= 0:
                continue
            buy_px = _apply_slippage(px, config.slippage_bps, +1)
            shares = per / buy_px
            buy_comm = shares * config.commission_per_share + config.commission_per_order
            outlay = per + buy_comm
            if outlay > cash + 1e-9:
                shares = max(0.0, (cash - config.commission_per_order)) / (
                    buy_px + config.commission_per_share
                )
                buy_comm = shares * config.commission_per_share + config.commission_per_order
                outlay = shares * buy_px + buy_comm
                if shares * buy_px < 1e-6:
                    continue
            cash -= outlay
            cash_flow[e_idx] -= outlay
            entry_cost = shares * px * slip + buy_comm
            total_costs += entry_cost
            open_pos[t] = {
                "shares": shares,
                "basis": outlay,
                "entry_price": buy_px,
                "raw_entry": px,
                "entry_cost": entry_cost,
                "signal_date": sig_date,
                "entry_date": entry_date,
            }
            intervals.append([e_idx, None, t, shares])
            bought += 1

    # Daily NAV: cash ledger + open market value per position interval.
    invested = np.zeros(n_days)
    for entry_idx, exit_idx, t, shares in intervals:
        hi = exit_idx if exit_idx is not None else n_days
        if hi <= entry_idx:
            continue
        px_arr = price_panel[t].to_numpy()[entry_idx:hi]
        invested[entry_idx:hi] += shares * np.nan_to_num(px_arr, nan=0.0)
    cash_series = config.initial_capital + np.cumsum(cash_flow)
    daily_nav = pd.Series(cash_series + invested, index=nav_index, dtype=float)
    daily_returns = daily_nav.pct_change().dropna()

    metrics = _compute_metrics(daily_nav, closed, risk_free_rate=config.risk_free_rate)
    metrics["total_costs"] = total_costs  # include open positions' entry costs
    metrics["n_open_positions"] = float(len(open_pos))

    start_date = pd.Timestamp(trading_dates[0])
    end_date = pd.Timestamp(trading_dates[-1])
    spy_nav, spy_ret, spy_metrics = _benchmark_leg(
        config, daily_nav, start_date, end_date, provider,
    )

    return BacktestResult(
        config=config,
        daily_nav=daily_nav,
        daily_returns=daily_returns,
        cohorts=tuple(closed),
        spy_daily_nav=spy_nav,
        spy_daily_returns=spy_ret,
        metrics=metrics,
        spy_metrics=spy_metrics,
        start_date=start_date,
        end_date=end_date,
    )


# Re-export reporting functions so `from stock_predictor.backtest import print_report` still works.
from stock_predictor.backtest_reporting import (  # noqa: E402, F401
    plot_backtest,
    plot_strategy_comparison,
    print_report,
    print_strategy_comparison,
)


def _load_scored(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Run portfolio backtest on walk-forward scored data.")
    p.add_argument("scored_path", type=Path, help="Path to scored parquet or CSV")
    p.add_argument(
        "--mode",
        default="cohort",
        choices=["cohort", "rank-hold"],
        help="cohort: fixed holding_days baskets; rank-hold: continuous portfolio, "
        "sell only when a name's rank decays beyond --exit-rank",
    )
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--holding-days", type=int, default=10)
    p.add_argument(
        "--exit-rank",
        type=int,
        default=40,
        help="rank-hold mode: sell held names ranked worse than this (>= top-n)",
    )
    p.add_argument(
        "--rebalance-day",
        default="Friday",
        choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "last"],
        help="Signal day per ISO week, or 'last' for last trading day in each week",
    )
    p.add_argument("--weighting", default="equal", choices=["equal", "probability"])
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument(
        "--commission-per-share",
        type=float,
        default=0.0,
        help="Per-share commission each time stock is bought or sold (0=off)",
    )
    p.add_argument(
        "--commission-per-order",
        type=float,
        default=0.0,
        help="Flat commission per ticker per buy or sell order (0=off)",
    )
    p.add_argument(
        "--min-prob",
        type=float,
        default=None,
        dest="min_prob",
        help="Score floor: never buy a name scoring below this (default: off). "
        "Compared against the panel's raw 'prob' column, which is uncalibrated "
        "for classifiers and unbounded for --rank-objective models",
    )
    p.add_argument(
        "--rf-rate",
        type=float,
        default=0.0,
        dest="rf_rate",
        help="Annualized risk-free rate for Sharpe/Sortino (e.g. 0.045). "
        "Default 0 keeps historical numbers, but overstates risk-adjusted "
        "performance over periods when cash actually paid",
    )
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--max-cohorts", type=int, default=2)
    p.add_argument("--vix-filter", type=float, default=None, help="Skip rebalance if VIX pct > this")
    p.add_argument(
        "--benchmark-ticker",
        default="SPY",
        help="Buy-and-hold benchmark (yfinance); use with --no-benchmark to skip download",
    )
    p.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Do not download a benchmark (metrics table shows N/A for benchmark column)",
    )
    p.add_argument("--plots-dir", type=Path, default=None)
    p.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        dest="compare_with",
        help="Second scored panel: same BacktestConfig, overlap metrics + equity_compare.png",
    )
    p.add_argument(
        "--compare-label-a",
        default=None,
        dest="compare_label_a",
        help="Legend label for primary scored file (default: filename stem)",
    )
    p.add_argument(
        "--compare-label-b",
        default=None,
        dest="compare_label_b",
        help="Legend label for --compare-with file (default: its filename stem)",
    )
    p.add_argument(
        "--provider",
        default="yfinance",
        choices=["yfinance", "tiingo"],
        help="Data provider for benchmark download (default: yfinance)",
    )
    args = p.parse_args()

    from stock_predictor.data_provider import get_provider

    bt_provider = get_provider(args.provider)
    path = args.scored_path
    scored = _load_scored(path)
    print(f"Loaded {len(scored)} scored rows from {path}")

    config = BacktestConfig(
        top_n=args.top_n,
        holding_days=args.holding_days,
        rebalance_day=args.rebalance_day,
        weighting=args.weighting,
        slippage_bps=args.slippage_bps,
        initial_capital=args.capital,
        max_overlapping_cohorts=args.max_cohorts,
        vix_filter_percentile=args.vix_filter,
        benchmark_ticker=None if args.no_benchmark else args.benchmark_ticker,
        commission_per_share=args.commission_per_share,
        commission_per_order=args.commission_per_order,
        exit_rank=args.exit_rank,
        min_prob=args.min_prob,
        risk_free_rate=args.rf_rate,
    )
    backtest_fn = run_rank_hold_backtest if args.mode == "rank-hold" else run_backtest
    result = backtest_fn(scored, config, provider=bt_provider)
    print_report(result)

    if args.plots_dir is not None:
        plot_backtest(result, args.plots_dir)

    if args.compare_with is not None:
        path_b = args.compare_with
        scored_b = _load_scored(path_b)
        print(f"Loaded {len(scored_b)} scored rows from {path_b} (comparison)")
        result_b = backtest_fn(scored_b, config, provider=bt_provider)
        la = args.compare_label_a or path.stem
        lb = args.compare_label_b or path_b.stem
        print_strategy_comparison(result, result_b, label_a=la, label_b=lb)
        if args.plots_dir is not None:
            plot_strategy_comparison(
                result,
                result_b,
                args.plots_dir / "equity_compare.png",
                label_a=la,
                label_b=lb,
            )


if __name__ == "__main__":
    main()
