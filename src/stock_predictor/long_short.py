"""Dollar-neutral long-short backtest.

The measures that reached significance in this project are the long-short
decile spread and the depth excess; the long-only engines cannot express
either, so their alpha stayed insignificant. This engine closes that gap: it
simulates the book the signal actually describes, and charges it properly.

Three costs the gross spread ignores and this does not:

* **Slippage and commissions on both sides**, on turnover only — names that
  stay in the book across a rebalance are not re-traded.
* **Short borrow**, accrued daily on short notional. This is the big one. A
  gross decile spread implicitly assumes shorting is free; it is not, and the
  bottom decile of a model that partly ranks volatility skews toward
  hard-to-borrow names, so a single general-collateral rate is optimistic.
  Run :func:`borrow_sensitivity` rather than trusting one number.
* **Financing**, as the risk-free rate earned on the cash balance (short
  proceeds are credited to cash and held as collateral).

Rebalancing is periodic and full: every ``rebalance_every`` sessions the book
is reconstituted to the current ranking. With ``rebalance_every`` set to the
label horizon, holding period and signal horizon agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_predictor.backtest import (
    _align_benchmark_to_nav,
    _apply_slippage,
    _compute_metrics,
    _download_benchmark,
    _prepare_scored,
    daily_risk_free,
)
from stock_predictor.borrow import resolve_borrow_rates

TRADING_DAYS = 252


@dataclass(frozen=True)
class LongShortConfig:
    """Configuration for :func:`run_long_short_backtest`."""

    decile: float = 0.1
    """Fraction of the ranked universe taken on each side (0.1 = deciles)."""
    long_weight: float = 0.5
    short_weight: float = 0.5
    """Notional per side as a fraction of capital. The defaults give 1.0x
    gross and 0.0 net, directly comparable to the long-only engines' exposure.
    Setting both to 1.0 reproduces the raw decile spread at 2.0x gross."""
    rebalance_every: int = 63
    """Sessions between full reconstitutions; match the label horizon."""
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    commission_per_order: float = 0.0
    short_borrow_annual: float = 0.005
    """Flat annualized borrow, used when no per-name rates are available.
    0.005 is roughly general collateral for liquid large caps."""
    per_name_borrow: bool = False
    """Charge borrow per position instead of one flat rate.

    A flat rate is optimistic here: this model partly ranks volatility, so the
    short book is drawn from the names that are expensive to borrow. Rates come
    from a ``borrow_rate`` column on the panel if present, otherwise from the
    stylised proxy in :mod:`stock_predictor.borrow` — which is a sensitivity
    tool, not a measurement."""
    risk_free_rate: float = 0.045
    """Earned on the cash balance and used for Sharpe."""
    initial_capital: float = 100_000.0
    benchmark_ticker: str | None = None
    min_names_per_side: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.decile <= 0.5:
            raise ValueError(f"decile must be in (0, 0.5], got {self.decile}")
        if self.long_weight < 0 or self.short_weight < 0:
            raise ValueError("long_weight and short_weight must be >= 0")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        if self.short_borrow_annual < 0:
            raise ValueError("short_borrow_annual must be >= 0")
        if self.min_names_per_side < 1:
            raise ValueError("min_names_per_side must be >= 1")


@dataclass(frozen=True)
class LongShortResult:
    config: LongShortConfig
    daily_nav: pd.Series
    daily_returns: pd.Series
    metrics: dict[str, float]
    turnover: pd.Series
    costs: dict[str, float]
    n_rebalances: int
    bench_daily_nav: pd.Series
    bench_metrics: dict[str, float]


def _target_book(
    scored_day: pd.DataFrame, config: LongShortConfig, capital: float,
) -> dict[str, float]:
    """Target dollar exposure per ticker: positive long, negative short."""
    ranked = scored_day.sort_values("prob", ascending=False)
    n_side = int(len(ranked) * config.decile)
    if n_side < config.min_names_per_side:
        return {}
    longs = ranked.head(n_side)["ticker"].tolist()
    shorts = ranked.tail(n_side)["ticker"].tolist()
    long_each = config.long_weight * capital / len(longs)
    short_each = config.short_weight * capital / len(shorts)
    book = {t: long_each for t in longs}
    for t in shorts:
        book[t] = book.get(t, 0.0) - short_each
    return book


def run_long_short_backtest(
    scored_df: pd.DataFrame,
    config: LongShortConfig = LongShortConfig(),
    *,
    provider: object | None = None,
) -> LongShortResult:
    """Simulate a dollar-neutral long-short book with borrow and trading costs."""
    df, trading_dates, price_panel = _prepare_scored(scored_df)
    n_days = len(trading_dates)
    nav_index = pd.DatetimeIndex(trading_dates)
    by_date = df.groupby("date")

    rebal_idx = list(range(0, n_days - 1, config.rebalance_every))
    rf_daily = daily_risk_free(config.risk_free_rate)
    borrow_daily = config.short_borrow_annual / TRADING_DAYS

    rate_frame = resolve_borrow_rates(
        df, flat_rate=config.short_borrow_annual, per_name=config.per_name_borrow,
    )
    rate_panel = None
    if rate_frame is not None:
        rate_panel = (
            rate_frame.pivot_table(index="date", columns="ticker",
                                   values="borrow_rate", aggfunc="first")
            .sort_index().ffill()
        )
    borrowed_rate_sum = 0.0
    borrowed_notional = 0.0

    cash = config.initial_capital
    shares: dict[str, float] = {}
    nav = np.zeros(n_days)
    turnover = np.zeros(n_days)
    cost_slip = cost_comm = cost_borrow = 0.0
    financing = 0.0
    n_rebal = 0

    for i in range(n_days):
        day = trading_dates[i]
        prices = price_panel.loc[day]

        # Mark to market before trading so the book is valued on today's close.
        gross_long = sum(s * prices.get(t, np.nan) for t, s in shares.items() if s > 0)
        gross_short = -sum(s * prices.get(t, np.nan) for t, s in shares.items() if s < 0)
        gross_long = 0.0 if gross_long != gross_long else gross_long
        gross_short = 0.0 if gross_short != gross_short else gross_short

        # Financing: cash earns, shorts are borrowed. Applied before rebalancing
        # so a position pays borrow for every day it is actually held.
        if i > 0:
            interest = cash * rf_daily
            if rate_panel is None:
                borrow = gross_short * borrow_daily
            else:
                # Per position: a book concentrated in specials costs more
                # than the same notional spread across general collateral.
                borrow = 0.0
                day_rates = rate_panel.loc[day] if day in rate_panel.index else None
                for t, sh in shares.items():
                    if sh >= 0:
                        continue
                    px = prices.get(t, np.nan)
                    if px != px:
                        continue
                    notional = -sh * px
                    rate = config.short_borrow_annual
                    if day_rates is not None:
                        r = day_rates.get(t, np.nan)
                        if r == r:
                            rate = float(r)
                    borrow += notional * rate / TRADING_DAYS
                    borrowed_rate_sum += notional * rate
                    borrowed_notional += notional
            cash += interest - borrow
            financing += interest
            cost_borrow += borrow

        if i in rebal_idx and day in by_date.groups:
            equity = cash + sum(
                s * prices.get(t, 0.0) for t, s in shares.items()
            )
            target = _target_book(by_date.get_group(day), config, equity)
            if target:
                n_rebal += 1
                traded = set(target) | set(shares)
                for t in sorted(traded):
                    px = prices.get(t, np.nan)
                    if px != px or px <= 0:
                        continue
                    want = target.get(t, 0.0) / px
                    have = shares.get(t, 0.0)
                    delta = want - have
                    if abs(delta * px) < 1e-9:
                        continue
                    # Buying (delta > 0) lifts the offer, selling hits the bid.
                    fill = _apply_slippage(
                        px, config.slippage_bps, 1 if delta > 0 else -1,
                    )
                    comm = (abs(delta) * config.commission_per_share
                            + config.commission_per_order)
                    cash -= delta * fill + comm
                    cost_slip += abs(delta) * px * config.slippage_bps / 10_000
                    cost_comm += comm
                    turnover[i] += abs(delta * px)
                    if abs(want) < 1e-12:
                        shares.pop(t, None)
                    else:
                        shares[t] = want

        nav[i] = cash + sum(s * prices.get(t, 0.0) for t, s in shares.items())

    daily_nav = pd.Series(nav, index=nav_index, dtype=float)
    daily_returns = daily_nav.pct_change().dropna()
    metrics = _compute_metrics(daily_nav, [], risk_free_rate=config.risk_free_rate)
    metrics["n_rebalances"] = float(n_rebal)
    metrics["gross_leverage"] = config.long_weight + config.short_weight
    metrics["effective_borrow_rate"] = (
        borrowed_rate_sum / borrowed_notional if borrowed_notional > 0
        else config.short_borrow_annual
    )

    bench_nav = pd.Series(dtype=float)
    bench_metrics: dict[str, float] = {}
    if config.benchmark_ticker:
        raw, _ = _download_benchmark(
            pd.Timestamp(trading_dates[0]), pd.Timestamp(trading_dates[-1]),
            config.benchmark_ticker, config.initial_capital, provider=provider,
        )
        bench_nav = _align_benchmark_to_nav(
            raw, daily_nav.index, config.initial_capital,
        )
        if len(bench_nav) > 1:
            bench_metrics = _compute_metrics(
                bench_nav, [], risk_free_rate=config.risk_free_rate,
            )

    return LongShortResult(
        config=config,
        daily_nav=daily_nav,
        daily_returns=daily_returns,
        metrics=metrics,
        turnover=pd.Series(turnover, index=nav_index, dtype=float),
        costs={
            "slippage": cost_slip,
            "commission": cost_comm,
            "borrow": cost_borrow,
            "financing_earned": financing,
            "total_trading": cost_slip + cost_comm,
        },
        n_rebalances=n_rebal,
        bench_daily_nav=bench_nav,
        bench_metrics=bench_metrics,
    )


def borrow_sensitivity(
    scored_df: pd.DataFrame,
    config: LongShortConfig = LongShortConfig(),
    rates: tuple[float, ...] = (0.0, 0.005, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Total return and Sharpe across borrow rates.

    Borrow is the least knowable input and the one the gross spread ignores
    entirely, so it deserves a curve rather than a point estimate. The rate at
    which the strategy stops paying is the number worth quoting.
    """
    from dataclasses import replace

    rows = []
    for rate in rates:
        res = run_long_short_backtest(scored_df, replace(config, short_borrow_annual=rate))
        rows.append({
            "short_borrow_annual": rate,
            "total_return": res.metrics.get("total_return", float("nan")),
            "cagr": res.metrics.get("cagr", float("nan")),
            "sharpe": res.metrics.get("sharpe", float("nan")),
            "max_drawdown": res.metrics.get("max_drawdown", float("nan")),
            "borrow_cost": res.costs["borrow"],
        })
    return pd.DataFrame(rows)
