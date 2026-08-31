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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_predictor.backtest import (
    _align_benchmark_to_nav,
    _apply_slippage,
    _compute_metrics,
    _download_benchmark,
    _fill_metrics,
    _prepare_scored,
    daily_risk_free,
)
from stock_predictor.borrow import resolve_borrow_rates
from stock_predictor.delisting import (
    DelistingPolicy,
    disposal_value,
    load_proceeds,
)
from stock_predictor.stats import market_exposure

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
    hedge_beta: float | None = None
    """Short this much benchmark exposure per unit of capital, as an overlay.

    Dollar-neutral is not market-neutral. On the real panel this book carries
    **beta +0.251 (t +4.10)**: the model ranks volatility positively, so the
    long leg holds beta-1.27 names and the short leg beta-0.66 ones, and about
    a quarter of the "neutral" return was market exposure.

    Hedging through the index rather than by scaling the short book leaves the
    stock selection untouched, keeps beta-estimation error out of position
    sizing, and needs one liquid instrument instead of heavier single-name
    shorts. ``None`` or 0 disables it, and the default is disabled: an
    unhedged book is a defensible choice, an unstated one is not."""
    reject_stale_fills: bool = True
    """Refuse to trade a name with no quote on the rebalance session.

    The price panel is forward-filled so open positions can be *marked*
    between quotes; executing against a carried-forward price fills at a price
    that did not exist that session. This engine previously ignored both the
    distinction and the diagnostics."""
    delisting_policy: DelistingPolicy = field(default_factory=DelistingPolicy)
    """What happens to a position that stops being sellable.

    This engine used to have no answer: a rejected exit simply retained the
    position, forever, marked at a carried-forward price. The other two engines
    defer and then dispose by evidence or a stated fallback (``specs.md:249``);
    all three now agree."""
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
        if self.hedge_beta is not None:
            if self.hedge_beta < 0:
                raise ValueError(f"hedge_beta must be >= 0, got {self.hedge_beta}")
            if self.hedge_beta > 0 and not self.benchmark_ticker:
                raise ValueError(
                    "hedge_beta needs a benchmark_ticker to hedge with; "
                    "ignoring the request silently would be worse than refusing it"
                )


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
    execution_prices: pd.DataFrame | None = None,
    delisting_proceeds: pd.DataFrame | None = None,
) -> LongShortResult:
    """Simulate a dollar-neutral long-short book with borrow and trading costs."""
    df, trading_dates, price_panel, actual = _prepare_scored(
        scored_df, execution_prices,
    )
    tally = {"requested": 0, "filled": 0, "rejected": 0}
    deferred_exits = 0
    disposals: dict[str, int] = {}
    proceeds_cash = 0.0
    proceeds_evidence = load_proceeds(delisting_proceeds)
    # Sessions since each name last printed, so the grace period is counted in
    # sessions rather than in rebalances -- which are 63 apart here.
    _pos = np.arange(len(actual), dtype=float)
    _last_priced = actual.mul(_pos, axis=0).where(actual).cummax()

    # The hedge is a synthetic short in the benchmark, priced alongside the
    # stocks so it pays the same slippage, borrow and financing as any other
    # short rather than being a costless return adjustment.
    hedge_beta = float(config.hedge_beta or 0.0)
    hedge_ticker = config.benchmark_ticker or ""
    bench_close = pd.Series(dtype=float)
    if hedge_beta > 0:
        if hedge_ticker in price_panel.columns:
            raise ValueError(
                f"benchmark {hedge_ticker!r} is also a name in the panel; "
                "hedging it would double-count the position"
            )
        bench_close, _ = _download_benchmark(
            pd.Timestamp(trading_dates[0]), pd.Timestamp(trading_dates[-1]),
            hedge_ticker, config.initial_capital, provider=provider,
        )
        if bench_close.empty:
            raise ValueError(
                f"hedge_beta={hedge_beta} requested but the benchmark "
                f"{hedge_ticker!r} could not be downloaded"
            )
        aligned = (bench_close.reindex(pd.DatetimeIndex(trading_dates).normalize())
                   .ffill().bfill())
        price_panel = price_panel.copy()
        price_panel[hedge_ticker] = aligned.to_numpy()
        # The hedge is a real instrument with real quotes; mark it as such, or
        # the stale-fill rejection treats the overlay as unpriceable and it
        # never trades.
        actual = actual.copy()
        actual[hedge_ticker] = aligned.notna().to_numpy() & (aligned > 0).to_numpy()

    n_days = len(trading_dates)
    nav_index = pd.DatetimeIndex(trading_dates)
    by_date = df.groupby("date")

    # Signal sessions; the fill lands on the *next* session, matching the
    # long-only engines. Trading on the signal-day close would use the very
    # bar the score was computed from.
    signal_idx = set(range(0, n_days - 1, config.rebalance_every))
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
    hedge_notional = 0.0

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

        signal_day = trading_dates[i - 1] if i > 0 else None
        if i > 0 and (i - 1) in signal_idx and signal_day in by_date.groups:
            equity = cash + sum(
                s * prices.get(t, 0.0) for t, s in shares.items()
            )
            target = _target_book(by_date.get_group(signal_day), config, equity)
            if target and hedge_beta > 0:
                # Added after selection, so it never consumes a decile slot.
                target[hedge_ticker] = -hedge_beta * equity
            if target:
                n_rebal += 1
                traded = set(target) | set(shares)
                for t in sorted(traded):
                    px = prices.get(t, np.nan)
                    tally["requested"] += 1
                    real = (t in actual.columns and day in actual.index
                            and bool(actual.at[day, t]))
                    unfillable = (px != px or px <= 0
                                  or (config.reject_stale_fills and not real))
                    if unfillable:
                        tally["rejected"] += 1
                        held = shares.get(t, 0.0)
                        # Refusing an *entry* is the end of it: nothing was
                        # opened. Refusing an *exit* leaves a position that has
                        # to be resolved, or it sits in the book forever at a
                        # carried-forward mark.
                        if abs(held) > 1e-12:
                            last = (
                                _last_priced.at[day, t]
                                if t in _last_priced.columns
                                and day in _last_priced.index
                                else float("nan")
                            )
                            gap = i + 1 if last != last else int(i - last)
                            disposal = disposal_value(
                                t, day, evidence=proceeds_evidence,
                                sessions_unpriced=gap,
                                policy=config.delisting_policy,
                            )
                            if disposal is None:
                                deferred_exits += 1
                            else:
                                dpx, source = disposal
                                disposals[source] = disposals.get(source, 0) + 1
                                proceeds_cash += held * dpx
                                cash += held * dpx
                                shares.pop(t, None)
                        continue
                    tally["filled"] += 1
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
                    if t == hedge_ticker and hedge_beta > 0:
                        hedge_notional = abs(want * px)

        # Recorded after any fill, so nav[0] is untouched capital and the
        # opening trade's cost shows up in the first return rather than being
        # normalized away.
        nav[i] = cash + sum(s * prices.get(t, 0.0) for t, s in shares.items())

    daily_nav = pd.Series(nav, index=nav_index, dtype=float)
    daily_returns = daily_nav.pct_change().dropna()
    metrics = _compute_metrics(daily_nav, [], risk_free_rate=config.risk_free_rate)
    metrics["n_rebalances"] = float(n_rebal)
    metrics["gross_leverage"] = config.long_weight + config.short_weight
    metrics["hedge_beta"] = hedge_beta
    metrics.update(_fill_metrics(tally))
    # specs.md:249 -- missing exits appear in the diagnostics, by source.
    metrics["exits_deferred"] = float(deferred_exits)
    metrics["disposals_by_evidence"] = float(disposals.get("evidence", 0))
    metrics["disposals_written_off"] = float(disposals.get("write_off", 0))
    metrics["disposal_proceeds"] = float(proceeds_cash)
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
            # Dollar-neutral is not market-neutral; say what the beta is.
            bench_ret = bench_nav.pct_change().dropna()
            shared = daily_returns.index.intersection(bench_ret.index)
            if len(shared) > 2:
                metrics.update(market_exposure(
                    daily_returns.loc[shared], bench_ret.loc[shared],
                    overlap=config.rebalance_every,
                    risk_free_rate=config.risk_free_rate,
                ))

    return LongShortResult(
        config=config,
        daily_nav=daily_nav,
        daily_returns=daily_returns,
        metrics=metrics,
        turnover=pd.Series(turnover, index=nav_index, dtype=float),
        costs={
            "hedge_notional": hedge_notional,
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
