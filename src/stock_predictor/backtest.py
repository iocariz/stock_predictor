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
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.bundle import (
    describe_bundle,
    validate_execution_panel,
)
from stock_predictor.delisting import (
    DelistingPolicy,
    disposal_value,
    load_proceeds,
)
from stock_predictor.execution import (
    CostModel,
    SelectionRules,
    eligible_candidates,
    portfolio_weights,
    rank_exits,
)
from stock_predictor.execution_calendar import next_trading_day, offset_trading_days
from stock_predictor.stats import downside_deviation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# US cash proxy used when a panel carries no realized rate. Chosen as a
# round mid-2020s short-rate level; pass --rf-rate explicitly for other eras.
DEFAULT_RISK_FREE_RATE = 0.045


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
    risk_free_rate: float | None = None
    """Annualized risk-free rate for Sharpe and Sortino, or ``None`` to infer.

    A zero funding assumption flatters every risk-adjusted metric, so funding
    costs are on by default. Resolution order:

    1. An explicit float here always wins (``0.0`` switches funding off).
    2. Otherwise the panel's realized ``irx_yield`` (13-week T-bill, quoted in
       percent) is charged per date — correct across regimes, and what
       walk-forward panels carry.
    3. Otherwise :data:`DEFAULT_RISK_FREE_RATE`, a US cash proxy.

    Total return, CAGR and drawdown are unaffected. The rate actually applied
    is printed in the report header and returned as
    ``metrics["risk_free_rate_used"]``."""
    rank_offset: int = 0
    """Skip this many top-ranked names before selecting, so the portfolio
    trades a *band* (``rank_offset+1 .. rank_offset+top_n``) rather than the
    head of the list.

    Selection narrows; the price panel does not. Pre-filtering a scored panel
    to a band achieves the same picks but strips the price history of every
    excluded name, leaving held positions marked at stale forward-filled
    prices and badly understating beta — use this instead.

    In rank-hold mode the offset gates *entry* only; exits stay governed by
    ``exit_rank``, so a holding that climbs into the skipped head is kept
    rather than churned."""
    min_prob: float | None = None
    """Score floor: never buy a name scoring below this, even if it is in the
    top_n. Baskets shrink (weights renormalize over survivors) and a date with
    no eligible name simply does not trade. ``None`` disables the floor.

    The floor is compared against the panel's raw ``prob`` column. For a
    classifier that is a probability, but ``scale_pos_weight`` leaves it
    uncalibrated, and for ``--rank-objective`` it is an unbounded lambdarank
    score — so a threshold is only comparable across runs of the same model
    family and configuration."""

    delisting_policy: DelistingPolicy = field(default_factory=DelistingPolicy)
    """How to dispose of a holding that can no longer be priced.

    Rejecting unpriceable fills leaves capital locked in positions that can
    never exit. specs.md requires explicit evidence or an explicit conservative
    fallback, and forbids treating a data gap as proof of delisting -- hence
    the grace period. See :mod:`stock_predictor.delisting`."""
    reject_stale_fills: bool = True
    """Refuse to fill a leg with no quote on the session it executes.

    The price panel is forward-filled so open positions can be *marked* between
    quotes. Executing against a carried-forward price is a different thing: it
    fills at a price that did not exist on that session. specs.md requires every
    requested fill to record Filled or Rejected, and forward filling to be
    confined to aged valuation marks. Set False to reproduce the old behaviour."""
    min_cross_section: int | None = None
    """Fewest scored names a date must carry before it may open positions.

    Separating row roles keeps the newest sessions in the panel — correctly,
    since those are what a live model ranks — but a panel can still end
    ragged, and "the top 15" of a two-name date is not a selection. ``None``
    derives the floor as ``rank_offset + top_n``: the tightest non-arbitrary
    bound, since a basket cannot be filled from fewer names than it holds.

    Entries only. Exits stay governed by ``holding_days``/``exit_rank``, so a
    narrowing cross-section can never strand an open position."""

    @property
    def selection_rules(self) -> SelectionRules:
        """The strategy's selection, in the form every engine consumes.

        Backtest, paper and live share these rules; only the loop around them
        differs. See :mod:`stock_predictor.execution`.
        """
        return SelectionRules(
            top_n=self.top_n,
            rank_offset=self.rank_offset,
            min_prob=self.min_prob,
            min_cross_section=self.min_cross_section,
            weighting=self.weighting,
            exit_rank=self.exit_rank,
        )

    @property
    def cost_model(self) -> CostModel:
        """Slippage and commissions, shared with the live path."""
        return CostModel(
            slippage_bps=self.slippage_bps,
            commission_per_share=self.commission_per_share,
            commission_per_order=self.commission_per_order,
        )

    @property
    def effective_min_cross_section(self) -> int:
        """The floor actually applied; see :attr:`min_cross_section`."""
        return self.selection_rules.effective_min_cross_section

    def __post_init__(self) -> None:
        if self.min_cross_section is not None and self.min_cross_section < 1:
            raise ValueError(
                f"min_cross_section must be >= 1, got {self.min_cross_section}"
            )
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
        if self.risk_free_rate is not None and (
            not np.isfinite(self.risk_free_rate) or not -0.01 <= self.risk_free_rate <= 1.0
        ):
            raise ValueError(
                f"risk_free_rate must be None or an annual rate in [-0.01, 1.0], "
                f"got {self.risk_free_rate!r}"
            )
        if self.rank_offset < 0:
            raise ValueError(f"rank_offset must be >= 0, got {self.rank_offset}")
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
    """Deprecated alias for :func:`stock_predictor.execution.portfolio_weights`."""
    return portfolio_weights(probs, weighting)


def _apply_slippage(price: float, slippage_bps: float, direction: int) -> float:
    """direction: +1 for buy, -1 for sell. See :class:`execution.CostModel`."""
    return CostModel(slippage_bps=slippage_bps).fill_price(price, direction)


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


def _fill_metrics(tally: dict[str, int]) -> dict[str, float]:
    """Every requested fill accounted for, as specs.md requires."""
    req = float(tally["requested"])
    return {
        "fills_requested": req,
        "fills_filled": float(tally["filled"]),
        "fills_rejected": float(tally["rejected"]),
        "fill_reject_rate": float(tally["rejected"] / req) if req else 0.0,
    }


def _resolve_leg_exit(
    ticker: str,
    exit_date: pd.Timestamp,
    *,
    price_panel: pd.DataFrame,
    actual: pd.DataFrame | None,
    trading_dates: np.ndarray,
    config: BacktestConfig,
    evidence: dict,
    last_priced: pd.DataFrame | None,
) -> tuple[float, pd.Timestamp, str] | None:
    """What this leg actually sold for, and when.

    Asked *after* the position exists, so nothing here can influence whether it
    was opened. The nominal exit is only a first attempt:

    * a real quote on the exit session fills there;
    * no quote defers the sale to the next session that prints one -- you still
      own it, and the cash is not real until the fill is;
    * a name that stays unpriceable is disposed of by explicit evidence, or by
      the configured fallback once the grace period lapses (``specs.md:181``,
      ``:249``, ``:587``);
    * ``None`` means it never settled, which the caller shows as capital still
      tied up rather than quietly returning it.
    """
    def _real(when: pd.Timestamp) -> bool:
        if actual is None:
            return True
        return (ticker in actual.columns and when in actual.index
                and bool(actual.at[when, ticker]))

    def _quote(when: pd.Timestamp) -> float:
        try:
            return float(price_panel.at[when, ticker])
        except KeyError:
            return float("nan")

    i0 = int(np.searchsorted(trading_dates, np.datetime64(exit_date)))
    for i in range(i0, len(trading_dates)):
        when = pd.Timestamp(trading_dates[i])
        px = _quote(when)
        if px == px and px > 0 and (not config.reject_stale_fills or _real(when)):
            return (px, when, "quote" if i == i0 else "deferred")
        # Not sellable today. A gap is never itself proof of a delisting, so
        # the policy decides -- and only once the grace period has elapsed.
        last = float("nan")
        if last_priced is not None and ticker in last_priced.columns \
                and when in last_priced.index:
            last = last_priced.at[when, ticker]
        gap = i + 1 if last != last else int(i - last)
        disposal = disposal_value(
            ticker, when, evidence=evidence, sessions_unpriced=gap,
            policy=config.delisting_policy,
        )
        if disposal is not None:
            return (disposal[0], when, disposal[1])
    return None


def _build_cohort(
    signal_date: pd.Timestamp,
    price_panel: pd.DataFrame,
    scored_day: pd.DataFrame,
    config: BacktestConfig,
    trading_dates: np.ndarray,
    capital: float,
    actual: pd.DataFrame | None = None,
    tally: dict[str, int] | None = None,
    evidence: dict | None = None,
    last_priced: pd.DataFrame | None = None,
    deferrals: dict[str, int] | None = None,
) -> list[Cohort]:
    """Open a cohort using only what is knowable on the entry date.

    This used to price the entry *and* the exit here and drop any name whose
    exit quote was missing -- a decision on the signal date that depended on a
    session ``holding_days`` in the future. The names it removed are
    disproportionately the ones that stopped being quoted, so the survivors
    were the survivors twice over.

    Entry eligibility now asks one question: is there a real, positive price to
    buy at today? Exits are resolved separately by :func:`_resolve_leg_exit`.

    Returns a list because a leg whose exit defers settles on a different date
    from the rest and becomes its own single-ticker cohort, exactly as
    rank-hold already models a disposal.
    """
    entry_date = next_trading_day(signal_date, trading_dates)
    if entry_date is None:
        return []
    exit_date = offset_trading_days(entry_date, config.holding_days, trading_dates)
    if exit_date is None:
        return []

    # Selection is shared with the live path; only the price lookup below is
    # simulation-specific.
    picks = eligible_candidates(scored_day, config.selection_rules)[: config.top_n]
    if not picks:
        return []

    evidence = evidence or {}
    tickers: list[str] = []
    entry_prices: list[float] = []
    raw_entries: list[float] = []
    probs: list[float] = []

    for cand in picks:
        t = cand.ticker
        if tally is not None:
            tally["requested"] += 1        # the entry; the exit is counted later
        try:
            ep = price_panel.at[entry_date, t]
        except KeyError:
            if tally is not None:
                tally["rejected"] += 1
            continue
        if np.isnan(ep) or ep <= 0:
            if tally is not None:
                tally["rejected"] += 1
            continue
        # A carried-forward price is a valuation mark, not a fill.
        if config.reject_stale_fills and actual is not None:
            real = (t in actual.columns and entry_date in actual.index
                    and bool(actual.at[entry_date, t]))
            if not real:
                if tally is not None:
                    tally["rejected"] += 1
                continue
        if tally is not None:
            tally["filled"] += 1
        tickers.append(t)
        raw_entries.append(float(ep))
        entry_prices.append(_apply_slippage(ep, config.slippage_bps, +1))
        probs.append(cand.prob)

    if not tickers:
        return []

    weights = portfolio_weights(np.array(probs), config.weighting)

    # Now, and only now, ask what each leg sold for.
    on_time: list[int] = []
    deferred: list[tuple[int, float, pd.Timestamp, str]] = []
    unsettled: list[int] = []
    for i, t in enumerate(tickers):
        if tally is not None:
            tally["requested"] += 1        # the exit
        resolved = _resolve_leg_exit(
            t, exit_date, price_panel=price_panel, actual=actual,
            trading_dates=trading_dates, config=config, evidence=evidence,
            last_priced=last_priced,
        )
        if resolved is None:
            if tally is not None:
                tally["rejected"] += 1
            unsettled.append(i)
            continue
        px, when, source = resolved
        if source != "quote":
            if deferrals is not None:
                deferrals[source] = deferrals.get(source, 0) + 1
            if tally is not None:
                tally["rejected"] += 1
            deferred.append((i, px, when, source))
        else:
            if tally is not None:
                tally["filled"] += 1
            on_time.append(i)

    out: list[Cohort] = []

    def _leg_cohort(idx: list[int], when: pd.Timestamp,
                    prices: dict[int, float]) -> Cohort | None:
        if not idx:
            return None
        w = np.array([weights[i] for i in idx], dtype=float)
        leg_capital = capital * float(w.sum())
        if leg_capital <= 0:
            return None
        w = w / w.sum()
        eps = [entry_prices[i] for i in idx]
        raws = [raw_entries[i] for i in idx]
        xps = [_apply_slippage(prices[i], config.slippage_bps, -1) for i in idx]
        gross = sum(wi * (prices[i] / raws[j] - 1)
                    for j, (wi, i) in enumerate(zip(w, idx)))
        net = sum(wi * (xp / ep - 1) for wi, ep, xp in zip(w, eps, xps))
        slip_cost = leg_capital * sum(wi * 2 * config.slippage_bps / 10_000 for wi in w)
        comm = _cohort_commission_dollars(
            w, leg_capital, list(eps),
            config.commission_per_share, config.commission_per_order,
        )
        return Cohort(
            signal_date=signal_date,
            entry_date=entry_date,
            exit_date=when,
            tickers=tuple(tickers[i] for i in idx),
            weights=tuple(w.tolist()),
            entry_prices=tuple(eps),
            exit_prices=tuple(xps),
            capital=leg_capital,
            gross_return=gross,
            cost=slip_cost + comm,
            net_return=net - comm / leg_capital,
        )

    main = _leg_cohort(
        on_time, exit_date,
        {i: float(price_panel.at[exit_date, tickers[i]]) for i in on_time},
    )
    if main is not None:
        out.append(main)

    for i, px, when, _source in deferred:
        leg = _leg_cohort([i], when, {i: px})
        if leg is not None:
            out.append(leg)

    # Never settled: hold it past the end of the calendar so the capital shows
    # as still tied up instead of silently coming back.
    if unsettled:
        beyond = pd.Timestamp(trading_dates[-1]) + pd.Timedelta(days=1)
        for i in unsettled:
            leg = _leg_cohort([i], beyond, {i: raw_entries[i]})
            if leg is not None:
                out.append(leg)

    return out


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


def _panel_risk_free(df: pd.DataFrame) -> pd.Series | None:
    """Realized annualized cash rate per date, from a carried ``irx_yield``.

    yfinance and FRED both quote the 13-week bill in percent (4.5 = 4.5%).
    Returns ``None`` when the panel has no usable rate column.
    """
    if "irx_yield" not in df.columns:
        return None
    by_date = df.groupby("date", sort=True)["irx_yield"].first().astype(float)
    by_date = by_date.ffill().bfill()
    if by_date.isna().all():
        return None
    return by_date / 100.0


def _daily_excess(daily_ret: pd.Series, risk_free: float | pd.Series) -> pd.Series:
    """Return series net of the funding charge (scalar or per-date rate)."""
    if isinstance(risk_free, pd.Series):
        rf = risk_free.reindex(daily_ret.index).ffill().bfill().fillna(0.0)
        return daily_ret - ((1.0 + rf) ** (1.0 / 252.0) - 1.0)
    return daily_ret - daily_risk_free(risk_free)


def _compute_metrics(
    nav: pd.Series,
    cohorts: list[Cohort],
    *,
    risk_free_rate: float | pd.Series = 0.0,
) -> dict[str, float]:
    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days == 0:
        return {}
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n_days, 1)) - 1
    # Sharpe and Sortino are excess-return statistics; total return, CAGR and
    # drawdown deliberately stay on raw returns.
    excess = _daily_excess(daily_ret, risk_free_rate)
    mean_r = excess.mean()
    std_r = excess.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > _FLAT_EPS else float("nan")
    # Root-mean-square shortfall below zero excess, over every observation —
    # not the standard deviation of the losses, which demeans them and counts
    # only the losing periods.
    downside = downside_deviation(excess)
    sortino = (mean_r / downside * np.sqrt(252)) if downside > _FLAT_EPS else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = drawdown.min()
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    total_costs = sum(c.cost for c in cohorts)
    wins = sum(1 for c in cohorts if c.net_return > 0)
    n_cohorts = len(cohorts)
    win_rate = wins / n_cohorts if n_cohorts > 0 else float("nan")

    rate_used = (
        float(risk_free_rate.reindex(daily_ret.index).ffill().bfill().mean())
        if isinstance(risk_free_rate, pd.Series)
        else float(risk_free_rate)
    )
    return {
        "risk_free_rate_used": rate_used,
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
    execution_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Validate a scored panel.

    Returns ``(df, trading_dates, price_panel, actual)`` where *price_panel* is
    forward-filled for lookup and *actual* marks which of those prices were
    really observed rather than carried forward.
    """
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

    raw = df.pivot_table(
        index="date", columns="ticker", values="adj_close", aggfunc="first",
    ).sort_index()

    # The scored panel is point-in-time filtered, so a holding that leaves the
    # index simply stops having rows. Forward-filling then carries its last
    # in-index price forward indefinitely and fills execute against it. The PIT
    # filter decides what may be *selected*; it does not decide what a holding
    # is *worth*. An unfiltered execution panel, when supplied, prices the fills.
    if execution_prices is not None and len(execution_prices):
        ex = execution_prices.copy()
        ex.index = pd.DatetimeIndex(ex.index).normalize()
        ex = ex.reindex(raw.index)
        # One pass: assigning column by column across a wide panel fragments
        # the frame and is quadratic in the number of tickers.
        shared = raw.columns.intersection(ex.columns)
        extra = ex.columns.difference(raw.columns)
        if len(shared):
            raw[shared] = raw[shared].where(raw[shared].notna(), ex[shared])
        if len(extra):
            raw = pd.concat([raw, ex[extra]], axis=1)

    # Which prices are real, so a fill against a carried-forward one is counted
    # rather than passing silently.
    actual = raw.notna() & (raw > 0)
    return df, trading_dates, raw.ffill(), actual


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


def _resolve_risk_free(df: pd.DataFrame, config: BacktestConfig) -> float | pd.Series:
    """An explicit rate wins; else the panel's realized rate; else the default."""
    if config.risk_free_rate is not None:
        return config.risk_free_rate
    series = _panel_risk_free(df)
    return DEFAULT_RISK_FREE_RATE if series is None else series


def _benchmark_leg(
    config: BacktestConfig,
    daily_nav: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    provider: object | None,
    risk_free: float | pd.Series = 0.0,
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
        _compute_metrics(spy_nav, [], risk_free_rate=risk_free)
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
    execution_prices: pd.DataFrame | None = None,
    delisting_proceeds: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the weekly-rebalance cohort backtest on walk-forward scored data."""
    df, trading_dates, price_panel, actual = _prepare_scored(
        scored_df, execution_prices,
    )
    tally = {"requested": 0, "filled": 0, "rejected": 0}
    deferrals: dict[str, int] = {}
    proceeds_evidence = load_proceeds(delisting_proceeds)
    # Sessions since each ticker last had a real quote, per session -- the
    # grace period is counted in sessions, not in exit attempts.
    _pos = np.arange(len(actual), dtype=float)
    _last_priced = actual.mul(_pos, axis=0).where(actual).cummax()
    _validate_weighting(df, config)
    risk_free = _resolve_risk_free(df, config)

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
    # A deferred leg settles after the cohort it came from, so exits are no
    # longer in entry order and a single advancing pointer would stall on it.
    settled: list[bool] = []

    for sig_date in rebalance_dates:
        # Credit realized proceeds from cohorts that exited before this signal.
        for i, c in enumerate(cohorts):
            if not settled[i] and c.exit_date < sig_date:
                cash += c.capital * (1.0 + c.net_return)
                settled[i] = True
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
        built = _build_cohort(
            sig_date, price_panel, scored_day, config, trading_dates, capital,
            actual=actual, tally=tally, evidence=proceeds_evidence,
            last_priced=_last_priced, deferrals=deferrals,
        )
        for c in built:
            cohorts.append(c)
            settled.append(False)
            cash -= c.capital

    # Daily NAV
    daily_nav = _build_daily_nav(cohorts, trading_dates, price_panel, config)
    daily_returns = daily_nav.pct_change().dropna()

    # Metrics
    metrics = _compute_metrics(daily_nav, cohorts, risk_free_rate=risk_free)
    metrics.update(_fill_metrics(tally))
    # specs.md:249 -- missing exits must appear in the diagnostics rather than
    # being resolved silently, whichever way they were resolved.
    metrics["exits_deferred"] = float(deferrals.get("deferred", 0))
    for source, n in sorted(deferrals.items()):
        if source != "deferred":
            metrics[f"disposals_{source}"] = float(n)

    start_date = pd.Timestamp(trading_dates[0])
    end_date = pd.Timestamp(trading_dates[-1])
    spy_nav, spy_ret, spy_metrics = _benchmark_leg(
        config, daily_nav, start_date, end_date, provider, risk_free,
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
    execution_prices: pd.DataFrame | None = None,
    delisting_proceeds: pd.DataFrame | None = None,
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
    df, trading_dates, price_panel, actual = _prepare_scored(
        scored_df, execution_prices,
    )
    tally = {"requested": 0, "filled": 0, "rejected": 0}
    deferred_exits = 0
    # Sessions since each ticker last had a real quote, per session. The grace
    # period is expressed in sessions, so counting exit *attempts* (weekly)
    # against it would be off by the rebalance interval.
    _pos = np.arange(len(actual), dtype=float)
    _last_priced = actual.mul(_pos, axis=0).where(actual).cummax()
    disposals: dict[str, int] = {}
    proceeds_cash = 0.0
    proceeds_evidence = load_proceeds(delisting_proceeds)
    _validate_weighting(df, config)
    risk_free = _resolve_risk_free(df, config)
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

        # Sells: rank decayed beyond exit_rank, or name left the universe.
        # Shared with the live path so a simulated exit and a real one agree.
        for t in sorted(rank_exits(set(open_pos), ranked_tickers, config.exit_rank)):
            pos = open_pos.pop(t)
            tally["requested"] += 1
            real = (t in actual.columns and entry_date in actual.index
                    and bool(actual.at[entry_date, t]))
            if config.reject_stale_fills and not real:
                # No quote on the exit session: keep the position rather than
                # exiting flat at the entry price, which invents a fill.
                tally["rejected"] += 1
                # Deferred, not abandoned -- but a position that stays
                # unpriceable has to be resolved eventually, by evidence or by
                # a named policy. A gap alone is never proof (specs.md:587).
                last = (
                    _last_priced.at[entry_date, t]
                    if t in _last_priced.columns and entry_date in _last_priced.index
                    else float("nan")
                )
                gap = e_idx + 1 if last != last else int(e_idx - last)
                disposal = disposal_value(
                    t, entry_date, evidence=proceeds_evidence,
                    sessions_unpriced=gap,
                    policy=config.delisting_policy,
                )
                if disposal is None:
                    deferred_exits += 1
                    open_pos[t] = pos
                    continue
                px, source = disposal
                disposals[source] = disposals.get(source, 0) + 1
                proceeds_cash += pos["shares"] * px
                cash += pos["shares"] * px
                cash_flow[e_idx] += pos["shares"] * px
                closed.append(Cohort(
                    signal_date=pos["signal_date"], entry_date=pos["entry_date"],
                    exit_date=entry_date, tickers=(t,), weights=(1.0,),
                    entry_prices=(pos["entry_price"],), exit_prices=(px,),
                    capital=pos["basis"],
                    gross_return=px / pos["raw_entry"] - 1.0,
                    cost=0.0,
                    net_return=(pos["shares"] * px) / pos["basis"] - 1.0,
                ))
                for iv in intervals:
                    if iv[2] == t and iv[1] is None:
                        iv[1] = e_idx
                        break
                continue
            tally["filled"] += 1
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
        # Entry selection is shared with the live path: the cross-section
        # floor, the score floor and the rank offset all apply here too.
        candidates = eligible_candidates(scored_day, config.selection_rules)
        if not candidates:
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
        buy_candidates = [c.ticker for c in candidates]
        for t in buy_candidates:
            if bought >= slots:
                break
            if t in open_pos or t not in price_panel.columns:
                continue
            tally["requested"] += 1
            if config.reject_stale_fills and not (
                t in actual.columns and entry_date in actual.index
                and bool(actual.at[entry_date, t])
            ):
                # Buys were never counted at all, so a name bought at a
                # carried-forward price reported nothing.
                tally["rejected"] += 1
                continue
            tally["filled"] += 1
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

    metrics = _compute_metrics(daily_nav, closed, risk_free_rate=risk_free)
    metrics["total_costs"] = total_costs  # include open positions' entry costs
    # The old denominator was 2 * len(closed), which ignored every entry for a
    # position still open.
    metrics.update(_fill_metrics(tally))
    # specs.md: missing exits must appear in the diagnostics. A deferred exit
    # keeps capital tied up in a position that cannot be priced, which is
    # honest but not free -- delisting proceeds are still unmodelled.
    metrics["exits_deferred"] = float(deferred_exits)
    # specs.md:249 -- delistings appear in the diagnostics, by source.
    metrics["disposals_by_evidence"] = float(disposals.get("evidence", 0))
    metrics["disposals_written_off"] = float(disposals.get("write_off", 0))
    metrics["disposal_proceeds"] = float(proceeds_cash)
    metrics["n_open_positions"] = float(len(open_pos))

    start_date = pd.Timestamp(trading_dates[0])
    end_date = pd.Timestamp(trading_dates[-1])
    spy_nav, spy_ret, spy_metrics = _benchmark_leg(
        config, daily_nav, start_date, end_date, provider, risk_free,
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


# Re-export reporting functions so `from stock_predictor.backtest import
# print_report` still works.
#
# Resolved lazily (PEP 562). backtest_reporting imports *this* module for
# BacktestResult and daily_risk_free, so importing it eagerly here made the
# cycle order-dependent: `import stock_predictor.backtest` happened to work,
# `import stock_predictor.backtest_reporting` raised ImportError in a fresh
# interpreter. Import order is not meant to be part of the public API.
_REPORTING_EXPORTS = (
    "plot_backtest",
    "plot_strategy_comparison",
    "print_report",
    "print_strategy_comparison",
)


def __getattr__(name: str):
    if name in _REPORTING_EXPORTS:
        from stock_predictor import backtest_reporting

        return getattr(backtest_reporting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_REPORTING_EXPORTS])


def _load_scored(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["date"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Extracted from main() so the flag contract can be tested without running."""
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
        "--rank-offset",
        type=int,
        default=0,
        dest="rank_offset",
        help="Skip this many top-ranked names before selecting, so the "
        "portfolio trades the band rank_offset+1..rank_offset+top_n instead "
        "of the head of the list (default 0 = trade the top)",
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
        "--allow-price-mismatch",
        action="store_true",
        dest="allow_price_mismatch",
        help="Proceed when the execution panel does not cover the scored panel. "
        "Off by default: a stale or absent panel silently changes results",
    )
    p.add_argument(
        "--delisting-proceeds",
        type=Path,
        default=None,
        dest="delisting_proceeds",
        help="Parquet/CSV of (ticker, date, proceeds) giving explicit disposal "
        "evidence — a cash acquisition price, or zero for a bankruptcy. Used "
        "in preference to any fallback, point-in-time",
    )
    p.add_argument(
        "--delisting-fallback",
        choices=["write_off", "hold"],
        default="write_off",
        dest="delisting_fallback",
        help="What to do with a holding still unpriceable after the grace "
        "period (default: %(default)s). 'hold' leaves the capital visibly "
        "stuck instead",
    )
    p.add_argument(
        "--delisting-grace-sessions",
        type=int,
        default=63,
        dest="delisting_grace_sessions",
        help="Sessions of silence tolerated before the fallback applies "
        "(default: %(default)s). A shorter gap is a halt or an outage, not a "
        "delisting",
    )
    p.add_argument(
        "--allow-stale-fills",
        action="store_true",
        dest="allow_stale_fills",
        help="Fill against a forward-filled price when the session has no "
        "quote (the old behaviour). Off by default: specs.md requires a "
        "missing entry or exit price to create a rejected fill",
    )
    p.add_argument(
        "--execution-prices",
        type=Path,
        default=None,
        dest="execution_prices",
        help="Wide parquet of adj_close (dates x tickers) covering the FULL "
        "download, used to price fills. The scored panel is point-in-time "
        "filtered, so a holding that leaves the index stops having rows and "
        "its last in-index price is carried forward — fills then execute "
        "against a stale quote. See metrics['stale_fill_rate']",
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=None,
        dest="min_cross_section",
        help="Fewest scored names a date must carry before it may open "
        "positions (default: rank_offset + top_n). Gates entries only; exits "
        "are never blocked, so a narrowing cross-section cannot strand a "
        "position. Guards against trading a ragged panel edge as if it were "
        "a ranking",
    )
    p.add_argument(
        "--rf-rate",
        type=float,
        default=None,
        dest="rf_rate",
        help="Annualized risk-free rate for Sharpe/Sortino (e.g. 0.045). "
        "Default: charge the panel's realized irx_yield per date when present, "
        "else 4.5%%. Pass 0 to switch funding costs off",
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
        choices=["yfinance", "tiingo", "hybrid"],
        help="Data provider for benchmark download (default: yfinance)",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

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
        min_cross_section=args.min_cross_section,
        reject_stale_fills=not args.allow_stale_fills,
        delisting_policy=DelistingPolicy(
            fallback=args.delisting_fallback,
            grace_sessions=args.delisting_grace_sessions,
        ),
        rank_offset=args.rank_offset,
        risk_free_rate=args.rf_rate,
    )
    backtest_fn = run_rank_hold_backtest if args.mode == "rank-hold" else run_backtest
    exec_px = None
    if args.execution_prices is not None:
        exec_px = pd.read_parquet(args.execution_prices)
        print(f"Execution prices from {args.execution_prices} "
              f"({exec_px.shape[0]} dates x {exec_px.shape[1]} tickers)")
    # Nothing used to produce an execution panel, so the backtest paired
    # whatever stale parquet was on disk -- or, absent one, fell back to
    # forward-filled prices, which on rank-hold is +17.28% against +22.95%.
    findings = validate_execution_panel(scored, exec_px)
    if findings:
        print(describe_bundle(findings))
        if not args.allow_price_mismatch:
            sys.exit(
                "Refusing to backtest against a mismatched execution panel. "
                "Regenerate it with train-sp500 --execution-prices-path, or "
                "pass --allow-price-mismatch to proceed."
            )
        print("  --allow-price-mismatch: proceeding anyway.")
    kwargs = {"execution_prices": exec_px}
    if args.delisting_proceeds is not None:
        path = args.delisting_proceeds
        ev = (pd.read_parquet(path) if str(path).endswith(".parquet")
              else pd.read_csv(path))
        print(f"Delisting evidence from {path} ({len(ev)} rows)")
        # Both engines use it now. The cohort engine used to drop any name
        # whose exit did not print, so evidence had nothing to attach to.
        kwargs["delisting_proceeds"] = ev
    # Imported here, not at module scope: backtest_reporting imports this
    # module, so a top-level import would reinstate the cycle. Module-level
    # __getattr__ serves external callers but not this module's own globals.
    from stock_predictor.backtest_reporting import (
        plot_backtest,
        plot_strategy_comparison,
        print_report,
        print_strategy_comparison,
    )

    result = backtest_fn(scored, config, provider=bt_provider, **kwargs)
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
