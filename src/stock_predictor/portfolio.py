"""Persistent portfolio state, order generation, and risk management."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.execution import (
    CostModel,
    SelectionRules,
    Target,
    eligible_candidates,
    portfolio_weights,
    rank_exits,
    size_targets,
)
from stock_predictor.execution import (
    cohort_id as _cohort_id,
)
from stock_predictor.execution_calendar import (
    exit_date_iso_after_hold,
    extend_calendar,
    next_trading_day,
)


@dataclass(frozen=True)
class Position:
    ticker: str
    shares: int
    entry_price: float
    entry_date: str  # ISO YYYY-MM-DD
    expiry_date: str
    cohort_id: str
    sessions_unpriced: int = 0
    """Consecutive sessions with no usable quote. A duration, not a verdict:
    a gap alone never proves a delisting, which is why the policy carries a
    grace period."""
    last_price: float = 0.0
    """Most recent observed price, refreshed whenever a quote is available.

    Without it a holding that stops being quoted — delisted, halted, or simply
    dropped by the data vendor — falls back to its entry price and can never
    register a loss, so the kill-switch cannot see it."""


@dataclass(frozen=True)
class Order:
    action: str  # "BUY" or "SELL"
    ticker: str
    shares: int
    price: float
    cohort_id: str
    reason: str  # "expiration" or "new_pick"


@dataclass(frozen=True)
class PortfolioState:
    initial_capital: float = 100_000.0
    cash: float = 100_000.0
    high_watermark: float = 100_000.0
    positions: tuple[Position, ...] = ()
    created_at: str = ""
    updated_at: str = ""
    history: tuple[dict, ...] = ()
    last_signal_date: str = ""
    """as_of of the last run that opened a cohort, so re-running one signal
    does not open a second."""
    last_accrual_date: str = ""
    """as_of of the last session whose carry was charged. Distinct from
    *last_signal_date*: borrow accrues every session the short leg is open, and
    measuring it from the last rebalance instead charges 1+2+...+n sessions
    over an n-session holding period rather than n."""

    def __post_init__(self) -> None:
        if not self.created_at:
            now = datetime.now(timezone.utc).isoformat()
            object.__setattr__(self, "created_at", now)
            object.__setattr__(self, "updated_at", now)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def init_state(initial_capital: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        initial_capital=initial_capital,
        cash=initial_capital,
        high_watermark=initial_capital,
    )


def load_state(path: Path) -> PortfolioState:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    known = {f.name for f in fields(Position)}
    positions = tuple(
        Position(**{k: v for k, v in p.items() if k in known})
        for p in raw.get("positions", [])
    )
    history = tuple(raw.get("history", []))
    return PortfolioState(
        initial_capital=raw["initial_capital"],
        cash=raw["cash"],
        high_watermark=raw["high_watermark"],
        positions=positions,
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        history=history,
        last_signal_date=raw.get("last_signal_date", ""),
    )


def save_state(state: PortfolioState, path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "initial_capital": state.initial_capital,
        "cash": state.cash,
        "high_watermark": state.high_watermark,
        "positions": [asdict(p) for p in state.positions],
        "created_at": state.created_at,
        "updated_at": now,
        "history": list(state.history),
        "last_signal_date": state.last_signal_date,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Valuation & risk
# ---------------------------------------------------------------------------


def valid_quote(prices: dict[str, float], ticker: str) -> float | None:
    """A tradable price for *ticker*, or ``None``.

    Zero is a placeholder rather than a price and NaN divides badly, so neither
    counts. This is deliberately stricter than :func:`mark_price`: marking a
    position is an estimate, executing against it is not.
    """
    px = prices.get(ticker)
    if px is None:
        return None
    try:
        px = float(px)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(px) or px <= 0:
        return None
    return px


def mark_price(position: Position, prices: dict[str, float]) -> float:
    """Live quote, else the last one seen, else entry price.

    Falling straight back to entry price marks a collapsed or delisted holding
    at cost, which hides the loss from every downstream metric including the
    kill-switch.
    """
    live = prices.get(position.ticker)
    if live is not None and live == live and live > 0:
        return float(live)
    if position.last_price > 0:
        return float(position.last_price)
    return float(position.entry_price)


def stale_positions(
    state: PortfolioState, prices: dict[str, float],
) -> tuple[str, ...]:
    """Tickers with no live quote, marked from history instead."""
    return tuple(
        p.ticker for p in state.positions
        if not (prices.get(p.ticker) or 0) > 0
    )


def portfolio_value(state: PortfolioState, prices: dict[str, float]) -> float:
    """Mark-to-market: cash + sum(shares * marked price)."""
    return state.cash + sum(
        p.shares * mark_price(p, prices) for p in state.positions
    )


def check_kill_switch(
    state: PortfolioState,
    prices: dict[str, float],
    max_drawdown: float,
) -> tuple[bool, float, float]:
    """Return (is_halted, current_nav, drawdown_pct)."""
    nav = portfolio_value(state, prices)
    watermark = max(state.high_watermark, nav)
    dd = (nav / watermark - 1) if watermark > 0 else 0.0
    halted = dd < -max_drawdown
    return halted, nav, dd


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


def find_expiring_positions(
    state: PortfolioState, as_of: str,
) -> tuple[Position, ...]:
    return tuple(p for p in state.positions if p.expiry_date <= as_of)


def active_cohort_ids(state: PortfolioState, as_of: str) -> set[str]:
    return {p.cohort_id for p in state.positions if p.expiry_date > as_of}


def held_tickers(state: PortfolioState, as_of: str) -> set[str]:
    """Tickers in non-expiring positions."""
    return {p.ticker for p in state.positions if p.expiry_date > as_of}


# ---------------------------------------------------------------------------
# Order generation
# ---------------------------------------------------------------------------


def _long_only_weights(probs: np.ndarray, weighting: str) -> np.ndarray:
    """Deprecated alias for :func:`stock_predictor.execution.portfolio_weights`.

    Kept so existing callers keep working; the live path and the backtest now
    share one implementation instead of two copies that drifted.
    """
    return portfolio_weights(probs, weighting)


def generate_orders(
    state: PortfolioState,
    picks: list[dict],
    prices: dict[str, float],
    *,
    top_n: int,
    max_cohorts: int,
    holding_days: int,
    slippage_bps: float,
    as_of: str,
    trading_dates: np.ndarray,
    weighting: str = "equal",
    rank_offset: int = 0,
    min_prob: float | None = None,
    min_cross_section: int | None = None,
    commission_per_share: float = 0.0,
    commission_per_order: float = 0.0,
    allow_buys: bool = True,
    allow_duplicate_holdings: bool = True,
    rebalance_day: str | None = None,
    force: bool = False,
) -> tuple[tuple[Order, ...], PortfolioState]:
    """
    Sell expired positions, then open a new cohort if a slot is free.

    Returns (orders, proposed_new_state). The caller decides whether to persist.

    Execution assumptions (aligned with :mod:`stock_predictor.backtest`):
    - Entry session is the first trading day strictly after *as_of* — the same
      convention as the backtest's next-day entry; expiry is *holding_days*
      trading sessions after that day (same offset as the backtest exit).
      *trading_dates* ends at the current session, so the calendar is extended
      with future business days to place entry and expiry.
    - The new cohort is funded with ``free_cash / free_slots``, the same rule
      the backtest uses, so a portfolio with one of two slots open deploys all
      of its free cash rather than half of it.
    - *weighting* ``equal`` or ``probability`` matches cohort weights; lots use
      integer shares so dollar weights are approximate vs a fractional backtest.
    -     *trading_dates* should be the sorted unique session dates from the price
      index used for scoring (see :func:`stock_predictor.execution_calendar.trading_dates_from_index`).
    *allow_buys* — set False when a risk kill-switch is active so expiries still
      liquidate but no new cohort opens (avoids persisting buys while halted).
    *rebalance_day* — only open a cohort when *as_of* falls on this weekday,
      matching the backtest's one-signal-per-week schedule. ``None`` allows any
      day. Expiries always settle regardless.
    *force* — bypass both the schedule and the repeat-signal guard.
    *allow_duplicate_holdings* — defaults to ``True`` for parity with the
      cohort backtest, which lets a persistently top-ranked name sit in two
      overlapping cohorts at double weight. Set ``False`` to cap each ticker
      at one lot, accepting that live will then under-weight names the
      simulation kept buying.
    """
    if weighting not in ("equal", "probability"):
        raise ValueError(f"weighting must be 'equal' or 'probability', got {weighting!r}")
    if commission_per_share < 0 or commission_per_order < 0:
        raise ValueError("commission_per_share and commission_per_order must be >= 0")

    today = as_of
    expiring = find_expiring_positions(state, today)
    active_ids = active_cohort_ids(state, today)
    already_held = held_tickers(state, today)

    # Sell orders for expirations
    sell_orders: list[Order] = []
    cash_from_sells = 0.0
    deferred: list[Position] = []
    for p in expiring:
        # A missing quote is not a fill. Falling back to entry_price invented
        # proceeds -- 10 shares entered at $100 and last seen at $40 sold for
        # $1,000 -- so the exit is deferred and the position retained instead.
        px = valid_quote(prices, p.ticker)
        if px is None:
            deferred.append(p)
            continue
        sell_px = px * (1 - slippage_bps / 10_000)
        sell_comm = p.shares * commission_per_share + commission_per_order
        sell_orders.append(Order(
            action="SELL", ticker=p.ticker, shares=p.shares,
            price=sell_px, cohort_id=p.cohort_id, reason="expiration",
        ))
        cash_from_sells += p.shares * sell_px - sell_comm

    if deferred:
        print(
            f"  Deferring {len(deferred)} exit(s) with no quote; positions "
            f"retained: {', '.join(sorted(p.ticker for p in deferred))}"
        )
    expiring = [p for p in expiring if p not in deferred]

    # How many cohort slots are available after removing expired ones?
    # Expired cohort IDs are no longer active
    expired_ids = {p.cohort_id for p in expiring}
    remaining_active = active_ids - expired_ids
    available_slots = max_cohorts - len(remaining_active)

    buy_orders: list[Order] = []
    new_positions: list[Position] = []
    cash_used = 0.0

    # A confirmed run used to open a cohort every time it was invoked, so a
    # daily cron built five cohorts a week against a backtest that models one.
    repeat_signal = bool(state.last_signal_date) and as_of == state.last_signal_date
    off_schedule = (
        rebalance_day is not None
        and pd.Timestamp(as_of).day_name() != rebalance_day
    )
    # `and` binds tighter than `or`, so
    #     allow_buys and not (...) or force
    # parsed as `(allow_buys and not ...) or force`, and force alone opened
    # positions -- including through an engaged drawdown kill switch. force is
    # about timing: it overrides the weekly schedule and the repeat-signal
    # guard, never risk.
    may_open = allow_buys and (force or not (repeat_signal or off_schedule))
    if repeat_signal and not force:
        print(f"  Signal for {as_of} already acted on; no new cohort "
              "(pass force=True to override).")
    elif off_schedule and not force:
        print(f"  {as_of} is not the {rebalance_day} rebalance day; no new cohort.")

    if available_slots > 0 and may_open:
        capital_available = state.cash + cash_from_sells
        # Divide by *free* slots, matching the backtest's `cash / free_slots`.
        # Dividing by max_cohorts under-deploys whenever a slot is occupied —
        # at steady state one always is, so the shortfall never gets invested.
        cap_per_cohort = capital_available / available_slots

        # Selection is the shared core, so a score floor or rank band tuned in
        # the backtest reaches the account instead of being silently dropped.
        rules = SelectionRules(
            top_n=top_n, rank_offset=rank_offset, min_prob=min_prob,
            min_cross_section=min_cross_section, weighting=weighting,
            exit_rank=max(top_n, 30),
        )
        candidates = eligible_candidates(pd.DataFrame(picks), rules)
        if not allow_duplicate_holdings:
            candidates = [c for c in candidates if c.ticker not in already_held]
        eligible = candidates[:top_n]
        # Extend the calendar past the last downloaded session so entry (next
        # trading day after as_of) and expiry (holding_days sessions later)
        # always exist — the raw price calendar ends "today".
        # Anchored at as_of, so a stale price panel does not eat the margin
        # and leave expiry unplaceable.
        cal = extend_calendar(trading_dates, holding_days + 5, anchor=as_of)
        entry = next_trading_day(as_of, cal)
        expiry_iso = (
            exit_date_iso_after_hold(entry, holding_days, cal)
            if entry is not None
            else None
        )
        if entry is None or expiry_iso is None:
            # Say so. This used to fall through to "no available cohort slots",
            # which is a different and much less alarming statement.
            print(
                f"  Cannot place entry/expiry sessions for {as_of} "
                f"(holding_days={holding_days}); no cohort opened."
            )
        if eligible and entry is not None and expiry_iso is not None:
            wts = portfolio_weights(
                np.array([c.prob for c in eligible], dtype=float), weighting,
            )
            targets = [
                Target(ticker=c.ticker,
                       weight=float(w),
                       price=float(prices.get(c.ticker, c.price)))
                for c, w in zip(eligible, wts, strict=True)
            ]
            # whole_shares=True is the *only* thing that differs from the
            # simulation here: an account cannot hold a fractional lot. The
            # budget guard and the fee model are the same code, so commissions
            # can no longer overdraw the account.
            costs = CostModel(
                slippage_bps=slippage_bps,
                commission_per_share=commission_per_share,
                commission_per_order=commission_per_order,
            )
            entry_iso = entry.strftime("%Y-%m-%d")
            lots = size_targets(targets, cap_per_cohort, costs, whole_shares=True)
            # Derived from the decision, so a re-run of an unchanged signal
            # names the same cohort the same way. holding_days is part of the
            # identity: the same names held for 10 days is not the same
            # position as the same names held for 63.
            cohort_id = _cohort_id(
                as_of, [lot.ticker for lot in lots], salt=str(holding_days),
            )
            for lot in lots:
                shares = int(lot.shares)
                buy_orders.append(Order(
                    action="BUY", ticker=lot.ticker, shares=shares,
                    price=lot.fill_price, cohort_id=cohort_id, reason="new_pick",
                ))
                new_positions.append(Position(
                    ticker=lot.ticker, shares=shares, entry_price=lot.fill_price,
                    entry_date=entry_iso, expiry_date=expiry_iso,
                    cohort_id=cohort_id, last_price=float(lot.price),
                ))
                cash_used += lot.cost

    # Build new state, refreshing marks so a later run without a quote falls
    # back to the newest price seen rather than the entry price.
    kept_positions = tuple(
        replace(p, last_price=mark_price(p, prices))
        for p in state.positions if p not in expiring
    )
    new_cash = state.cash + cash_from_sells - cash_used
    nav = new_cash + sum(
        p.shares * mark_price(p, prices)
        for p in (*kept_positions, *new_positions)
    )
    new_watermark = max(state.high_watermark, nav)

    # Record expired cohorts in history
    new_history = list(state.history)
    for cid in expired_ids:
        cohort_positions = [p for p in expiring if p.cohort_id == cid]
        pnl = 0.0
        for p in cohort_positions:
            px = prices.get(p.ticker, p.entry_price)
            sell_px = px * (1 - slippage_bps / 10_000)
            sell_comm = p.shares * commission_per_share + commission_per_order
            pnl += p.shares * sell_px - sell_comm - p.shares * p.entry_price
        new_history.append({
            "cohort_id": cid,
            "closed_date": today,
            "tickers": [p.ticker for p in cohort_positions],
            "pnl": round(pnl, 2),
        })

    new_state = PortfolioState(
        initial_capital=state.initial_capital,
        cash=round(new_cash, 2),
        high_watermark=round(new_watermark, 2),
        positions=(*kept_positions, *new_positions),
        created_at=state.created_at,
        updated_at=datetime.now(timezone.utc).isoformat(),
        history=tuple(new_history),
        last_signal_date=as_of if buy_orders else state.last_signal_date,
    )
    all_orders = (*sell_orders, *buy_orders)
    return all_orders, new_state


# Sentinel expiry for rank-hold positions: they close on rank decay, not time.
OPEN_ENDED_EXPIRY = "9999-12-31"


def generate_orders_rank_hold(
    state: PortfolioState,
    ranked_picks: list[dict],
    prices: dict[str, float],
    *,
    top_n: int,
    exit_rank: int,
    slippage_bps: float,
    as_of: str,
    trading_dates: np.ndarray,
    rank_offset: int = 0,
    min_prob: float | None = None,
    min_cross_section: int | None = None,
    commission_per_share: float = 0.0,
    commission_per_order: float = 0.0,
    allow_buys: bool = True,
    rebalance_day: str | None = None,
    force: bool = False,
) -> tuple[tuple[Order, ...], PortfolioState]:
    """Rank-hold order generation (mirrors :func:`run_rank_hold_backtest`).

    *ranked_picks* must be the FULL universe scored today, best first — the
    exit decision needs every held name's current rank, not just the top of
    the list. Held names ranked worse than *exit_rank* (or missing from the
    ranking) are sold; freed capital is split equally across the open slots
    up to *top_n* positions. Positions carry the OPEN_ENDED_EXPIRY sentinel:
    they are closed by rank decay, never by the fixed-expiry path, so don't
    mix rank-hold and fixed-hold orders on the same state file.
    """
    if exit_rank < top_n:
        raise ValueError(f"exit_rank ({exit_rank}) must be >= top_n ({top_n})")
    if commission_per_share < 0 or commission_per_order < 0:
        raise ValueError("commission_per_share and commission_per_order must be >= 0")

    ranked_tickers = [p["ticker"] for p in ranked_picks]
    # Exit logic is the shared core, so a simulated exit and a real one agree.
    exiting = rank_exits({p.ticker for p in state.positions}, ranked_tickers, exit_rank)

    sell_orders: list[Order] = []
    kept_positions: list[Position] = []
    deferred: list[Position] = []
    cash_from_sells = 0.0
    new_history = list(state.history)
    for p in state.positions:
        if p.ticker not in exiting:
            kept_positions.append(p)
            continue
        px = valid_quote(prices, p.ticker)
        if px is None:
            # No quote, no sale: keep it and try again next run.
            deferred.append(p)
            kept_positions.append(p)
            continue
        sell_px = px * (1 - slippage_bps / 10_000)
        sell_comm = p.shares * commission_per_share + commission_per_order
        sell_orders.append(Order(
            action="SELL", ticker=p.ticker, shares=p.shares,
            price=sell_px, cohort_id=p.cohort_id, reason="rank_exit",
        ))
        cash_from_sells += p.shares * sell_px - sell_comm
        new_history.append({
            "cohort_id": p.cohort_id,
            "closed_date": as_of,
            "tickers": [p.ticker],
            "pnl": round(p.shares * sell_px - sell_comm - p.shares * p.entry_price, 2),
        })

    if deferred:
        print(
            f"  Deferring {len(deferred)} exit(s) with no quote; positions "
            f"retained: {', '.join(sorted(p.ticker for p in deferred))}"
        )

    # Buys: fill open slots from the top of the ranking
    buy_orders: list[Order] = []
    new_positions: list[Position] = []
    cash_used = 0.0
    repeat_signal = bool(state.last_signal_date) and as_of == state.last_signal_date
    off_schedule = (
        rebalance_day is not None
        and pd.Timestamp(as_of).day_name() != rebalance_day
    )
    # `and` binds tighter than `or`, so
    #     allow_buys and not (...) or force
    # parsed as `(allow_buys and not ...) or force`, and force alone opened
    # positions -- including through an engaged drawdown kill switch. force is
    # about timing: it overrides the weekly schedule and the repeat-signal
    # guard, never risk.
    may_open = allow_buys and (force or not (repeat_signal or off_schedule))
    slots = top_n - len(kept_positions)
    if slots > 0 and may_open:
        held = {p.ticker for p in kept_positions}
        cal = extend_calendar(trading_dates, 10)
        entry = next_trading_day(as_of, cal)
        if entry is not None:
            entry_iso = entry.strftime("%Y-%m-%d")
            cash_available = state.cash + cash_from_sells
            # Entry selection shares the backtest's rules: the cross-section
            # floor, the score floor and the rank offset all apply live too.
            rules = SelectionRules(
                top_n=top_n, rank_offset=rank_offset, min_prob=min_prob,
                min_cross_section=min_cross_section, exit_rank=exit_rank,
            )
            costs = CostModel(
                slippage_bps=slippage_bps,
                commission_per_share=commission_per_share,
                commission_per_order=commission_per_order,
            )
            candidates = [
                c for c in eligible_candidates(pd.DataFrame(ranked_picks), rules)
                if c.ticker not in held
            ][:slots]
            # Equal weight per free slot, then the shared whole-share sizer.
            targets = [
                Target(ticker=c.ticker, weight=1.0 / slots,
                       price=float(prices.get(c.ticker, c.price)))
                for c in candidates
            ]
            lots = size_targets(targets, cash_available, costs, whole_shares=True)
            cohort_id = _cohort_id(
                as_of, [lot.ticker for lot in lots], salt=f"rank{exit_rank}",
            )
            for lot in lots:
                shares = int(lot.shares)
                buy_orders.append(Order(
                    action="BUY", ticker=lot.ticker, shares=shares,
                    price=lot.fill_price, cohort_id=cohort_id, reason="new_pick",
                ))
                new_positions.append(Position(
                    ticker=lot.ticker, shares=shares, entry_price=lot.fill_price,
                    entry_date=entry_iso, expiry_date=OPEN_ENDED_EXPIRY,
                    cohort_id=cohort_id, last_price=float(lot.price),
                ))
                cash_used += lot.cost

    kept_positions = [replace(p, last_price=mark_price(p, prices)) for p in kept_positions]
    new_cash = state.cash + cash_from_sells - cash_used
    nav = new_cash + sum(
        p.shares * mark_price(p, prices)
        for p in (*kept_positions, *new_positions)
    )
    new_state = PortfolioState(
        initial_capital=state.initial_capital,
        cash=round(new_cash, 2),
        high_watermark=round(max(state.high_watermark, nav), 2),
        positions=(*kept_positions, *new_positions),
        created_at=state.created_at,
        updated_at=datetime.now(timezone.utc).isoformat(),
        history=tuple(new_history),
        last_signal_date=as_of if buy_orders else state.last_signal_date,
    )
    return (*sell_orders, *buy_orders), new_state


# ---------------------------------------------------------------------------
# Long-short
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
"""Sessions per year, for pro-rating the borrow charge. Matches the
backtest's constant so the two books accrue at the same rate."""

LONG_SHORT_COHORT = "long-short"
"""One book, not a sequence of baskets. Every position carries this id so the
cohort machinery can tell them apart from a fixed-hold basket, and so
:func:`active_cohort_ids` never counts them as competing for a slot."""


def _sessions_between(trading_dates, start: str, end: str) -> int:
    """Trading sessions strictly after *start*, up to and including *end*."""
    if not start:
        return 10**9          # never rebalanced: treat as overdue
    idx = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return int(((idx > lo) & (idx <= hi)).sum())


def generate_orders_long_short(
    state: PortfolioState,
    ranked_picks: list[dict],
    prices: dict[str, float],
    *,
    decile: float,
    long_weight: float,
    short_weight: float,
    rebalance_every: int,
    slippage_bps: float,
    as_of: str,
    trading_dates,
    min_names_per_side: int = 3,
    short_borrow_annual: float = 0.0,
    commission_per_share: float = 0.0,
    commission_per_order: float = 0.0,
    allow_new: bool = True,
    force: bool = False,
    delisting_policy=None,
    delisting_proceeds: pd.DataFrame | None = None,
) -> tuple[tuple[Order, ...], PortfolioState]:
    """Live long-short rebalance, mirroring :func:`run_long_short_backtest`.

    *ranked_picks* must be the FULL universe scored today, best first: the book
    is built from both ends of the ranking, so a truncated list silently
    changes what gets shorted.

    Sizing comes from :func:`~stock_predictor.long_short.target_book` and costs
    from :class:`~stock_predictor.execution.CostModel` -- the same functions the
    backtest uses, not reimplementations of them.

    Two things differ from the long-only paths. Shares go **negative**: a short
    is a liability that ``portfolio_value`` must mark against, not an absent
    position. And there is **no per-position expiry**: the book turns over on a
    calendar, so positions carry :data:`OPEN_ENDED_EXPIRY` and are closed by the
    next rebalance's target rather than by the fixed-expiry sweep. Do not point
    this at a state file a fixed-hold or rank-hold run has written.

    Set *allow_new* to ``False`` to unwind without re-entering, which is what a
    tripped kill switch wants.
    """
    from stock_predictor.delisting import (
        DelistingPolicy,
        disposal_value,
        load_proceeds,
    )
    from stock_predictor.execution import CostModel
    from stock_predictor.long_short import LongShortConfig, target_book

    costs = CostModel(
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        commission_per_order=commission_per_order,
    )
    cash = float(state.cash)

    # Borrow accrues on every session the short leg was open, whether or not
    # today is a rebalance. Charging it only when the book turns over would
    # make a long holding period look free.
    elapsed = _sessions_between(trading_dates, state.last_signal_date, as_of)
    # Carry is charged for sessions not yet charged, not for sessions since the
    # last rebalance. Measuring it from last_signal_date bills 1+2+...+n over an
    # n-session hold: a 63-session cycle paid 2,016 session-days of borrow
    # instead of 63, which cost ~20 points of NAV against the backtest.
    since_accrual = _sessions_between(trading_dates, state.last_accrual_date, as_of)
    if short_borrow_annual > 0 and 0 < since_accrual < 10**9:
        short_notional = sum(
            abs(p.shares) * mark_price(p, prices)
            for p in state.positions if p.shares < 0
        )
        cash -= (short_notional * short_borrow_annual
                 / TRADING_DAYS * since_accrual)

    # Refresh marks even on a quiet session, so a name that stops printing is
    # still valued from its last real quote rather than its entry price.
    marked = tuple(
        replace(p, last_price=mark_price(p, prices),
                sessions_unpriced=(0 if valid_quote(prices, p.ticker) is not None
                                   else p.sessions_unpriced + 1))
        for p in state.positions
    )
    quiet = replace(state, cash=cash, positions=marked,
                    last_accrual_date=as_of,
                    updated_at=datetime.now(timezone.utc).isoformat())

    due = force or elapsed >= rebalance_every
    if not due:
        return (), quiet

    nav = portfolio_value(quiet, prices)
    current = {p.ticker: p for p in marked}
    proceeds_by_ticker = load_proceeds(delisting_proceeds)
    policy = delisting_policy or DelistingPolicy()
    deferred: list[str] = []
    disposed: list[tuple[str, str]] = []

    target: dict[str, float] = {}
    if allow_new:
        frame = pd.DataFrame(ranked_picks)
        if not frame.empty and {"ticker", "prob"} <= set(frame.columns):
            cfg = LongShortConfig(
                decile=decile, long_weight=long_weight,
                short_weight=short_weight, rebalance_every=rebalance_every,
                slippage_bps=slippage_bps, min_names_per_side=min_names_per_side,
            )
            target = target_book(frame, cfg, nav)
        if not target:
            # Too thin a cross-section to build a book. Hold what is there
            # rather than liquidating into a bad tape.
            return (), quiet

    orders: list[Order] = []
    positions: dict[str, Position] = {}
    for ticker in sorted(set(current) | set(target)):
        held = current.get(ticker)
        have = held.shares if held else 0
        px = valid_quote(prices, ticker)
        if px is None:
            # No executable quote, so no fill: pricing one against a mark is
            # what specs.md:405 forbids. But holding it forever is the other
            # failure -- the backtest engine had exactly this bug, and a live
            # book carrying eight delisted names accumulates uncontrolled
            # exposure. Defer, then dispose under the stated policy.
            if held is None:
                continue
            unpriced = held.sessions_unpriced + 1
            settled = disposal_value(
                ticker, as_of, evidence=proceeds_by_ticker,
                sessions_unpriced=unpriced, policy=policy,
            )
            if settled is None:
                positions[ticker] = replace(held, sessions_unpriced=unpriced)
                deferred.append(ticker)
                continue
            per_share, source = settled
            cash += held.shares * per_share
            disposed.append((ticker, source))
            continue
        # Nearest, not truncated. int() rounds toward zero, which biases every
        # leg of a ~50-name book downward by up to a share -- measured at ~9%
        # of intended gross exposure against the backtest's fractional sizing.
        want = int(round(target.get(ticker, 0.0) / px))
        delta = want - have
        if delta == 0:
            if want:
                positions[ticker] = held if held else None
                if positions[ticker] is None:
                    del positions[ticker]
            elif held is not None and have:
                positions[ticker] = held
            continue

        side = 1 if delta > 0 else -1
        fill = costs.fill_price(px, side)
        cash -= delta * fill + costs.commission(delta)
        orders.append(Order(
            action="BUY" if delta > 0 else "SELL",
            ticker=ticker, shares=abs(delta), price=fill,
            cohort_id=LONG_SHORT_COHORT,
            reason=_long_short_reason(have, want),
        ))
        if want:
            positions[ticker] = Position(
                ticker=ticker, shares=want,
                entry_price=fill if have == 0 else (held.entry_price if held else fill),
                entry_date=as_of if have == 0 else (held.entry_date if held else as_of),
                expiry_date=OPEN_ENDED_EXPIRY, cohort_id=LONG_SHORT_COHORT,
                last_price=px,
            )

    new_state = replace(
        quiet,
        cash=cash,
        positions=tuple(positions[t] for t in sorted(positions)),
        last_signal_date=as_of,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return tuple(orders), new_state


def _long_short_reason(have: int, want: int) -> str:
    """Why a leg traded, in the language the report prints."""
    if have == 0:
        return "ls_open_long" if want > 0 else "ls_open_short"
    if want == 0:
        return "ls_close_long" if have > 0 else "ls_cover_short"
    if (have > 0) != (want > 0):
        return "ls_flip"
    return "ls_resize"
