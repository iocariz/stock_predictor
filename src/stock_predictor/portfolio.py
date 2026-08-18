"""Persistent portfolio state, order generation, and risk management."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Non-negative weights summing to 1 (parity with the backtest engine).

    See :func:`stock_predictor.backtest._compute_weights`: normalizing signed
    lambdarank scores by their sum produces short positions and unbounded
    leverage, so those inputs are rejected rather than traded.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if weighting != "probability":
        return np.ones(n, dtype=float) / n
    if np.any(probs < 0):
        raise ValueError(
            "weighting='probability' requires non-negative scores, got "
            f"min={float(np.min(probs)):.6g}. Raw lambdarank scores are not "
            "probabilities — use weighting='equal' with --rank-objective models."
        )
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(n, dtype=float) / n
    return probs / total


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
    for p in expiring:
        px = prices.get(p.ticker, p.entry_price)
        sell_px = px * (1 - slippage_bps / 10_000)
        sell_comm = p.shares * commission_per_share + commission_per_order
        sell_orders.append(Order(
            action="SELL", ticker=p.ticker, shares=p.shares,
            price=sell_px, cohort_id=p.cohort_id, reason="expiration",
        ))
        cash_from_sells += p.shares * sell_px - sell_comm

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
    may_open = allow_buys and not (repeat_signal or off_schedule) or force
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
        cohort_id = uuid.uuid4().hex[:8]

        eligible = list(picks)
        if not allow_duplicate_holdings:
            eligible = [p for p in eligible if p["ticker"] not in already_held]
        eligible = eligible[:top_n]
        # Extend the calendar past the last downloaded session so entry (next
        # trading day after as_of) and expiry (holding_days sessions later)
        # always exist — the raw price calendar ends "today".
        cal = extend_calendar(trading_dates, holding_days + 5)
        entry = next_trading_day(as_of, cal)
        expiry_iso = (
            exit_date_iso_after_hold(entry, holding_days, cal)
            if entry is not None
            else None
        )
        if eligible and entry is not None and expiry_iso is not None:
            probs = np.array([float(p.get("prob", 1.0)) for p in eligible], dtype=float)
            # Mirrors backtest._compute_weights: signed (lambdarank) scores
            # would otherwise yield negative dollar targets and, when they
            # nearly cancel, absurd position sizes.
            wts = _long_only_weights(probs, weighting)

            entry_iso = entry.strftime("%Y-%m-%d")
            for pick, w in zip(eligible, wts, strict=True):
                ticker = pick["ticker"]
                px = prices.get(ticker, pick.get("adj_close", 0))
                if px <= 0:
                    continue
                buy_px = px * (1 + slippage_bps / 10_000)
                dollar_amount = float(w) * cap_per_cohort
                gross_shares = dollar_amount / buy_px
                if gross_shares < 1:
                    continue
                shares = floor(gross_shares)
                buy_comm = shares * commission_per_share + commission_per_order
                cash_need = shares * buy_px + buy_comm
                # Sizing used to ignore fees, so commissions could overdraw the
                # account: $1,000 cash with $100/order produced -$200.
                budget_left = capital_available - cash_used
                if cash_need > budget_left:
                    affordable = (budget_left - commission_per_order) / (
                        buy_px + commission_per_share
                    )
                    shares = max(0, floor(affordable))
                    if shares < 1:
                        continue
                    buy_comm = shares * commission_per_share + commission_per_order
                    cash_need = shares * buy_px + buy_comm
                    if cash_need > budget_left:
                        continue
                buy_orders.append(Order(
                    action="BUY", ticker=ticker, shares=shares,
                    price=buy_px, cohort_id=cohort_id, reason="new_pick",
                ))
                new_positions.append(Position(
                    ticker=ticker, shares=shares, entry_price=buy_px,
                    entry_date=entry_iso, expiry_date=expiry_iso,
                    cohort_id=cohort_id, last_price=float(px),
                ))
                cash_used += cash_need

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

    rank_of = {p["ticker"]: i + 1 for i, p in enumerate(ranked_picks)}

    # Sells: rank decayed or ticker no longer scored
    sell_orders: list[Order] = []
    kept_positions: list[Position] = []
    cash_from_sells = 0.0
    new_history = list(state.history)
    for p in state.positions:
        if rank_of.get(p.ticker, exit_rank + 1) <= exit_rank:
            kept_positions.append(p)
            continue
        px = prices.get(p.ticker, p.entry_price)
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

    # Buys: fill open slots from the top of the ranking
    buy_orders: list[Order] = []
    new_positions: list[Position] = []
    cash_used = 0.0
    repeat_signal = bool(state.last_signal_date) and as_of == state.last_signal_date
    off_schedule = (
        rebalance_day is not None
        and pd.Timestamp(as_of).day_name() != rebalance_day
    )
    may_open = allow_buys and not (repeat_signal or off_schedule) or force
    slots = top_n - len(kept_positions)
    if slots > 0 and may_open:
        held = {p.ticker for p in kept_positions}
        cal = extend_calendar(trading_dates, 10)
        entry = next_trading_day(as_of, cal)
        if entry is not None:
            entry_iso = entry.strftime("%Y-%m-%d")
            cash_available = state.cash + cash_from_sells
            per = cash_available / slots
            cohort_id = uuid.uuid4().hex[:8]
            bought = 0
            for pick in ranked_picks:
                if bought >= slots:
                    break
                ticker = pick["ticker"]
                if ticker in held:
                    continue
                px = prices.get(ticker, pick.get("adj_close", 0))
                if px <= 0:
                    continue
                buy_px = px * (1 + slippage_bps / 10_000)
                shares = floor(per / buy_px)
                if shares < 1:
                    continue
                buy_comm = shares * commission_per_share + commission_per_order
                buy_orders.append(Order(
                    action="BUY", ticker=ticker, shares=shares,
                    price=buy_px, cohort_id=cohort_id, reason="new_pick",
                ))
                new_positions.append(Position(
                    ticker=ticker, shares=shares, entry_price=buy_px,
                    entry_date=entry_iso, expiry_date=OPEN_ENDED_EXPIRY,
                    cohort_id=cohort_id, last_price=float(px),
                ))
                cash_used += shares * buy_px + buy_comm
                bought += 1

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
