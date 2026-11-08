"""Keep live cost basis on the same scale as the vendor's adjusted history.

The live book stores a fill price and marks against the latest print. Neither
moves when the vendor back-adjusts a held name's history for a dividend, a
split or a spin-off, but the scale they are compared on does: the latest print
is post-event, the entry price is pre-event, and the difference is booked as
P&L -- in NAV, and therefore in the kill switch.

Corteva spun off Vylor on 2026-10-01. The book was short 12 CTVA from $87.82;
CTVA then printed $11.90 and its history was rescaled by ~0.15. The book
reported +$911 on the leg. It also owed 12 VYLR (~$842) it had no record of,
so the gain was the distribution, counted as profit.

Each run compares the adjusted close on the session a leg's reference price
came from (``entry_ref_date``) against that reference. A ratio ``f`` away from
1 is the vendor's adjustment factor since entry. What it means depends on the
event, and only two kinds can be settled:

* **distribution** (dividend, spin-off): paid in cash, as the backtest's
  total-return panel assumes. ``cash += shares * entry_price * (1 - f)`` -- a
  long receives it, a short pays it -- and the leg's prices are scaled by
  ``f``. The cash is the distribution scaled to the entry price rather than
  the pre-event close, which the history no longer contains; on a name that
  has drifted since entry the two differ by that drift.
* **split**: no cash. Shares scale by ``1 / f`` and prices by ``f``.

The two are not distinguishable from the factor alone -- a 2:1 split and a
spin-off worth half the parent both read 0.5 -- and settling one as the other
leaves the wrong exposure on the book. So the code only decides for itself
where the answer is not in doubt: an adjustment under
:data:`CORPORATE_ACTION_THRESHOLD` on a leg with a recorded reference is a
dividend. Anything larger, and any gap on a leg written before the reference
existed, is reported every run and left exactly as it is until an operator
names it in ``confirmed``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from stock_predictor.portfolio import PortfolioState, Position

ADJUSTMENT_TOLERANCE = 1e-3
"""Relative gap below which the history is taken as unadjusted. Vendors
re-serve history with rounding noise that a tighter band would settle, and
re-settle, every day; a quarterly dividend is above it."""

LEGACY_ADOPT_TOLERANCE = 0.005
"""For legs written before ``entry_ref_price`` existed the entry price stands
in, and it carries slippage. Within this band the history's value is adopted as
the reference rather than reported, so no legacy leg is flagged for its own
slippage. A dividend this small on a legacy leg is adopted too, once."""

CORPORATE_ACTION_THRESHOLD = 0.03
"""An adjustment this large is not a routine dividend, and is not settled
without an operator saying what it was."""

DISTRIBUTION = "distribution"
SPLIT = "split"
KINDS = (DISTRIBUTION, SPLIT)


@dataclass(frozen=True)
class Adjustment:
    ticker: str
    session: pd.Timestamp
    factor: float
    kind: str | None
    """``"distribution"`` or ``"split"`` when settled; ``None`` when pending."""
    cash_adjustment: float = 0.0
    shares_before: int = 0
    shares_after: int = 0

    @property
    def settled(self) -> bool:
        return self.kind is not None

    @property
    def is_corporate_action(self) -> bool:
        return abs(self.factor - 1.0) >= CORPORATE_ACTION_THRESHOLD


def _reference_session_close(
    adj_close: pd.DataFrame,
    position: Position,
) -> tuple[pd.Timestamp, float] | None:
    """Adjusted close on the last session at or before the reference date.

    The reference date is the session the fill was priced from. Legs written
    before it was recorded fall back to entry_date, which for the live book is
    the long-short engine's fill session -- or, for legs opened before that
    fix, Labor Day 2026-09-07, which has no session; the prior close is the
    fill. Nothing is forward-filled: a missing print is no evidence of an
    adjustment."""
    if position.ticker not in adj_close.columns:
        return None
    when = position.entry_ref_date or position.entry_date
    index = pd.DatetimeIndex(adj_close.index)
    pos = index.searchsorted(pd.Timestamp(when), side="right") - 1
    if pos < 0:
        return None
    px = adj_close[position.ticker].iloc[pos]
    if px is None or not np.isfinite(px) or px <= 0:
        return None
    return index[pos], float(px)


def _settle_distribution(
    position: Position,
    factor: float,
    close: float,
    session: pd.Timestamp,
) -> tuple[Position, Adjustment]:
    cash = position.shares * position.entry_price * (1.0 - factor)
    settled = replace(
        position,
        entry_price=position.entry_price * factor,
        entry_ref_price=close,
        entry_ref_date=str(session.date()),
        last_price=position.last_price * factor,
    )
    note = Adjustment(
        position.ticker, session, factor, DISTRIBUTION, cash, position.shares, position.shares
    )
    return settled, note


def _settle_split(
    position: Position,
    factor: float,
    close: float,
    session: pd.Timestamp,
) -> tuple[Position, Adjustment]:
    exact = position.shares / factor
    shares = round(exact)
    if abs(exact - shares) > 1e-6 * max(1.0, abs(exact)):
        raise ValueError(
            f"{position.ticker}: {position.shares} shares at a factor of "
            f"{factor:.6f} is {exact:.4f} shares, not a whole number -- not a "
            "split this book can hold. Settle it by hand."
        )
    settled = replace(
        position,
        shares=shares,
        entry_price=position.entry_price * factor,
        entry_ref_price=close,
        entry_ref_date=str(session.date()),
        last_price=position.last_price * factor,
    )
    note = Adjustment(position.ticker, session, factor, SPLIT, 0.0, position.shares, shares)
    return settled, note


def _rebase(
    position: Position,
    adj_close: pd.DataFrame,
    confirmed: Mapping[str, str],
) -> tuple[Position, Adjustment | None]:
    found = _reference_session_close(adj_close, position)
    if found is None:
        return position, None
    session, close = found
    legacy = position.entry_ref_price <= 0
    ref = position.entry_price if legacy else position.entry_ref_price
    if ref <= 0:
        return position, None
    factor = close / ref
    gap = abs(factor - 1.0)
    if legacy and gap <= LEGACY_ADOPT_TOLERANCE:
        return replace(position, entry_ref_price=close, entry_ref_date=str(session.date())), None
    if gap <= ADJUSTMENT_TOLERANCE:
        return position, None

    kind = confirmed.get(position.ticker)
    if kind is None and not legacy and gap < CORPORATE_ACTION_THRESHOLD:
        kind = DISTRIBUTION
    if kind == DISTRIBUTION:
        return _settle_distribution(position, factor, close, session)
    if kind == SPLIT:
        return _settle_split(position, factor, close, session)
    return position, Adjustment(position.ticker, session, factor, None)


def rebase_to_adjusted_history(
    state: PortfolioState,
    adj_close: pd.DataFrame,
    *,
    confirmed: Mapping[str, str] | None = None,
) -> tuple[PortfolioState, tuple[Adjustment, ...]]:
    """Settle any adjustment the vendor applied to a held name since entry.

    *confirmed* maps a ticker to ``"distribution"`` or ``"split"``: the
    operator's word on an event the code will not settle by itself. Returns a
    new state and one :class:`Adjustment` per leg that was settled or is
    pending. Legs with no usable history on their reference session are left
    exactly as they were. Applying it twice is the same as once: a settled
    leg's reference is the history's own value, so the next pass finds 1.
    """
    confirmed = dict(confirmed or {})
    unknown = {t: k for t, k in confirmed.items() if k not in KINDS}
    if unknown:
        raise ValueError(f"confirmed kinds must be one of {KINDS}: {unknown}")
    if adj_close is None or not len(adj_close) or not state.positions:
        return state, ()
    results = [_rebase(p, adj_close, confirmed) for p in state.positions]
    positions = tuple(p for p, _ in results)
    notes = tuple(a for _, a in results if a is not None)
    if positions == state.positions:
        return state, notes
    cash = state.cash + sum(a.cash_adjustment for a in notes)
    return replace(state, positions=positions, cash=cash), notes


def describe_adjustments(notes: tuple[Adjustment, ...]) -> str:
    """One line per leg: what was settled, and what is waiting on an operator."""
    lines = []
    for a in notes:
        where = f"history rescaled x{a.factor:.4f} at {a.session.date()}"
        if a.kind == DISTRIBUTION:
            tag = "CORPORATE ACTION" if a.is_corporate_action else "dividend"
            lines.append(f"  {a.ticker}: {tag}, {where}; settled {a.cash_adjustment:+,.2f} in cash")
        elif a.kind == SPLIT:
            lines.append(
                f"  {a.ticker}: SPLIT, {where}; shares {a.shares_before} -> {a.shares_after}"
            )
        else:
            lines.append(
                f"  {a.ticker}: PENDING CORPORATE ACTION, {where}; left "
                "unsettled until confirmed as a distribution or a split. "
                "NAV is overstated or understated on this leg until then."
            )
    return "\n".join(lines)
