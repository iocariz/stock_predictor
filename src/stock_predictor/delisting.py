"""What happens to a position that cannot be sold.

Rejecting unpriceable fills is correct — a missing quote is not a fill — but it
leaves capital locked in positions that can never exit. Something has to
dispose of them, and ``specs.md`` is specific about what that may be:

* **:181** — historical delisting treatment must use *explicit evidence* or an
  *explicit conservative fallback*. A ticker's last available row alone must
  not prove delisting.
* **:249** — missing exits, halts and delistings must follow a documented
  configurable policy and appear in the result diagnostics.
* **:587** — missing terminal vendor data is not automatically a delisting.

So a gap is never itself proof. Three things can happen to an unpriceable
holding, in order of preference:

1. **Evidence.** A cash acquisition at $58.50 or a bankruptcy at zero is a
   fact. Supply it and it is used, point-in-time: a deal that settles next
   month cannot pay today.
2. **A named fallback, after a grace period.** A halt or a vendor outage
   resolves in days; a quarter of silence is different. The default fallback is
   **zero**, because a stock you cannot sell is not worth its last quote, and
   marking it there flatters the book.
3. **Holding indefinitely**, for anyone who would rather see the capital
   visibly stuck than written off.

What is never available is inventing a price. Forward-filled quotes may value a
position (``specs.md`` permits that for valuation only); they may not dispose of
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

FALLBACKS = ("write_off", "hold")
DEFAULT_GRACE_SESSIONS = 63
"""About a quarter. Long enough that a halt or a vendor gap resolves first."""


@dataclass(frozen=True)
class DelistingPolicy:
    """How to dispose of a holding that has stopped being priceable."""

    fallback: Literal["write_off", "hold"] = "write_off"
    """``write_off`` realises zero once the grace period lapses — the
    conservative fallback :rfc:`specs.md:181` calls for. ``hold`` keeps the
    position indefinitely and leaves the capital visibly stuck."""
    grace_sessions: int = DEFAULT_GRACE_SESSIONS
    """Sessions of silence tolerated before the fallback applies. A gap shorter
    than this is treated as a halt or an outage, not a delisting."""

    def __post_init__(self) -> None:
        if self.fallback not in FALLBACKS:
            raise ValueError(
                f"fallback must be one of {FALLBACKS}, got {self.fallback!r}"
            )
        if self.grace_sessions < 0:
            raise ValueError(
                f"grace_sessions must be >= 0, got {self.grace_sessions}"
            )


def load_proceeds(frame: pd.DataFrame | None) -> dict[str, tuple[pd.Timestamp, float]]:
    """Index explicit disposal evidence by ticker.

    *frame* needs ``ticker``, ``date`` and ``proceeds`` (per share, in the same
    adjusted currency as the price panel). Where a ticker appears more than
    once the **earliest** row wins: the first settlement is the one that
    happened.
    """
    if frame is None or not len(frame):
        return {}
    missing = {"ticker", "date", "proceeds"} - set(frame.columns)
    if missing:
        raise ValueError(f"proceeds frame missing columns: {sorted(missing)}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["proceeds"] = pd.to_numeric(work["proceeds"], errors="coerce")
    if (work["proceeds"] < 0).any():
        raise ValueError("proceeds must not be negative")
    work = work.dropna(subset=["date", "proceeds"])
    work = work.sort_values("date", kind="stable").drop_duplicates(
        subset="ticker", keep="first",
    )
    return {
        str(r.ticker): (pd.Timestamp(r.date), float(r.proceeds))
        for r in work.itertuples()
    }


def disposal_value(
    ticker: str,
    as_of: pd.Timestamp | str,
    *,
    evidence: dict[str, tuple[pd.Timestamp, float]],
    sessions_unpriced: int,
    policy: DelistingPolicy,
) -> tuple[float, str] | None:
    """Per-share proceeds and their source, or ``None`` to keep holding.

    *sessions_unpriced* is how long the holding has had no usable quote. It is
    a duration, not a verdict: a gap alone never proves a delisting, which is
    why the grace period exists.
    """
    known = evidence.get(str(ticker))
    if known is not None:
        settled, proceeds = known
        # Point in time: evidence dated after this run has not happened yet.
        if settled <= pd.Timestamp(as_of):
            return (proceeds, "evidence")

    if policy.fallback == "hold":
        return None
    if sessions_unpriced > policy.grace_sessions:
        return (0.0, "write_off")
    return None
