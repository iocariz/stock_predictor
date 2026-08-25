"""Separate the price you can trade at from the price you can value at.

A live run asks three different questions of the same price panel, and they
have different right answers when a quote is missing:

============  ====================================  ========================
question      what a missing quote means            forward fill?
============  ====================================  ========================
execution     there is no fill; defer the exit      **no** (``specs.md:405``)
valuation     mark it at the last observed price    yes (``specs.md:248``)
reporting     say so, with the date and the age     n/a
============  ====================================  ========================

``predict.py`` built one dictionary from ``adj_close.ffill().iloc[-1]`` and
used it for all three. Forward fill is correct for valuation, so the bug hid
inside a line that looked right: a holding last printed at $41 three sessions
ago arrived at order generation as a clean $41. ``valid_quote()`` — added
precisely to refuse missing quotes — never saw one, and ``stale_positions``
saw a populated dictionary and stayed quiet. The exit sold at a price that did
not exist on the session it claimed to trade.

Splitting the roles here means the guards downstream get the input they were
written against, and the staleness that used to be invisible is now a number
with a date on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEVER_PRICED = int(np.iinfo(np.int64).max)
"""Age for a ticker with no observed price anywhere in the panel. Not zero,
and not "missing" — unbounded, so any freshness comparison rejects it."""


def _usable(value: object) -> float | None:
    """A price you could actually transact at, or ``None``."""
    if value is None:
        return None
    try:
        px = float(value)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(px) or px <= 0:
        return None
    return px


def execution_quotes(panel: pd.DataFrame) -> dict[str, float]:
    """Prices from the panel's **final session only**, with no forward fill.

    A ticker absent from the result has no quote on the session being traded,
    which is the whole point: the caller must then defer rather than invent a
    fill. Non-positive and non-finite prints are dropped for the same reason a
    missing one is — neither is a price you could transact at.
    """
    if panel is None or not len(panel):
        return {}
    final = panel.iloc[-1]
    out: dict[str, float] = {}
    for ticker, raw in final.items():
        px = _usable(raw)
        if px is not None:
            out[str(ticker)] = px
    return out


def valuation_marks(panel: pd.DataFrame) -> dict[str, float]:
    """Forward-filled marks, for NAV and the drawdown kill switch.

    Permitted for valuation by ``specs.md:248`` and necessary for it: a holding
    that stops being quoted must still be markable, or it silently falls back
    to its entry price and can never register a loss the kill switch could see.

    Never pass this to order generation.
    """
    if panel is None or not len(panel):
        return {}
    filled = panel.ffill().iloc[-1]
    out: dict[str, float] = {}
    for ticker, raw in filled.items():
        px = _usable(raw)
        if px is not None:
            out[str(ticker)] = px
    return out


def quote_ages(panel: pd.DataFrame) -> dict[str, int]:
    """Sessions since each ticker's last real print; 0 on the final session.

    This is the number that was missing. Forward fill made a three-session gap
    and a fresh quote indistinguishable at the point of use, so the age is
    carried alongside the mark rather than inferred from it.
    """
    if panel is None or not len(panel):
        return {}
    n = len(panel)
    out: dict[str, int] = {}
    for ticker in panel.columns:
        col = panel[ticker]
        priced = col.notna() & col.apply(lambda v: _usable(v) is not None)
        if not bool(priced.any()):
            out[str(ticker)] = NEVER_PRICED
            continue
        last = int(np.flatnonzero(priced.to_numpy())[-1])
        out[str(ticker)] = n - 1 - last
    return out


def last_quote_dates(panel: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Session of each ticker's last real print. Absent if it never printed."""
    if panel is None or not len(panel):
        return {}
    out: dict[str, pd.Timestamp] = {}
    for ticker in panel.columns:
        col = panel[ticker]
        priced = col.notna() & col.apply(lambda v: _usable(v) is not None)
        idx = np.flatnonzero(priced.to_numpy())
        if len(idx):
            out[str(ticker)] = pd.Timestamp(panel.index[int(idx[-1])])
    return out


def describe_quote_gaps(
    tickers: list[str] | tuple[str, ...],
    ages: dict[str, int],
    marks: dict[str, float],
    dates: dict[str, pd.Timestamp],
    *,
    last_session: pd.Timestamp | str,
    limit: int = 8,
) -> str:
    """Operator-facing summary of holdings with no quote on *last_session*.

    *dates* is positional and required on purpose. "with quote date and age"
    was the whole ask; an optional argument is one a caller forgets, and the
    forgetting is invisible.

    Empty string when nothing is stale, so the caller can print unconditionally.
    """
    if not tickers:
        return ""
    stamp = pd.Timestamp(last_session)
    lines = [
        f"Warning: {len(tickers)} holding(s) have no quote on {stamp.date()} "
        "and cannot be sold. Exits are deferred; the positions are retained.",
    ]
    for t in list(tickers)[:limit]:
        age = ages.get(t, NEVER_PRICED)
        mark = marks.get(t)
        when = dates.get(t)
        age_txt = "never priced" if age >= NEVER_PRICED else f"{age} session(s) stale"
        when_txt = f", last print {pd.Timestamp(when).date()}" if when is not None else ""
        mark_txt = f", marked at {mark:.2f}" if mark is not None else ", unmarkable"
        lines.append(f"  {t}: {age_txt}{when_txt}{mark_txt}")
    if len(tickers) > limit:
        lines.append(f"  … and {len(tickers) - limit} more")
    return "\n".join(lines)
