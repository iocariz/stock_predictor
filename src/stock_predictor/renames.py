"""Ticker renames: the same company, filed under a different symbol.

Membership stints name tickers. Prices are served under whatever symbol a
company trades as *today*. When a company renames, the old symbol is retired --
Yahoo and Tiingo return nothing for ``ANTM`` -- and Anthem's entire 2002-2022
history sits under ``ELV``. So the stint for ANTM pointed at an empty column,
its rows dropped out of the panel, and the company was absent from the
cross-section for the twenty years it was actually a member.

That reads as survivorship in every diagnostic: a departed name with no data.
It is not. The company never departed and the data was never missing; the two
were filed under different symbols. Of the 45 names the baseline's survivorship
gate tolerated as "unavailable upstream", 15 are this, and most of their
successors were already sitting in the panel with complete history.

Every entry below is **validated against prices**, because the plausible ones
are not all real. ``CBS -> PARA`` and ``RX -> IQV`` look exactly like renames
and are not -- a merger and a re-IPO -- and their successors carry *zero*
sessions across the predecessor's stint. ``ESV -> VAL`` fails the same way:
Valaris relisted after Chapter 11 and its history does not carry back. Trusting
the plausible-sounding list would have attached Paramount's returns to CBS's
membership and called it a fix.

:func:`rename_coverage` is that check, and ``tests/test_renames.py`` keeps it
honest.
"""

from __future__ import annotations

import pandas as pd

TICKER_RENAMES: dict[str, str] = {
    # old      new       company / when
    "ABC":     "COR",   # AmerisourceBergen -> Cencora, 2023-08
    "ADS":     "BFH",   # Alliance Data -> Bread Financial, 2022-03
    "ANTM":    "ELV",   # Anthem -> Elevance Health, 2022-06
    "BLL":     "BALL",  # Ball Corp, symbol change 2024
    "CDAY":    "DAY",   # Ceridian -> Dayforce, 2024-02
    "COG":     "CTRA",  # Cabot Oil & Gas -> Coterra Energy, 2021-10
    "CTL":     "LUMN",  # CenturyLink -> Lumen Technologies, 2020-09
    "FBHS":    "FBIN",  # Fortune Brands Home -> Fortune Brands Innovations, 2022-12
    "GPS":     "GAP",   # Gap Inc, symbol change 2024
    "HFC":     "DINO",  # HollyFrontier -> HF Sinclair, 2022-03
    "PEAK":    "DOC",   # Healthpeak Properties, symbol change 2024
    "PKI":     "RVTY",  # PerkinElmer -> Revvity, 2023-05
    "RE":      "EG",    # Everest Re -> Everest Group, 2023-07
    "TMK":     "GL",    # Torchmark -> Globe Life, 2019-08
    "WLTW":    "WTW",   # Willis Towers Watson, symbol change 2022-01
}
"""Old symbol -> the symbol the same company trades under now.

Deliberately excluded, each rejected by :func:`rename_coverage` returning 0%
over the predecessor's stint:

* ``CBS -> PARA`` -- CBS merged with Viacom before the Paramount rename, so
  PARA's series does not reach back through CBS's membership;
* ``RX -> IQV`` -- IMS Health went private and re-listed as IQVIA;
* ``ESV -> VAL`` -- Valaris relisted after Chapter 11;
* ``MDP -> IAC`` -- Meredith was acquired and broken up, not renamed.

A rename is only in this map if the successor's prices actually span the
predecessor's membership. Anything else is a different corporate event wearing
a rename's clothes."""

MERGE_GAP_DAYS = 7
"""How close two stints must be to count as one continuous membership.

A rename produces stints that meet at a day boundary. A company that left the
index and rejoined years later produces two genuine stints, and merging those
would invent membership it never had."""


def canonical(ticker: str, mapping: dict[str, str] | None = None) -> str:
    """The symbol *ticker* trades under now, following renames to the end.

    A company can rename twice, so this follows the chain rather than applying
    one hop. A cycle raises rather than looping.
    """
    mapping = TICKER_RENAMES if mapping is None else mapping
    seen: set[str] = set()
    current = str(ticker)
    while current in mapping:
        if current in seen:
            raise ValueError(f"rename cycle through {current!r}")
        seen.add(current)
        current = mapping[current]
    return current


def canonicalize_stints(
    stints: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    *,
    ticker_col: str = "ticker",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> pd.DataFrame:
    """Rewrite stint tickers to their current symbol and rejoin split memberships.

    Anthem's membership did not lapse when it became Elevance, so the two
    stints either side of the rename become one. Two stints separated by years
    stay separate -- that is a company leaving and rejoining, and merging them
    would fabricate membership.
    """
    if stints is None or not len(stints):
        return stints.copy() if stints is not None else stints

    work = stints.copy()
    work[ticker_col] = [canonical(t, mapping) for t in work[ticker_col].astype(str)]
    work[start_col] = pd.to_datetime(work[start_col])
    work[end_col] = pd.to_datetime(work[end_col])

    out: list[dict] = []
    for name, group in work.groupby(ticker_col, sort=False):
        rows = group.sort_values(start_col, kind="stable")
        current: dict | None = None
        for row in rows.to_dict("records"):
            if current is None:
                current = dict(row)
                continue
            open_ended = pd.isna(current[end_col])
            gap = None if open_ended else (row[start_col] - current[end_col]).days
            if open_ended or (gap is not None and gap <= MERGE_GAP_DAYS):
                # One continuous membership split by a symbol change.
                if pd.isna(row[end_col]) or (
                    not pd.isna(current[end_col]) and row[end_col] > current[end_col]
                ):
                    current[end_col] = row[end_col]
            else:
                out.append(current)
                current = dict(row)
        if current is not None:
            out.append(current)

    result = pd.DataFrame(out, columns=list(work.columns))
    return result.sort_values([ticker_col, start_col], kind="stable").reset_index(drop=True)


def rename_coverage(
    stints: pd.DataFrame,
    prices: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    *,
    ticker_col: str = "ticker",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> dict[str, dict]:
    """For each rename, how much of the old stint the successor actually prices.

    This is what separates a rename from a merger. A real rename carries the
    company's whole history forward under the new symbol and scores 1.0; a
    merger or a re-listing scores near zero, because the surviving entity's
    series begins when it began.
    """
    mapping = TICKER_RENAMES if mapping is None else mapping
    if stints is None or not len(stints) or prices is None or not len(prices):
        return {}

    idx = pd.DatetimeIndex(prices.index)
    work = stints.copy()
    work[ticker_col] = work[ticker_col].astype(str)
    work[start_col] = pd.to_datetime(work[start_col])
    work[end_col] = pd.to_datetime(work[end_col])

    out: dict[str, dict] = {}
    for old, new in mapping.items():
        rows = work[work[ticker_col] == old]
        if rows.empty:
            continue
        lo = max(rows[start_col].min(), idx.min())
        raw_hi = rows[end_col].max()
        hi = min(idx.max(), raw_hi if pd.notna(raw_hi) else idx.max())
        window = idx[(idx >= lo) & (idx <= hi)]
        have = 0
        if new in prices.columns and len(window):
            col = prices[new].reindex(window)
            have = int(((col.notna()) & (col > 0)).sum())
        out[old] = {
            "successor": new,
            "sessions_wanted": int(len(window)),
            "sessions_priced": have,
            "coverage": (have / len(window)) if len(window) else 0.0,
        }
    return out
