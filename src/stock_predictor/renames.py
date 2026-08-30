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

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Rename:
    """One symbol change, with the date it took effect and what happened.

    A bare ``old -> new`` pair records none of the things that make a rename
    checkable. The effective date lets the two symbols be tested for
    *concurrent trading*, which is the only falsification available from
    prices alone, and it gives a human enough to look the event up.
    """

    old: str
    new: str
    effective: str
    note: str


RENAMES: tuple[Rename, ...] = (
    Rename("ABC",  "COR",  "2023-08-30", "AmerisourceBergen renamed Cencora"),
    Rename("ADS",  "BFH",  "2022-03-23", "Alliance Data renamed Bread Financial"),
    Rename("ANTM", "ELV",  "2022-06-28", "Anthem renamed Elevance Health"),
    Rename("BLL",  "BALL", "2024-01-02", "Ball Corp symbol change"),
    Rename("CDAY", "DAY",  "2024-02-01", "Ceridian renamed Dayforce"),
    Rename("COG",  "CTRA", "2021-10-04", "Cabot Oil & Gas renamed Coterra Energy"),
    Rename("CTL",  "LUMN", "2020-09-18", "CenturyLink renamed Lumen Technologies"),
    Rename("FBHS", "FBIN", "2022-12-19", "Fortune Brands Home renamed Innovations"),
    Rename("GPS",  "GAP",  "2024-01-02", "Gap Inc symbol change"),
    Rename("HFC",  "DINO", "2022-03-14", "HollyFrontier renamed HF Sinclair"),
    Rename("PEAK", "DOC",  "2024-03-04", "Healthpeak Properties symbol change"),
    Rename("PKI",  "RVTY", "2023-05-16", "PerkinElmer renamed Revvity"),
    Rename("RE",   "EG",   "2023-07-10", "Everest Re renamed Everest Group"),
    Rename("TMK",  "GL",   "2019-08-08", "Torchmark renamed Globe Life"),
    Rename("WLTW", "WTW",  "2022-01-10", "Willis Towers Watson symbol change"),
)

TICKER_RENAMES: dict[str, str] = {r.old: r.new for r in RENAMES}
"""Lookup view of :data:`RENAMES`."""

EFFECTIVE: dict[str, pd.Timestamp] = {
    r.old: pd.Timestamp(r.effective) for r in RENAMES
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
    original = work[ticker_col].astype(str)
    work[ticker_col] = [canonical(t, mapping) for t in original]
    # specs.md:157 -- "Symbol mappings and corporate-action aliases MUST be
    # recorded, not applied invisibly." Rewriting the ticker and dropping the
    # original did exactly the thing that forbids: downstream saw ELV with no
    # trace that the row came from ANTM, so the substitution could not be
    # audited from the data it produced.
    work["alias"] = [o if o != c else "" for o, c in zip(original, work[ticker_col])]
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
                # Keep every symbol the merged membership traded under.
                merged_alias = {a for a in (current.get("alias", ""),
                                            row.get("alias", "")) if a}
                current["alias"] = "|".join(sorted(merged_alias))
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
    """Evidence for each rename, from prices alone.

    Two things are measured, and it is worth being exact about what each can
    and cannot establish.

    ``coverage`` is how much of the predecessor's membership the successor
    prices. A real rename carries the company's history forward under the new
    symbol and scores 1.0; a merger or a re-listing scores near zero because
    the surviving entity's series begins when it began. This is *necessary*.

    ``concurrent`` is the number of sessions after the effective date on which
    **both** symbols have prices. One issuer cannot trade under two symbols at
    once, so any concurrency falsifies the claim outright. This is the only
    real falsifier available here.

    Neither is *sufficient*. Two unrelated companies can have overlapping price
    histories and never trade concurrently — a successor that simply has long
    history will satisfy both tests. Prices cannot establish issuer identity;
    that needs a permanent identifier (CUSIP/CIK/FIGI) this project does not
    carry. Each entry's ``note`` records the corporate event so a human can
    check it against a source, and that remains the actual warrant.
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
        # Concurrency: after the symbol changed, only one of the two can
        # trade. Both printing prices on the same session means they are not
        # the same issuer, whatever the coverage says.
        concurrent = 0
        eff = EFFECTIVE.get(old)
        if eff is not None and old in prices.columns and new in prices.columns:
            after = idx[idx > eff]
            if len(after):
                a = prices[old].reindex(after)
                b = prices[new].reindex(after)
                both = ((a.notna()) & (a > 0) & (b.notna()) & (b > 0))
                concurrent = int(both.sum())

        out[old] = {
            "successor": new,
            "effective": str(eff.date()) if eff is not None else None,
            "sessions_wanted": int(len(window)),
            "sessions_priced": have,
            "coverage": (have / len(window)) if len(window) else 0.0,
            "concurrent_sessions": concurrent,
        }
    return out
