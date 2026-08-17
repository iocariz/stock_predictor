"""Point-in-time fundamentals from SEC EDGAR XBRL company facts.

Why EDGAR rather than a convenience vendor: every observation carries the
``filed`` date of the report it came from. That is the difference between a
point-in-time feature and a lookahead bug. Observed filing lags on real
filings run 17-48 days and vary filing to filing, so the usual shortcut —
fiscal period end plus a fixed lag — is wrong by weeks in both directions.

Two hazards this module exists to handle:

**Availability.** A figure for the quarter ending March is not knowable until
it is filed in April or May. :func:`asof_join_fundamentals` joins on ``filed``,
never on period end.

**Restatement.** EDGAR also carries later filings that revise earlier periods.
Indexing on ``filed`` handles this for free: as of any date, you see only what
had actually been filed by then, restatements included once they exist and not
before.

Free, no key, ~3.8 MB per company. SEC asks for a descriptive User-Agent and
caps requests at 10/s.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
def sec_user_agent() -> str:
    """User-Agent for SEC requests.

    SEC's access policy requires a descriptive agent with working contact
    details and answers 403 to anything it considers anonymous. Set
    ``SEC_USER_AGENT`` (e.g. "my-research (me@example.com)"); the fallback
    below is a best effort and may be refused.
    """
    return os.environ.get(
        "SEC_USER_AGENT", "stock-predictor educational research (no-reply@example.com)"
    )


SEC_USER_AGENT = None  # resolved per call via sec_user_agent()
SEC_MIN_INTERVAL_S = 0.11  # SEC caps at 10 requests/second
SEC_FORMS = ("10-K", "10-Q")

# Companies tag the same economic quantity differently, so each logical concept
# lists candidate us-gaap tags in order of preference.
CONCEPT_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "cash_ops": ("NetCashProvidedByUsedInOperatingActivities",
                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "eps_diluted": ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
}

# Flow items are per-period and get summed to trailing twelve months; stock
# items are balances at an instant and are used as-of.
# A fiscal quarter is ~13 weeks and a fiscal year ~52, but companies use
# 4-4-5 and 52/53-week calendars, so both need tolerance bands.
QUARTER_SPAN_DAYS = (80, 100)
ANNUAL_SPAN_DAYS = (350, 380)

FLOW_CONCEPTS = frozenset({
    "revenue", "net_income", "gross_profit", "operating_income",
    "cash_ops", "capex", "eps_diluted",
})

FUNDAMENTAL_FEATURE_COLS: list[str] = [
    "fund_earnings_yield",
    "fund_book_to_price",
    "fund_sales_to_price",
    "fund_fcf_yield",
    "fund_roe",
    "fund_gross_margin",
    "fund_operating_margin",
    "fund_debt_to_equity",
    "fund_revenue_growth_yoy",
    "fund_eps_growth_yoy",
    "fund_accruals",
]


class SecFetchError(RuntimeError):
    """A company's facts could not be retrieved."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _get_json(url: str, *, user_agent: str | None = None, timeout: int = 60):
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent or sec_user_agent()},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cik_map(*, user_agent: str | None = None) -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK."""
    raw = _get_json(SEC_TICKER_MAP_URL, user_agent=user_agent)
    return {
        str(v["ticker"]).upper().replace(".", "-"): str(v["cik_str"]).zfill(10)
        for v in raw.values()
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_concepts(
    facts: dict,
    ticker: str,
    *,
    concept_tags: dict[str, tuple[str, ...]] | None = None,
    forms: tuple[str, ...] = SEC_FORMS,
) -> pd.DataFrame:
    """Flatten one company's XBRL facts into (ticker, concept, end, filed, value).

    For each logical concept the first candidate tag that yields data wins, so
    a company using ``Revenues`` and one using
    ``RevenueFromContractWithCustomer...`` both produce a ``revenue`` row.
    """
    tags = concept_tags or CONCEPT_TAGS
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    rows: list[dict] = []

    for concept, candidates in tags.items():
        for tag in candidates:
            node = gaap.get(tag)
            if not node:
                continue
            picked = False
            for unit_obs in (node.get("units") or {}).values():
                for obs in unit_obs:
                    if obs.get("form") not in forms:
                        continue
                    if obs.get("val") is None or not obs.get("filed"):
                        continue
                    rows.append({
                        "ticker": ticker,
                        "concept": concept,
                        "period_end": obs.get("end"),
                        "period_start": obs.get("start"),
                        "filed": obs["filed"],
                        "value": float(obs["val"]),
                        "form": obs["form"],
                        "fy": obs.get("fy"),
                        "fp": obs.get("fp"),
                    })
                    picked = True
            if picked:
                break  # first tag with data wins

    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "concept", "period_end", "period_start",
            "filed", "value", "form", "fy", "fp",
        ])
    out = pd.DataFrame(rows)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce")
    out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
    return out.dropna(subset=["period_end", "filed"])


def fetch_fundamentals(
    tickers: list[str],
    *,
    cache_dir: Path | None = None,
    user_agent: str | None = None,
    cik_map: dict[str, str] | None = None,
    min_interval_s: float = SEC_MIN_INTERVAL_S,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Long fundamentals table for *tickers*, cached one parquet per ticker.

    Missing or un-mapped tickers are skipped with a note rather than failing
    the run — an S&P panel always contains symbols EDGAR cannot resolve
    (foreign issuers, share-class aliases, renamed entities).
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    cmap = cik_map
    frames: list[pd.DataFrame] = []
    unmapped: list[str] = []
    failed: list[str] = []
    last_call = 0.0

    for i, ticker in enumerate(sorted(set(tickers)), 1):
        cached = cache_dir / f"{ticker}.parquet" if cache_dir else None
        if cached is not None and cached.exists():
            frames.append(pd.read_parquet(cached))
            continue
        if cmap is None:
            # Deferred: a fully cached universe needs no network at all.
            cmap = load_cik_map(user_agent=user_agent)
        cik = cmap.get(ticker.upper())
        if cik is None:
            unmapped.append(ticker)
            continue
        wait = min_interval_s - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            facts = _get_json(SEC_COMPANY_FACTS.format(cik=cik), user_agent=user_agent)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            failed.append(f"{ticker} ({exc})")
            last_call = time.monotonic()
            continue
        last_call = time.monotonic()
        df = extract_concepts(facts, ticker)
        if cached is not None:
            df.to_parquet(cached, index=False)
        frames.append(df)
        if progress_every and i % progress_every == 0:
            print(f"  EDGAR: {i}/{len(set(tickers))} tickers")

    if unmapped:
        print(f"  EDGAR: {len(unmapped)} tickers not in the SEC map "
              f"(e.g. {', '.join(unmapped[:6])})")
    if failed:
        print(f"  EDGAR: {len(failed)} fetch failures (e.g. {failed[0]})")

    usable = [f for f in frames if not f.empty]
    if not usable:
        return pd.DataFrame(columns=[
            "ticker", "concept", "period_end", "period_start",
            "filed", "value", "form", "fy", "fp",
        ])
    return pd.concat(usable, ignore_index=True)


# ---------------------------------------------------------------------------
# Point-in-time shaping
# ---------------------------------------------------------------------------


def _dedupe_filings(fund: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, concept, period_end, filed) — the last value wins.

    A single filing can report the same period more than once across XBRL
    contexts. Keeping the last is arbitrary but stable; what matters is that
    the *filed* date is preserved so the as-of join stays honest.
    """
    return (
        fund.sort_values(["ticker", "concept", "period_end", "filed"])
        .drop_duplicates(["ticker", "concept", "period_end", "filed"], keep="last")
    )


def _derive_q4(quarters: pd.DataFrame, annuals: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the missing fourth quarter as FY minus Q1+Q2+Q3.

    US filers report only three 10-Qs a year; the fourth quarter exists only
    inside the 10-K. Without reconstructing it, four consecutive quarters never
    line up and TTM can only ever update once a year. The derived quarter
    inherits the annual report's filing date, which is exactly when it became
    knowable.
    """
    if quarters.empty or annuals.empty:
        return pd.DataFrame(columns=quarters.columns)
    q_sum = (
        quarters[quarters["fp"].isin(["Q1", "Q2", "Q3"])]
        .groupby(["ticker", "concept", "fy"], sort=False)
        .agg(n=("value", "size"), s=("value", "sum"))
        .reset_index()
    )
    q_sum = q_sum[q_sum["n"] == 3]
    if q_sum.empty:
        return pd.DataFrame(columns=quarters.columns)
    fy = annuals[["ticker", "concept", "fy", "period_end", "filed", "value"]]
    merged = fy.merge(q_sum, on=["ticker", "concept", "fy"], how="inner")
    out = merged.assign(value=merged["value"] - merged["s"], fp="Q4")
    return out[["ticker", "concept", "period_end", "filed", "value", "fp", "fy"]]


def trailing_twelve_months(fund: pd.DataFrame) -> pd.DataFrame:
    """Add a trailing-twelve-month sum for flow concepts, keyed by filing date.

    Each TTM figure inherits the **latest** filing date among its components,
    so it never becomes available before the last piece of it was public.

    Two structural details of EDGAR filings drive the implementation:
    a 10-Q reports both the quarter and the year-to-date cumulative for the
    same period end (the cumulatives are dropped), and the fourth quarter is
    never filed on its own (it is reconstructed by :func:`_derive_q4`).
    """
    if fund.empty:
        return fund.assign(ttm=np.nan)

    work = _dedupe_filings(fund)
    # Earliest filing per period: when the figure actually became public.
    # Sort on both keys — pandas' default sort is not stable, so sorting on
    # period_end alone leaves the filed order arbitrary and keep="first" then
    # picks an arbitrary filing, silently selecting later restatements.
    work = work.sort_values(["ticker", "concept", "period_end", "filed"], kind="stable")
    work = work.drop_duplicates(["ticker", "concept", "period_end"], keep="first")
    work["span_days"] = (work["period_end"] - work["period_start"]).dt.days

    flows = work[work["concept"].isin(FLOW_CONCEPTS)].copy()
    stocks = work[~work["concept"].isin(FLOW_CONCEPTS)].copy()
    stocks["ttm"] = np.nan

    keep = ["ticker", "concept", "period_end", "filed", "value", "ttm"]
    if flows.empty:
        return stocks[keep].reset_index(drop=True)

    quarters = flows[flows["span_days"].between(*QUARTER_SPAN_DAYS)].copy()
    annuals = flows[flows["span_days"].between(*ANNUAL_SPAN_DAYS)].copy()
    # Anything else is a 6-/9-month cumulative for a period end we already
    # cover as a discrete quarter; summing those double-counts.
    all_q = pd.concat(
        [quarters[["ticker", "concept", "period_end", "filed", "value", "fp", "fy"]],
         _derive_q4(quarters, annuals)],
        ignore_index=True,
    )

    # Concatenating derived Q4 rows can widen dtypes to object; pin them back
    # before any datetime or numeric arithmetic.
    all_q["filed"] = pd.to_datetime(all_q["filed"])
    all_q["period_end"] = pd.to_datetime(all_q["period_end"])
    all_q["value"] = all_q["value"].astype(float)

    out: list[pd.DataFrame] = []
    for _, grp in all_q.groupby(["ticker", "concept"], sort=False):
        g = (
            grp.sort_values(["period_end", "filed"], kind="stable")
            .drop_duplicates("period_end", keep="first")
            .reset_index(drop=True)
        )
        g["ttm"] = g["value"].rolling(4).sum()
        # A TTM is knowable only once its newest component has been filed.
        # rolling() cannot aggregate datetimes and int64 nanoseconds exceed
        # float64's exact-integer range, so take the max across explicit
        # shifts, which keeps the dtype and the precision.
        window = pd.concat([g["filed"].shift(k) for k in range(4)], axis=1)
        g["filed"] = window.max(axis=1)
        out.append(g)

    ttm_tbl = pd.concat(out, ignore_index=True)
    # Annual rows stand on their own as a twelve-month figure.
    ann = annuals[["ticker", "concept", "period_end", "filed", "value"]].copy()
    ann["ttm"] = ann["value"]
    parts = [p[keep] for p in (ttm_tbl, ann, stocks) if not p.empty]
    return pd.concat(parts, ignore_index=True)


_MERGE_UNIT = "datetime64[ns]"


def _as_merge_key(series: pd.Series) -> pd.Series:
    """Datetimes pinned to one resolution.

    pandas >= 3.0 keeps whatever unit it inferred, and merge_asof refuses to
    join ``M8[s]`` against ``M8[us]``. Panel dates and EDGAR filing dates
    routinely land on different units, so both keys are pinned here.
    """
    return pd.to_datetime(series).astype(_MERGE_UNIT)


def asof_join_fundamentals(
    panel: pd.DataFrame,
    fund: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Attach, per (ticker, date), the newest figures **filed on or before** that date.

    This is the point-in-time guarantee. Joining on ``period_end`` instead
    would hand the model figures weeks before they were public.
    """
    if fund is None or fund.empty:
        return panel

    wide_cols: dict[str, pd.Series] = {}
    out = panel.copy()
    # Pin the merge key locally only: rewriting the caller's date column would
    # silently change its dtype for every downstream stage.
    left = out[[date_col, ticker_col]].reset_index(drop=True)
    left[date_col] = _as_merge_key(left[date_col])
    # merge_asof requires identical `by` dtypes; parquet round-trips can hand
    # back StringDtype on one side and object on the other.
    left[ticker_col] = left[ticker_col].astype(str)
    left = left.sort_values(date_col, kind="stable")

    for concept, grp in fund.groupby("concept", sort=False):
        g = grp.copy()
        g["filed"] = _as_merge_key(g["filed"])
        # A flow concept is *only* ever its TTM sum, never a bare quarter.
        # Falling back to the single period would make the column mean
        # "one quarter" early in a company's history and "twelve months"
        # later on — a level shift the model would happily learn.
        if concept in FLOW_CONCEPTS:
            g["v"] = g["ttm"]
        else:
            g["v"] = g["value"]
        g[ticker_col] = g[ticker_col].astype(str)
        g = (
            g.dropna(subset=["v"])
            .sort_values(["filed", "period_end"])
            .drop_duplicates([ticker_col, "filed"], keep="last")
            .sort_values("filed", kind="stable")
        )
        if g.empty:
            continue
        merged = pd.merge_asof(
            left, g[[ticker_col, "filed", "v", "period_end"]],
            left_on=date_col, right_on="filed", by=ticker_col,
            direction="backward", allow_exact_matches=True,
        )
        wide_cols[f"raw_{concept}"] = merged["v"].to_numpy()
        if concept == "revenue":
            wide_cols["fund_period_end"] = merged["period_end"].to_numpy()
            wide_cols["fund_filed"] = merged["filed"].to_numpy()

    if not wide_cols:
        return panel
    attached = pd.DataFrame(wide_cols, index=left.index)
    return out.join(attached)


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------


def _safe_div(num: pd.Series, den: pd.Series, *, den_positive: bool = False) -> pd.Series:
    """Ratio with a NaN (never inf) result for unusable denominators."""
    d = den.astype(float)
    d = d.where(d > 0) if den_positive else d.where(d.abs() > 1e-9)
    return (num.astype(float) / d).replace([np.inf, -np.inf], np.nan)


def add_fundamental_features(
    panel: pd.DataFrame, *, price_col: str = "adj_close",
) -> pd.DataFrame:
    """Scale-free fundamental ratios from the joined ``raw_*`` columns.

    Every feature is a ratio, so it is comparable across companies of very
    different size — which is what a cross-sectional ranker needs. Levels
    (revenue in dollars) would just proxy for market cap.

    Growth is computed against the value four quarters earlier *as filed*,
    using the per-ticker history of the joined series rather than a
    period-end shift, so it inherits the same point-in-time guarantee.
    """
    out = panel.copy()
    need = {f"raw_{c}" for c in ("revenue", "net_income", "equity", "assets")}
    if not need & set(out.columns):
        for col in FUNDAMENTAL_FEATURE_COLS:
            out[col] = np.nan
        return out

    def raw(name: str) -> pd.Series:
        col = f"raw_{name}"
        return out[col] if col in out.columns else pd.Series(np.nan, index=out.index)

    price = out[price_col].astype(float)
    shares = raw("shares_diluted")
    equity, assets = raw("equity"), raw("assets")
    revenue, net_income = raw("revenue"), raw("net_income")
    cash_ops, capex = raw("cash_ops"), raw("capex")

    # Per-share quantities divided by price give yields; a yield is the
    # inverse of a multiple and behaves better cross-sectionally (no blow-up
    # as earnings approach zero from above).
    out["fund_earnings_yield"] = _safe_div(
        _safe_div(net_income, shares, den_positive=True), price, den_positive=True,
    )
    # Negative book equity is a different regime, not a cheap valuation, so
    # the ratio is undefined there rather than signed. Earnings yield below is
    # deliberately left signed: a loss-making firm ranking worst is correct.
    out["fund_book_to_price"] = _safe_div(
        _safe_div(equity.where(equity > 0), shares, den_positive=True),
        price, den_positive=True,
    )
    out["fund_sales_to_price"] = _safe_div(
        _safe_div(revenue, shares, den_positive=True), price, den_positive=True,
    )
    fcf = cash_ops.astype(float) - capex.astype(float).fillna(0.0)
    out["fund_fcf_yield"] = _safe_div(
        _safe_div(fcf, shares, den_positive=True), price, den_positive=True,
    )

    out["fund_roe"] = _safe_div(net_income, equity, den_positive=True)
    out["fund_gross_margin"] = _safe_div(raw("gross_profit"), revenue, den_positive=True)
    out["fund_operating_margin"] = _safe_div(
        raw("operating_income"), revenue, den_positive=True,
    )
    out["fund_debt_to_equity"] = _safe_div(raw("liabilities"), equity, den_positive=True)
    # Accruals: earnings not backed by cash. Negative is the healthier sign.
    out["fund_accruals"] = _safe_div(net_income - cash_ops, assets, den_positive=True)

    out = _add_growth(out, "fund_revenue_growth_yoy", "raw_revenue")
    out = _add_growth(out, "fund_eps_growth_yoy", "raw_eps_diluted")
    for col in FUNDAMENTAL_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _add_growth(panel: pd.DataFrame, out_col: str, src_col: str) -> pd.DataFrame:
    """Year-on-year change of a joined TTM series, per ticker.

    Compares against the value in force roughly a year earlier on the panel's
    own calendar, so it never reaches for a figure that had not been filed.
    """
    if src_col not in panel.columns:
        panel[out_col] = np.nan
        return panel
    out = panel.sort_values(["ticker", "date"], kind="stable")
    prior = out.groupby("ticker", sort=False)[src_col].shift(252)
    growth = _safe_div(out[src_col] - prior, prior.abs())
    out[out_col] = growth
    return out.sort_index()
