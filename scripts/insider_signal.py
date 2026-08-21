"""Measure SEC Form 3/4/5 insider buying against a scored panel.

Written to answer "should insider transactions be a feature?" before building
a pipeline for them. The answer on this universe was **no**, and this script
exists so that answer can be re-derived rather than trusted — see the
Limitations section of the README.

What it measures, all point-in-time (filings join on ``FILING_DATE``, never
``TRANS_DATE``: the market cannot act on a purchase before it is disclosed):

1. Standalone rank IC of trailing insider buying against forward returns.
2. Excess forward return of the names with insider buying, versus their own
   same-day cross-section — the test that matters when a signal is live on
   only a fraction of rows.
3. Whether a larger purchase is a stronger signal.

Only ``TRANS_CODE == "P"`` counts. Grants (A), option exercises (M), tax
withholding (F), gifts (G) and sales (S) carry no directional view, and
together they are roughly 94% of all reported transactions.

Two data caveats the output repeats, because both bound the conclusion:

* The 10b5-1 checkbox is **absent from the datasets before 2023q1**, so the
  discretionary-versus-scheduled split is only available on recent history.
* Excess returns here are raw, not sector- or style-neutral. Insider clusters
  form after drawdowns, so a negative reading on a momentum-led tape may be
  reporting a value tilt rather than an insider effect.

Usage:
    uv run python scripts/insider_signal.py artifacts/roles/wf_control.parquet
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.fundamentals import sec_user_agent
from stock_predictor.stats import hac_mean_tstat

BASE = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
)
OPEN_MARKET_PURCHASE = "P"


def quarters(start: str, end: str) -> list[str]:
    """Inclusive list of ``YYYYqN`` labels."""

    def parse(s: str) -> tuple[int, int]:
        y, q = s.lower().split("q")
        return int(y), int(q)

    (y, q), last = parse(start), parse(end)
    out = []
    while (y, q) <= last:
        out.append(f"{y}q{q}")
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def load_quarter(q: str, cache_dir: Path) -> pd.DataFrame | None:
    """Open-market purchases for one quarter, cached as parquet.

    Returns ``None`` when the quarter is not published yet; a quarter with no
    purchases caches as an empty frame so it is not re-fetched.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{q}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    try:
        req = urllib.request.Request(BASE.format(q=q), headers={"User-Agent": sec_user_agent()})
        with urllib.request.urlopen(req, timeout=180) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"  {q}: unavailable ({exc})", file=sys.stderr)
        return None

    cols = ["ticker", "filed", "value", "owner", "planned"]
    sub = pd.read_csv(archive.open("SUBMISSION.tsv"), sep="\t", low_memory=False)
    nd = pd.read_csv(archive.open("NONDERIV_TRANS.tsv"), sep="\t", low_memory=False)
    own = pd.read_csv(archive.open("REPORTINGOWNER.tsv"), sep="\t", low_memory=False)

    buys = nd[nd["TRANS_CODE"] == OPEN_MARKET_PURCHASE]
    if buys.empty:
        empty = pd.DataFrame(columns=cols)
        empty.to_parquet(cached, index=False)
        return empty

    # The 10b5-1 checkbox was added to Form 4 partway through this history.
    # Where the column is absent the flag is unknown, not False — assuming
    # "not scheduled" would fabricate a value on two thirds of the sample.
    if "AFF10B5ONE" not in sub.columns:
        sub = sub.assign(AFF10B5ONE=pd.NA)
        print(f"  {q}: no 10b5-1 column", file=sys.stderr)

    merged = buys.merge(
        sub[["ACCESSION_NUMBER", "ISSUERTRADINGSYMBOL", "FILING_DATE", "AFF10B5ONE"]],
        on="ACCESSION_NUMBER",
        how="left",
    ).merge(
        own[["ACCESSION_NUMBER", "RPTOWNERCIK"]].drop_duplicates(),
        on="ACCESSION_NUMBER",
        how="left",
    )
    merged["ticker"] = (
        merged["ISSUERTRADINGSYMBOL"].astype(str).str.upper().str.replace(".", "-", regex=False)
    )
    merged["filed"] = pd.to_datetime(merged["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    merged["value"] = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce") * pd.to_numeric(
        merged["TRANS_PRICEPERSHARE"], errors="coerce"
    )
    merged["planned"] = merged["AFF10B5ONE"].astype(str).str.lower().isin(("1", "true"))
    merged = merged.rename(columns={"RPTOWNERCIK": "owner"})
    out = merged.loc[merged["filed"].notna() & (merged["value"] > 0), cols]
    out.to_parquet(cached, index=False)
    return out


def build_signal(panel: pd.DataFrame, filings: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Trailing-*horizon* insider buying, on the panel's own session calendar."""
    sessions = np.sort(panel["date"].unique())
    # A filing lodged on a Saturday becomes actionable on the next session.
    idx = np.searchsorted(sessions, filings["filed"].to_numpy(), side="left")
    inside = idx < len(sessions)
    dated = filings.loc[inside].copy()
    dated["date"] = sessions[idx[inside]]

    daily = (
        dated.groupby(["ticker", "date"])
        .agg(
            buy_value=("value", "sum"),
            buyers=("owner", "nunique"),
            discretionary=("planned", lambda s: int((~s).sum())),
        )
        .reset_index()
    )

    out = panel.merge(daily, on=["ticker", "date"], how="left")
    measures = ("buy_value", "buyers", "discretionary")
    for col in measures:
        out[col] = out[col].fillna(0.0)
    out = out.sort_values(["ticker", "date"], kind="stable")
    grouped = out.groupby("ticker", sort=False)
    for col in measures:
        out[f"{col}_trail"] = grouped[col].transform(
            lambda s: s.rolling(horizon, min_periods=1).sum()
        )
    return out


def _excess(lab: pd.DataFrame, mask: pd.Series, label: str, horizon: int) -> None:
    """Excess forward return of *mask* against its own same-day cross-section."""
    subset = lab[mask]
    if subset.empty:
        print(f"{label:34s} no rows")
        return
    per_date = (
        subset.groupby("date")["fwd_ret"].mean() - lab.groupby("date")["fwd_ret"].mean()
    ).dropna()
    if len(per_date) < 10:
        print(f"{label:34s} too few dates ({len(per_date)})")
        return
    mean, t_stat, _ = hac_mean_tstat(per_date.to_numpy(), overlap=horizon)
    width = int(subset.groupby("date").size().mean())
    print(
        f"{label:34s} {mean:+8.4%}  HAC t {t_stat:+6.2f}  "
        f"~{width:3d} names/date, {len(per_date)} dates"
    )


def report(df: pd.DataFrame, horizon: int) -> None:
    lab = df[df["fwd_ret"].notna()]
    print(f"\nlabelled rows: {len(lab):,} | tickers {lab['ticker'].nunique()}")

    live = lab["buy_value_trail"] > 0
    print(f"rows with any insider buy in trailing {horizon}d : {live.mean():6.2%}")
    print(f"rows with a cluster buy (>=2 buyers)       : {(lab['buyers_trail'] >= 2).mean():6.2%}")

    ic = (
        lab.groupby("date")
        .apply(
            lambda g: g["buy_value_trail"].corr(g["fwd_ret"], method="spearman"),
            include_groups=False,
        )
        .dropna()
    )
    mean, t_stat, _ = hac_mean_tstat(ic.to_numpy(), overlap=horizon)
    print(
        f"\n1) standalone rank IC (all names): {mean:+.4f}  HAC t {t_stat:+.2f}  n={len(ic)} dates"
    )

    print("\n2) excess forward return vs the same-day universe:")
    _excess(lab, live, "   any insider buy", horizon)
    _excess(lab, lab["buyers_trail"] >= 2, "   cluster buy (>=2 buyers)", horizon)
    _excess(lab, lab["discretionary_trail"] > 0, "   discretionary (non-10b5-1)", horizon)
    _excess(lab, lab["buyers_trail"] >= 3, "   strong cluster (>=3 buyers)", horizon)

    if live.sum() > 1000:
        buying = lab[live]
        size_ic = (
            buying.groupby("date")
            .apply(
                lambda g: (
                    g["buy_value_trail"].corr(g["fwd_ret"], method="spearman")
                    if len(g) >= 5
                    else np.nan
                ),
                include_groups=False,
            )
            .dropna()
        )
        if len(size_ic) > 10:
            mean, t_stat, _ = hac_mean_tstat(size_ic.to_numpy(), overlap=horizon)
            print(
                f"\n3) rank IC of buy SIZE within the buying names: "
                f"{mean:+.4f}  HAC t {t_stat:+.2f}  n={len(size_ic)} dates"
            )

    print(
        "\nExcess returns are raw, not sector- or style-neutral. Insider clusters\n"
        "form after drawdowns, so a negative reading on a momentum-led tape may\n"
        "be reporting a value tilt rather than an insider effect."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("panel", type=Path, help="Scored panel with date, ticker, fwd_ret")
    p.add_argument("--horizon", type=int, default=63, help="Trailing window, sessions")
    p.add_argument("--start-quarter", default="2018q4", dest="start_quarter")
    p.add_argument("--end-quarter", default="2026q3", dest="end_quarter")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/insider_cache"),
        dest="cache_dir",
        help="Per-quarter parquet cache (~400MB for the full history)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    panel = pd.read_parquet(args.panel, columns=["date", "ticker", "fwd_ret"])
    panel["date"] = pd.to_datetime(panel["date"])
    universe = set(panel["ticker"].unique())
    print(
        f"panel {len(panel):,} rows, {len(universe)} tickers, "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()}"
    )

    labels = quarters(args.start_quarter, args.end_quarter)
    print(f"\nloading {len(labels)} quarters of Form 3/4/5 data…")
    frames = [
        df[df["ticker"].isin(universe)]
        for df in (load_quarter(q, args.cache_dir) for q in labels)
        if df is not None and len(df)
    ]
    if not frames:
        sys.exit("No insider data available for the requested quarters.")
    filings = pd.concat(frames, ignore_index=True)
    print(
        f"open-market purchases on this universe: {len(filings):,} rows, "
        f"{filings['ticker'].nunique()} tickers"
    )

    report(build_signal(panel, filings, args.horizon), args.horizon)


if __name__ == "__main__":
    main()
