"""Short-sale volume: a feature family that is not an OHLCV derivative.

Every feature the model currently sees -- price, volume, sector-relative,
regime, macro -- is arithmetic on one panel. Six research directions have come
back negative or neutral and validation peaks at 2-10 trees under every
configuration tried, which says the signal is not in the current inputs rather
than that the tuner is wrong. Fundamentals were the one other family tried;
they were used heavily by the model and still lost to the price-only control
at horizon 63.

FINRA publishes consolidated daily short-sale volume on a public CDN, one
dated file per session, no key:

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
    20261029|A|511423.702645|1090|1000128.599576|B,Q,N

Two properties make this worth the plumbing. It decomposes a session's volume
into short and non-short, which no amount of OHLCV arithmetic recovers. And it
is point-in-time by construction: a file dated D contains only what traded on
D, and a name that stops trading simply stops appearing -- so unlike the price
panel there is no survivorship repair to get wrong, and no vendor backfill to
distrust.

One hard limit: the archive starts in 2019. Any experiment on this feature is
bounded by that, and a walk-forward whose training window ends before 2019
cannot learn it at all.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

FINRA_DAILY = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
"""Consolidated tape. The per-venue files cover one exchange each, so a ratio
built from one of them is a ratio of the wrong denominator."""

ARCHIVE_START = "2019-01-01"
"""Earlier dates return 403. Stated here so a caller asking for 2010 gets a
bounded answer rather than a silently empty panel."""

DEFAULT_PUBLICATION_LAG = 1
"""Sessions to lag the feature behind the session it describes.

A file dated D is published after D's close. The pipeline already scores on
D's close and fills at D+1, so D's short volume is known before the fill and
same-day use would not strictly be lookahead. But a vendor publishing late
would turn that into lookahead for free, and the cost of insurance is one
session of staleness on a feature that moves slowly. Conservative by default.
"""

EXPECTED_HEADER = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
)

PANEL_COLUMNS = ["date", "ticker", "short_ratio", "short_volume", "total_volume"]

FEATURE_COLUMNS = [
    "short_ratio",
    "short_ratio_5d",
    "short_ratio_chg_5d",
    "short_ratio_xs_z",
]
"""Declared so a model trained with this family can still be scored without
it: the columns exist and are NaN rather than absent."""


def parse_short_volume_file(text: str) -> pd.DataFrame:
    """One FINRA daily file as ``(date, ticker, short_ratio, volumes)``.

    Raises on anything that is not a FINRA file. A 403 page served with status
    200 would otherwise parse to an empty frame, indistinguishable from a real
    quiet session -- and a quiet session is a plausible-looking lie.
    """
    first = text.lstrip().split("\n", 1)[0].strip()
    if not first.startswith("Date|Symbol|ShortVolume"):
        raise ValueError(
            f"not a FINRA short-volume file; header was {first[:60]!r}"
        )
    df = pd.read_csv(io.StringIO(text), sep="|", dtype={"Symbol": str})
    # A trailing summary line appears in some files.
    df = df[df["Date"].astype(str).str.fullmatch(r"\d{8}")]
    if df.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    out = pd.DataFrame({
        # From the row, not the filename: trusting a filename is how a
        # mislabelled file becomes silent lookahead.
        "date": pd.to_datetime(df["Date"].astype(str), format="%Y%m%d"),
        "ticker": df["Symbol"].astype(str).str.strip().str.upper(),
        "short_volume": pd.to_numeric(df["ShortVolume"], errors="coerce"),
        "total_volume": pd.to_numeric(df["TotalVolume"], errors="coerce"),
    })
    # 0/0 is not 0. A name that did not trade has no short share, and recording
    # one as zero reads as "nobody shorted it".
    denom = out["total_volume"].where(out["total_volume"] > 0)
    out["short_ratio"] = out["short_volume"] / denom
    return out[PANEL_COLUMNS].reset_index(drop=True)


def short_volume_panel(
    files: dict[pd.Timestamp, str], tickers: set[str] | None = None,
) -> pd.DataFrame:
    """Many daily files as one panel, optionally restricted to *tickers*.

    The files carry ~11,000 symbols against a universe of ~800, and every
    groupby downstream pays for the difference.
    """
    frames = []
    for _, text in sorted(files.items()):
        try:
            part = parse_short_volume_file(text)
        except ValueError:
            continue       # a bad file is skipped, not fatal
        if tickers is not None:
            part = part[part["ticker"].isin(tickers)]
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(frames, ignore_index=True)
    # A session published twice must not double-count into the rolling mean.
    panel = panel.drop_duplicates(subset=["date", "ticker"], keep="last")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def fetch_short_volume(
    start: str, end: str, cache_dir: Path, *, session=None,
) -> dict[pd.Timestamp, str]:
    """Download and cache the daily files covering ``[start, end]``.

    Cached per session file, so a re-run costs nothing and an interrupted run
    resumes. A date the archive does not serve is cached as empty so it is not
    re-asked -- the same convergence property the Tiingo cache needed, learned
    the same way.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    out: dict[pd.Timestamp, str] = {}
    lo = max(pd.Timestamp(start), pd.Timestamp(ARCHIVE_START))
    for day in pd.bdate_range(lo, pd.Timestamp(end)):
        key = day.strftime("%Y%m%d")
        cached = cache_dir / f"{key}.txt"
        if cached.exists():
            text = cached.read_text()
            if text.strip():
                out[day] = text
            continue
        try:
            resp = sess.get(FINRA_DAILY.format(date=key), timeout=30)
        except Exception as exc:  # noqa: BLE001 - one day must not kill the run
            # Not cached: a failure to ask is not an answer, which is the
            # distinction the Tiingo cache had to learn the hard way.
            print(f"    FINRA {key}: {type(exc).__name__}: {exc}")
            continue
        if resp.status_code == 200 and resp.text.lstrip().startswith("Date|"):
            cached.write_text(resp.text)
            out[day] = resp.text
        elif resp.status_code in (403, 404):
            # Holiday or outside the archive: an answer, so cache the absence.
            cached.write_text("")
        else:
            print(f"    FINRA {key}: HTTP {resp.status_code}")
    return out


def add_short_volume_features(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    lag_sessions: int = DEFAULT_PUBLICATION_LAG,
) -> pd.DataFrame:
    """Merge the short-volume family onto *features*, lagged and per-ticker.

    Row count is preserved: a feature merge that drops rows silently shrinks
    the training set. A ticker the files do not list comes back NaN rather
    than zero -- it is unmeasured, not unshorted, and LightGBM handles NaN
    natively.
    """
    out = features.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in FEATURE_COLUMNS:
        out[col] = float("nan")
    if panel is None or panel.empty:
        return out

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])
    g = p.groupby("ticker", observed=True)["short_ratio"]
    p["short_ratio_5d"] = g.transform(lambda s: s.rolling(5, min_periods=2).mean())
    p["short_ratio_chg_5d"] = g.transform(lambda s: s - s.shift(5))
    # Lag inside the group, so one name never inherits another's history.
    cols = ["short_ratio", "short_ratio_5d", "short_ratio_chg_5d"]
    p[cols] = p.groupby("ticker", observed=True)[cols].shift(lag_sessions)
    # Against the session's peers, not against the name's own history: the
    # model trades a cross-section.
    p["short_ratio_xs_z"] = p.groupby("date", observed=True)["short_ratio"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )

    merged = out.drop(columns=FEATURE_COLUMNS).merge(
        p[["date", "ticker", *FEATURE_COLUMNS]], on=["date", "ticker"], how="left",
    )
    assert len(merged) == len(features), "short-volume merge changed the row count"
    return merged
