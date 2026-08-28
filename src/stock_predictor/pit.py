"""Point-in-time S&P 500 membership helpers (community data source, not official S&P)."""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SP500_STINTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)

_DEFAULT_UA = (
    "stock-predictor/0.1 (Python; educational research; "
    "https://github.com/fja05680/sp500)"
)


DEFAULT_STINTS_CACHE = (
    Path(__file__).resolve().parents[2] / "artifacts" / "cache" / "sp500_stints.csv"
)
STINTS_CACHE_MAX_AGE_DAYS = 7


def _finalize_stints(df: pd.DataFrame, apply_renames: bool) -> pd.DataFrame:
    """Resolve renamed symbols, unless explicitly disabled.

    Applied at every return path rather than at the call sites, because a
    loader with three exits is a loader where one of them forgets.
    """
    if not apply_renames:
        return df
    from stock_predictor.renames import canonicalize_stints

    return canonicalize_stints(df)


def _read_stints_csv(raw: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(raw))
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def load_sp500_stints(
    url: str = SP500_STINTS_URL,
    *,
    user_agent: str = _DEFAULT_UA,
    timeout: int = 60,
    cache_path: Path | None = None,
    max_age_days: float = STINTS_CACHE_MAX_AGE_DAYS,
    retries: int = 3,
    backoff_s: float = 2.0,
    apply_renames: bool = True,
) -> pd.DataFrame:
    """Load membership stints: ticker, start_date, end_date (NaT = still in index per source).

    Cached on disk because every training run needs it and the upstream host
    rate-limits: a day of repeated runs earns HTTP 429 and every run then dies
    before it starts. Index membership changes a few times a month, so a
    week-old copy is not a research compromise.

    A fresh cache is used directly. Otherwise the fetch is retried with
    backoff, and if it still fails a **stale** cache is preferred over
    failing — a slightly old membership table beats no run at all.

    Tickers are resolved through :mod:`stock_predictor.renames` unless
    *apply_renames* is off. The source names companies by the symbol they used
    at the time, while prices are served under the symbol they use now, so
    Anthem's 2002-2022 membership pointed at ``ANTM`` — a symbol nothing
    prices — and dropped out of every panel. Resolving to ``ELV`` reattaches it
    and rejoins the two halves of what was always one continuous membership.
    """
    cache = Path(cache_path) if cache_path is not None else DEFAULT_STINTS_CACHE
    fresh = (
        cache.exists()
        and (time.time() - cache.stat().st_mtime) < max_age_days * 86400
    )
    if fresh:
        return _finalize_stints(
            _read_stints_csv(cache.read_text(encoding="utf-8")), apply_renames)

    raw = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries - 1:
                wait = backoff_s * 2**attempt
                print(f"  PIT stints fetch failed ({exc}); retry in {wait:.0f}s…")
                time.sleep(wait)
            elif cache.exists():
                age_d = (time.time() - cache.stat().st_mtime) / 86400
                print(f"  PIT stints fetch failed ({exc}); using cached copy "
                      f"{age_d:.1f} days old")
                return _finalize_stints(
                    _read_stints_csv(cache.read_text(encoding="utf-8")), apply_renames)
            else:
                raise

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    except OSError as exc:  # a read-only checkout must not fail the run
        print(f"  Could not cache PIT stints ({exc})")
    return _finalize_stints(_read_stints_csv(raw), apply_renames)


def tickers_overlapping_window(
    stints: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> list[str]:
    """Tickers with at least one stint overlapping [start, end] (end exclusive if set)."""
    ps = pd.Timestamp(start)
    pe = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    s = stints["start_date"]
    e = stints["end_date"]
    mask = (s <= pe) & (e.isna() | (e > ps))
    return sorted(stints.loc[mask, "ticker"].unique().tolist())


def current_members(stints: pd.DataFrame) -> set[str]:
    """Tickers still in the index per the source (latest stint has no end date).

    Useful for judging a price download: vendors reliably serve current
    members, so a gap there means a broken or throttled request. Departed
    members are a different matter — Yahoo drops most of them, which is the
    survivorship bias this project documents rather than a download fault.
    """
    if stints.empty:
        return set()
    open_stints = stints["end_date"].isna()
    return set(stints.loc[open_stints, "ticker"].astype(str).unique())


def filter_panel_to_pit(
    panel: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Keep rows where (date, ticker) falls in any stint [start_date, end_date) per source."""
    m = panel.merge(stints, on=ticker_col, how="inner")
    in_stint = (m[date_col] >= m["start_date"]) & (
        m["end_date"].isna() | (m[date_col] < m["end_date"])
    )
    m = m.loc[in_stint].drop(columns=["start_date", "end_date"])
    return m.drop_duplicates(subset=[date_col, ticker_col], keep="first")
