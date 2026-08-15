"""Point-in-time S&P 500 membership helpers (community data source, not official S&P)."""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

SP500_STINTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)

_DEFAULT_UA = (
    "stock-predictor/0.1 (Python; educational research; "
    "https://github.com/fja05680/sp500)"
)


def load_sp500_stints(
    url: str = SP500_STINTS_URL,
    *,
    user_agent: str = _DEFAULT_UA,
    timeout: int = 60,
) -> pd.DataFrame:
    """Load membership stints: ticker, start_date, end_date (NaT = still in index per source)."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


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
