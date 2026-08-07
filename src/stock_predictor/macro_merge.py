"""Merge Yahoo Finance and FRED macro panels to fill per-date gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Same series IDs as TiingoFredProvider (avoid importing tiingo → training cycle).
FRED_SERIES = {
    "vix": "VIXCLS",
    "tnx_yield": "DGS10",
    "irx_yield": "DTB3",
}

MACRO_COLS = ("vix", "tnx_yield", "irx_yield")


def download_macro_fred(start: str, end: str | None, api_key: str) -> pd.DataFrame:
    """Fetch VIX + Treasury yields from FRED (same shape as yfinance macro path)."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "FRED macro fallback requires fredapi. Install: uv sync --extra tiingo"
        ) from exc

    fred = Fred(api_key=api_key)
    series: dict[str, pd.Series] = {}
    for col, fred_id in FRED_SERIES.items():
        try:
            s = fred.get_series(fred_id, observation_start=start, observation_end=end)
            series[col] = s.dropna()
        except Exception as exc:
            print(f"FRED: failed to fetch {fred_id}: {exc}")

    if not series:
        return pd.DataFrame(columns=["date", *MACRO_COLS])

    macro = pd.DataFrame(series)
    idx = pd.to_datetime(macro.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    macro.index = idx.normalize()
    macro = macro.ffill()
    macro = macro.reset_index().rename(columns={"index": "date"})
    for col in MACRO_COLS:
        if col not in macro.columns:
            macro[col] = np.nan
    return macro[["date", *MACRO_COLS]].copy()


def merge_macro_panels(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Per column: use *primary* where non-NaN, else *fallback* (aligned on *date*)."""
    cols = ["date", *MACRO_COLS]
    if primary.empty and fallback.empty:
        return pd.DataFrame(columns=cols)
    if primary.empty:
        out = fallback[cols].copy() if all(c in fallback.columns for c in cols) else fallback.copy()
    elif fallback.empty:
        out = primary[cols].copy() if all(c in primary.columns for c in cols) else primary.copy()
    else:
        p = primary[[c for c in cols if c in primary.columns]].copy()
        f = fallback[[c for c in cols if c in fallback.columns]].copy()
        for df in (p, f):
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for col in MACRO_COLS:
            if col not in p.columns:
                p[col] = np.nan
            if col not in f.columns:
                f[col] = np.nan
        pi = p.set_index("date").sort_index()
        fi = f.set_index("date").sort_index()
        merged = pi.combine_first(fi).sort_index()
        out = merged.reset_index()
    for col in MACRO_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[["date", *MACRO_COLS]].copy()
