"""Tiingo (equities) + FRED (macro) data provider."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from fredapi import Fred

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"

# FRED series that map to the yfinance macro tickers.
FRED_SERIES = {
    "vix": "VIXCLS",
    "tnx_yield": "DGS10",
    "irx_yield": "DTB3",
}

# Tiingo free tier: ~50 requests/hour → sequential with backoff is safer
_MAX_RETRIES = 3
_RETRY_WAIT = 5.0  # seconds between retries on 429


class TiingoFredProvider:
    """Download equity data from Tiingo and macro data from FRED."""

    def __init__(self, *, tiingo_api_key: str, fred_api_key: str) -> None:
        self._tiingo_key = tiingo_api_key
        self._fred = Fred(api_key=fred_api_key)
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {tiingo_api_key}",
        })

    def _get_with_retry(self, url: str, params: dict | None = None) -> requests.Response:
        """GET with retry on 429 rate-limit."""
        for attempt in range(_MAX_RETRIES):
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code != 429:
                return resp
            wait = _RETRY_WAIT * (attempt + 1)
            print(f"  Tiingo rate-limited (429), waiting {wait:.0f}s…")
            time.sleep(wait)
        return resp  # return last response even if still 429

    # ------------------------------------------------------------------
    # Equities
    # ------------------------------------------------------------------

    def download_equity_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []
        rate_limited = False

        def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame | None, str]:
            try:
                url = f"{TIINGO_BASE}/{ticker}/prices"
                params: dict[str, str] = {"startDate": start}
                if end:
                    params["endDate"] = end
                resp = self._get_with_retry(url, params)
                if resp.status_code == 404:
                    return ticker, None, "not found"
                if resp.status_code == 429:
                    return ticker, None, "rate-limited"
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    return ticker, None, "empty"
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
                df = df.set_index("date")[["adjClose", "adjVolume"]]
                df.columns = ["close", "volume"]
                return ticker, df, ""
            except Exception as exc:
                return ticker, None, str(exc)

        # Use fewer workers to avoid overwhelming the rate limit
        max_workers = 4 if len(tickers) > 50 else 8
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker, df, reason = fut.result()
                if df is not None and not df.empty:
                    results[ticker] = df
                else:
                    failed.append(ticker)
                    if reason == "rate-limited":
                        rate_limited = True

        if failed:
            print(f"Tiingo: {len(failed)} tickers failed to download")
            if rate_limited:
                print(
                    "  Rate-limited by Tiingo (free tier: ~50 req/hour). "
                    "Wait an hour or upgrade at https://api.tiingo.com/pricing"
                )

        if not results:
            empty_idx = pd.DatetimeIndex([], name="date")
            return (
                pd.DataFrame(index=empty_idx, columns=tickers, dtype=float),
                pd.DataFrame(index=empty_idx, columns=tickers, dtype=float),
            )

        adj_close = pd.DataFrame({t: df["close"] for t, df in results.items()})
        volume = pd.DataFrame({t: df["volume"] for t, df in results.items()})
        adj_close.index.name = "Date"
        volume.index.name = "Date"
        return adj_close.sort_index(), volume.sort_index()

    # ------------------------------------------------------------------
    # Macro (FRED)
    # ------------------------------------------------------------------

    def download_macro(
        self,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        series: dict[str, pd.Series] = {}
        for col, fred_id in FRED_SERIES.items():
            try:
                s = self._fred.get_series(fred_id, observation_start=start, observation_end=end)
                s = s.dropna()
                series[col] = s
            except Exception as exc:
                print(f"FRED: failed to fetch {fred_id}: {exc}")

        if not series:
            return pd.DataFrame(columns=["date", "vix", "tnx_yield", "irx_yield"])

        macro = pd.DataFrame(series)
        idx = pd.to_datetime(macro.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        macro.index = idx.normalize()
        # FRED has gaps on holidays; forward-fill so we have values every trading day
        macro = macro.ffill()
        macro = macro.reset_index().rename(columns={"index": "date"})
        for col in ["vix", "tnx_yield", "irx_yield"]:
            if col not in macro.columns:
                macro[col] = np.nan
        return macro[["date", "vix", "tnx_yield", "irx_yield"]].copy()

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def download_benchmark(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.Series:
        try:
            url = f"{TIINGO_BASE}/{ticker}/prices"
            resp = self._get_with_retry(url, {"startDate": start, "endDate": end})
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return pd.Series(dtype=float, name=ticker)
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
            close = df.set_index("date")["adjClose"].astype(float)
            close.name = ticker
            return close.sort_index()
        except Exception as exc:
            print(f"Tiingo benchmark download failed ({ticker}): {exc}")
            return pd.Series(dtype=float, name=ticker)
