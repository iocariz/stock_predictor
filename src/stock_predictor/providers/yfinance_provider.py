"""YFinance data provider (default backend)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from stock_predictor.training import wide_field

MACRO_YF = ["^VIX", "^TNX", "^IRX"]

# Yahoo intermittently rate-limits single-ticker downloads (empty frame or
# exception); retry with exponential backoff before giving up.
_BENCHMARK_RETRIES = 4
_BENCHMARK_BACKOFF_S = 2.0


class YFinanceProvider:
    """Download equity, macro, and benchmark data via yfinance."""

    def download_equity_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
        adj_close = wide_field(data, "Close")
        volume = wide_field(data, "Volume")
        # Single-ticker downloads come back without a ticker column level, so
        # wide_field returns the field name ("Close") as the column — rename it
        # to the ticker so downstream stacking sees a real symbol.
        if len(tickers) == 1:
            if list(adj_close.columns) == ["Close"]:
                adj_close.columns = [tickers[0]]
            if list(volume.columns) == ["Volume"]:
                volume.columns = [tickers[0]]
        return adj_close, volume

    def download_macro(
        self,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        raw = yf.download(
            MACRO_YF,
            start=start,
            end=end,
            group_by="ticker",
            threads=True,
            auto_adjust=False,
            progress=False,
        )
        mw = _close_wide_macro(raw)
        mdf = mw.rename(
            columns={"^VIX": "vix", "^TNX": "tnx_yield", "^IRX": "irx_yield"},
        ).reset_index()
        dcol = mdf.columns[0]
        mdf = mdf.rename(columns={dcol: "date"})
        mdf["date"] = pd.to_datetime(mdf["date"]).dt.normalize()
        return mdf[["date", "vix", "tnx_yield", "irx_yield"]].copy()

    def download_benchmark(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.Series:
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=5)
        bench = pd.DataFrame()
        for attempt in range(_BENCHMARK_RETRIES):
            try:
                bench = yf.download(
                    ticker,
                    start=start,
                    end=end_ts.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                )
            except Exception as exc:
                print(f"  Benchmark download error ({ticker}): {exc}")
                bench = pd.DataFrame()
            if bench is not None and not bench.empty:
                break
            if attempt < _BENCHMARK_RETRIES - 1:
                wait = _BENCHMARK_BACKOFF_S * 2 ** attempt
                print(f"  Benchmark download empty ({ticker}); retry in {wait:.0f}s…")
                time.sleep(wait)
        if bench is None or bench.empty:
            return pd.Series(dtype=float, name=ticker)
        if isinstance(bench.columns, pd.MultiIndex):
            # Level order depends on group_by: (Price, Ticker) by default,
            # (Ticker, Price) with group_by="ticker" — find the Close level.
            level = next(
                (
                    i
                    for i in range(bench.columns.nlevels)
                    if "Close" in bench.columns.get_level_values(i)
                ),
                None,
            )
            if level is None:
                return pd.Series(dtype=float, name=ticker)
            close = bench.xs("Close", axis=1, level=level).squeeze()
        else:
            close = bench["Close"]
        close = close.dropna()
        if len(close) == 0:
            return pd.Series(dtype=float, name=ticker)
        idx = pd.DatetimeIndex(close.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        close.index = idx.normalize()
        close.name = ticker
        return close.astype(float)


def _close_wide_macro(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw[["Close"]].sort_index()
    if raw.columns.names[0] == "Price":
        return raw.xs("Close", level=0, axis=1).sort_index()
    return raw.xs("Close", level=-1, axis=1).sort_index()
