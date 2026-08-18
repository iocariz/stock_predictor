"""Data provider protocol and factory."""

from __future__ import annotations

import os
from typing import Protocol

import pandas as pd


def _load_dotenv() -> None:
    """Load .env from the project root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


class DataProvider(Protocol):
    """Minimal interface for downloading equity, macro, and benchmark data."""

    def download_equity_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (adj_close_wide, volume_wide) with DatetimeIndex rows, ticker columns."""
        ...

    def download_macro(
        self,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        """Return DataFrame with columns: date, vix, tnx_yield, irx_yield."""
        ...

    def download_benchmark(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.Series:
        """Return Series of adjusted close with DatetimeIndex, named *ticker*."""
        ...


def get_provider(name: str, *, batch_size: int | None = None) -> DataProvider:
    """Instantiate a data provider by name ('yfinance', 'tiingo' or 'hybrid').

    ``hybrid`` downloads through yfinance and falls back to Tiingo only for
    tickers Yahoo does not serve — mostly companies that were acquired,
    renamed or taken private. Those absences are the project's survivorship
    bias, and they are not random: they are disproportionately the failures.

    *batch_size* caps how many symbols go into one yfinance request. Lower it
    when Yahoo throttles a large universe; it has no effect on Tiingo, which
    is fetched per symbol.
    """
    if name == "yfinance":
        from stock_predictor.providers.yfinance_provider import YFinanceProvider

        if batch_size is not None:
            return YFinanceProvider(batch_size=batch_size)
        return YFinanceProvider()

    if name == "hybrid":
        # yfinance for the bulk, Tiingo only for the names Yahoo drops.
        _load_dotenv()
        tiingo_key = os.environ.get("TIINGO_API_KEY", "")
        if not tiingo_key:
            raise EnvironmentError(
                "TIINGO_API_KEY is required for --provider hybrid. "
                "Get a free key at https://www.tiingo.com"
            )
        from stock_predictor.providers.hybrid_provider import HybridProvider

        kw = {"batch_size": batch_size} if batch_size is not None else {}
        return HybridProvider(tiingo_api_key=tiingo_key, **kw)

    if name == "tiingo":
        _load_dotenv()
        tiingo_key = os.environ.get("TIINGO_API_KEY", "")
        fred_key = os.environ.get("FRED_API_KEY", "")
        if not tiingo_key:
            raise EnvironmentError(
                "TIINGO_API_KEY environment variable is required for --provider tiingo. "
                "Get a free key at https://www.tiingo.com"
            )
        if not fred_key:
            raise EnvironmentError(
                "FRED_API_KEY environment variable is required for --provider tiingo. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        try:
            from stock_predictor.providers.tiingo_provider import TiingoFredProvider
        except ImportError as exc:
            raise ImportError(
                "Tiingo provider requires extra dependencies. "
                "Install with: pip install stock-predictor[tiingo]  "
                "or: uv sync --extra tiingo"
            ) from exc

        return TiingoFredProvider(tiingo_api_key=tiingo_key, fred_api_key=fred_key)

    raise ValueError(
        f"Unknown provider: {name!r}. Choose 'yfinance', 'tiingo' or 'hybrid'."
    )
