"""Unit tests for data provider factory and implementations."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.data_provider import get_provider


def test_get_provider_yfinance() -> None:
    from stock_predictor.providers.yfinance_provider import YFinanceProvider

    provider = get_provider("yfinance")
    assert isinstance(provider, YFinanceProvider)


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("bloomberg")


def test_get_provider_tiingo_missing_tiingo_key() -> None:
    env = {"TIINGO_API_KEY": "", "FRED_API_KEY": "test"}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(EnvironmentError, match="TIINGO_API_KEY"):
            get_provider("tiingo")


def test_get_provider_tiingo_missing_fred_key() -> None:
    env = {"TIINGO_API_KEY": "test", "FRED_API_KEY": ""}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
            get_provider("tiingo")


class _FakeProvider:
    """Stub provider for testing DataFrame shape contracts."""

    def download_equity_ohlcv(
        self, tickers, start, end,
    ):
        dates = pd.bdate_range("2024-01-02", periods=5)
        adj_close = pd.DataFrame(
            np.random.default_rng(1).random((5, len(tickers))),
            index=dates,
            columns=tickers,
        )
        volume = pd.DataFrame(
            np.random.default_rng(2).integers(1_000_000, 5_000_000, (5, len(tickers))),
            index=dates,
            columns=tickers,
        )
        return adj_close, volume

    def download_macro(self, start, end):
        dates = pd.bdate_range("2024-01-02", periods=5)
        return pd.DataFrame({
            "date": dates,
            "vix": [15.0, 16.0, 15.5, 14.8, 15.2],
            "tnx_yield": [4.0, 4.1, 4.05, 3.95, 4.0],
            "irx_yield": [5.2, 5.3, 5.25, 5.1, 5.2],
        })

    def download_benchmark(self, ticker, start, end):
        dates = pd.bdate_range("2024-01-02", periods=5)
        return pd.Series(
            [500.0, 502.0, 501.0, 503.0, 505.0],
            index=dates,
            name=ticker,
        )


def test_provider_shape_contract_equity() -> None:
    p = _FakeProvider()
    adj, vol = p.download_equity_ohlcv(["AAPL", "MSFT"], "2024-01-02", "2024-01-10")
    assert list(adj.columns) == ["AAPL", "MSFT"]
    assert list(vol.columns) == ["AAPL", "MSFT"]
    assert isinstance(adj.index, pd.DatetimeIndex)
    assert len(adj) == 5


def test_provider_shape_contract_macro() -> None:
    p = _FakeProvider()
    macro = p.download_macro("2024-01-02", "2024-01-10")
    assert set(macro.columns) >= {"date", "vix", "tnx_yield", "irx_yield"}
    assert len(macro) == 5


def test_provider_shape_contract_benchmark() -> None:
    p = _FakeProvider()
    close = p.download_benchmark("SPY", "2024-01-02", "2024-01-10")
    assert isinstance(close, pd.Series)
    assert close.name == "SPY"
    assert len(close) == 5
