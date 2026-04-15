"""Unit tests for TiingoFredProvider (mocked network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def _tiingo_deps():
    """Skip if tiingo optional deps are not installed."""
    pytest.importorskip("requests")
    pytest.importorskip("fredapi")


@pytest.fixture()
def provider(_tiingo_deps):
    from stock_predictor.providers.tiingo_provider import TiingoFredProvider

    return TiingoFredProvider(tiingo_api_key="fake-key", fred_api_key="fake-key")


def _tiingo_json(ticker: str, n_days: int = 5) -> list[dict]:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    return [
        {
            "date": d.isoformat(),
            "close": 100.0 + i,
            "adjClose": 100.0 + i,
            "volume": 1_000_000 + i * 1000,
            "adjVolume": 1_000_000 + i * 1000,
        }
        for i, d in enumerate(dates)
    ]


def test_download_equity_ohlcv(provider) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _tiingo_json("AAPL")

    with patch.object(provider._session, "get", return_value=mock_resp):
        adj, vol = provider.download_equity_ohlcv(["AAPL"], "2024-01-02", "2024-01-10")

    assert "AAPL" in adj.columns
    assert "AAPL" in vol.columns
    assert len(adj) == 5
    assert isinstance(adj.index, pd.DatetimeIndex)


def test_download_equity_ohlcv_ticker_not_found(provider) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch.object(provider._session, "get", return_value=mock_resp):
        adj, vol = provider.download_equity_ohlcv(["FAKE"], "2024-01-02", "2024-01-10")

    assert adj.empty
    assert vol.empty


def test_download_macro(provider) -> None:
    vix = pd.Series(
        [15.0, 16.0, 15.5, 14.8, 15.2],
        index=pd.bdate_range("2024-01-02", periods=5),
    )
    tnx = pd.Series(
        [4.0, 4.1, 4.05, 3.95, 4.0],
        index=pd.bdate_range("2024-01-02", periods=5),
    )
    irx = pd.Series(
        [5.2, 5.3, 5.25, 5.1, 5.2],
        index=pd.bdate_range("2024-01-02", periods=5),
    )

    def fake_get_series(series_id, **kwargs):
        return {"VIXCLS": vix, "DGS10": tnx, "DTB3": irx}[series_id]

    with patch.object(provider._fred, "get_series", side_effect=fake_get_series):
        macro = provider.download_macro("2024-01-02", "2024-01-10")

    assert set(macro.columns) >= {"date", "vix", "tnx_yield", "irx_yield"}
    assert len(macro) == 5
    assert not macro["vix"].isna().any()


def test_download_macro_partial_failure(provider) -> None:
    vix = pd.Series(
        [15.0, 16.0],
        index=pd.bdate_range("2024-01-02", periods=2),
    )

    def fake_get_series(series_id, **kwargs):
        if series_id == "VIXCLS":
            return vix
        raise Exception("FRED timeout")

    with patch.object(provider._fred, "get_series", side_effect=fake_get_series):
        macro = provider.download_macro("2024-01-02", "2024-01-10")

    assert "vix" in macro.columns
    assert len(macro) == 2
    # Missing series should be NaN-filled
    assert macro["tnx_yield"].isna().all()


def test_download_benchmark(provider) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _tiingo_json("SPY")

    with patch.object(provider._session, "get", return_value=mock_resp):
        close = provider.download_benchmark("SPY", "2024-01-02", "2024-01-10")

    assert isinstance(close, pd.Series)
    assert close.name == "SPY"
    assert len(close) == 5


def test_download_benchmark_failure(provider) -> None:
    with patch.object(provider._session, "get", side_effect=Exception("network error")):
        close = provider.download_benchmark("SPY", "2024-01-02", "2024-01-10")

    assert close.empty
    assert close.name == "SPY"
