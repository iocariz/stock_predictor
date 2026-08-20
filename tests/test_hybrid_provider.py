"""Hybrid provider: Yahoo for the bulk, Tiingo for the names it drops.

Survivorship is the point. The names Yahoo omits are disproportionately the
failures, so recovering them should make results *worse* and more honest.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.providers.hybrid_provider import (
    HybridProvider,
    TiingoRateLimited,
)

IDX = pd.bdate_range("2024-01-02", periods=6)


class _FakeYF:
    """Stands in for yfinance: serves `served`, silently drops the rest."""

    def __init__(self, served: list[str]):
        self.served = served
        self.calls: list[list[str]] = []

    def download_equity_ohlcv(self, tickers, start, end):
        self.calls.append(list(tickers))
        cols = [t for t in tickers if t in self.served]
        if not cols:
            return pd.DataFrame(), pd.DataFrame()
        data = np.arange(len(IDX) * len(cols), dtype=float).reshape(len(IDX), len(cols))
        return (pd.DataFrame(data + 100, index=IDX, columns=cols),
                pd.DataFrame(data + 1000, index=IDX, columns=cols))

    def download_macro(self, start, end):
        return pd.DataFrame({"date": IDX, "vix": 15.0, "tnx_yield": 4.0, "irx_yield": 5.0})

    def download_benchmark(self, ticker, start, end):
        return pd.Series(1.0, index=IDX, name=ticker)


def _tiingo_rows(n: int = 6, start: str = "2024-01-02"):
    dates = pd.bdate_range(start, periods=n)
    return [{"date": d.strftime("%Y-%m-%dT00:00:00.000Z"),
             "adjClose": 50.0 + i, "adjVolume": 5000 + i}
            for i, d in enumerate(dates)]


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def _provider(served, tmp_path, **kw) -> HybridProvider:
    return HybridProvider(
        tiingo_api_key="k", cache_dir=tmp_path, yf_provider=_FakeYF(served),
        pause_s=0.0, backoff_s=0.0, **kw,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_a_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="tiingo_api_key"):
        HybridProvider(tiingo_api_key="", cache_dir=tmp_path)


def test_yahoo_only_path_never_touches_tiingo(tmp_path) -> None:
    p = _provider(["AAA", "BBB"], tmp_path)
    with patch.object(p, "_fetch_one", side_effect=AssertionError("must not call")):
        adj, vol = p.download_equity_ohlcv(["AAA", "BBB"], "2024-01-01", None)
    assert list(adj.columns) == ["AAA", "BBB"]
    assert list(vol.columns) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_delisted_names_are_recovered_from_tiingo(tmp_path) -> None:
    """The whole point: a name Yahoo drops must end up in the panel."""
    p = _provider(["AAA"], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, _tiingo_rows())
        adj, vol = p.download_equity_ohlcv(["AAA", "DEAD"], "2024-01-01", None)
    assert list(adj.columns) == ["AAA", "DEAD"], "requested order preserved"
    assert adj["DEAD"].notna().sum() == 6
    assert vol["DEAD"].notna().sum() == 6


def test_recovered_prices_are_split_and_dividend_adjusted(tmp_path) -> None:
    """adjClose/adjVolume, to match yfinance auto_adjust=True."""
    p = _provider([], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, [{
            "date": "2024-01-02T00:00:00.000Z",
            "close": 999.0, "adjClose": 50.0,
            "volume": 1, "adjVolume": 5000,
        }])
        adj, vol = p.download_equity_ohlcv(["DEAD"], "2024-01-01", None)
    assert adj["DEAD"].iloc[0] == 50.0, "must use adjClose, not raw close"
    assert vol["DEAD"].iloc[0] == 5000


def test_tiingo_dates_outside_yahoo_calendar_are_kept(tmp_path) -> None:
    """A delisted name's history predates nothing Yahoo returned for it, so
    the index is unioned rather than reindexed onto Yahoo's calendar."""
    p = _provider(["AAA"], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(
            200, _tiingo_rows(n=4, start="2023-12-20"),
        )
        adj, _ = p.download_equity_ohlcv(["AAA", "DEAD"], "2023-12-01", None)
    assert adj.index.min() < IDX.min(), "earlier Tiingo sessions must survive"
    assert adj["DEAD"].notna().sum() == 4


def test_a_ticker_neither_source_has_is_simply_absent(tmp_path) -> None:
    p = _provider(["AAA"], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, [])
        adj, _ = p.download_equity_ohlcv(["AAA", "GHOST"], "2024-01-01", None)
    assert "GHOST" not in adj.columns
    assert list(adj.columns) == ["AAA"]


# ---------------------------------------------------------------------------
# Rate limits and caching
# ---------------------------------------------------------------------------


def test_results_are_cached_and_not_refetched(tmp_path) -> None:
    p = _provider(["AAA"], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, _tiingo_rows())
        p.download_equity_ohlcv(["AAA", "DEAD"], "2024-01-01", None)
        first = sess.return_value.get.call_count
        p.download_equity_ohlcv(["AAA", "DEAD"], "2024-01-01", None)
        assert sess.return_value.get.call_count == first, "cache must be reused"
    assert (tmp_path / "DEAD.parquet").exists()


def test_empty_results_are_cached_too(tmp_path) -> None:
    """Otherwise every run re-spends rate limit asking about the same ghost."""
    p = _provider([], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, [])
        p.download_equity_ohlcv(["GHOST"], "2024-01-01", None)
        n = sess.return_value.get.call_count
        p.download_equity_ohlcv(["GHOST"], "2024-01-01", None)
        assert sess.return_value.get.call_count == n


def test_rate_limit_is_retried_then_stops_cleanly(tmp_path) -> None:
    """Hitting the daily cap must keep what was recovered, not lose the run."""
    p = _provider(["AAA"], tmp_path, max_retries=2)
    calls = {"n": 0}

    def _get(url, params=None, timeout=None):
        calls["n"] += 1
        # First ticker succeeds, everything after is rate-limited.
        return _Resp(200, _tiingo_rows()) if calls["n"] == 1 else _Resp(429)

    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.side_effect = _get
        adj, _ = p.download_equity_ohlcv(
            ["AAA", "D1", "D2", "D3"], "2024-01-01", None,
        )
    assert "D1" in adj.columns, "the one that succeeded must be kept"
    assert "D3" not in adj.columns
    assert (tmp_path / "D1.parquet").exists(), "partial progress must be resumable"


def test_rate_limit_raises_from_the_single_fetch(tmp_path) -> None:
    p = _provider([], tmp_path, max_retries=2)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(429)
        with pytest.raises(TiingoRateLimited):
            p._fetch_one("DEAD", "2024-01-01", None)


def test_network_error_on_one_ticker_does_not_kill_the_run(tmp_path) -> None:
    p = _provider(["AAA"], tmp_path)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.side_effect = OSError("connection reset")
        adj, _ = p.download_equity_ohlcv(["AAA", "DEAD"], "2024-01-01", None)
    assert list(adj.columns) == ["AAA"]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_macro_and_benchmark_delegate_to_yfinance(tmp_path) -> None:
    p = _provider(["AAA"], tmp_path)
    assert set(p.download_macro("2024-01-01", None).columns) >= {"date", "vix"}
    assert p.download_benchmark("SPY", "2024-01-01", "2024-01-10").name == "SPY"


def test_placeholder_columns_do_not_duplicate_recovered_tickers(tmp_path) -> None:
    """Regression: Yahoo returns an all-NaN column for a name it cannot serve.
    Concatenating Tiingo's version on top produced two columns for the same
    ticker — a real fetch came back 1006 wide for 845 requested — which then
    yields duplicate rows on stack()."""
    class _PlaceholderYF(_FakeYF):
        def download_equity_ohlcv(self, tickers, start, end):
            adj, vol = super().download_equity_ohlcv(tickers, start, end)
            for t in tickers:
                if t not in self.served:
                    adj[t] = np.nan      # the placeholder Yahoo actually emits
                    vol[t] = np.nan
            return adj, vol

    p = HybridProvider(tiingo_api_key="k", cache_dir=tmp_path,
                       yf_provider=_PlaceholderYF(["AAA"]), pause_s=0.0)
    with patch.object(p, "_get_session") as sess:
        sess.return_value.get.return_value = _Resp(200, _tiingo_rows())
        adj, vol = p.download_equity_ohlcv(["AAA", "DEAD"], "2024-01-01", None)

    assert list(adj.columns) == ["AAA", "DEAD"]
    assert not adj.columns.duplicated().any()
    assert not vol.columns.duplicated().any()
    assert adj["DEAD"].notna().sum() == 6, "Tiingo data must win over the placeholder"
