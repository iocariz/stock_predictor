"""Batched equity downloads for the yfinance provider.

Regression cover for the single unbatched ``yf.download`` over ~1000 symbols:
Yahoo rate-limits large requests and returns a partial frame, which used to
propagate as a silently truncated universe.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor.providers import yfinance_provider as yp

IDX = pd.bdate_range("2024-01-02", periods=4)


def _multi(tickers: list[str]) -> pd.DataFrame:
    """A group_by='ticker' style frame: MultiIndex (Ticker, Price)."""
    cols = pd.MultiIndex.from_product(
        [tickers, ["Close", "Volume"]], names=["Ticker", "Price"],
    )
    rng = np.random.default_rng(abs(hash(tuple(tickers))) % 2**31)
    data = rng.random((len(IDX), len(tickers) * 2)) * 100 + 10
    return pd.DataFrame(data, index=IDX, columns=cols)


def _universe(n: int) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _serve(fail_for: set[str] | None = None, empty_for: set[str] | None = None):
    """Fake yf.download that answers whatever batch it is handed."""
    fail_for = fail_for or set()
    empty_for = empty_for or set()

    def _download(tickers, **kwargs):
        batch = [tickers] if isinstance(tickers, str) else list(tickers)
        if any(t in fail_for for t in batch):
            raise RuntimeError("simulated Yahoo rate limit")
        served = [t for t in batch if t not in empty_for]
        if not served:
            return pd.DataFrame()
        return _multi(served)

    return _download


# ---------------------------------------------------------------------------
# Batching mechanics
# ---------------------------------------------------------------------------


def test_large_universe_is_split_into_batches() -> None:
    tickers = _universe(250)
    provider = yp.YFinanceProvider(batch_size=100)
    with (
        patch.object(yp.yf, "download", side_effect=_serve()) as dl,
        patch.object(yp.time, "sleep"),
    ):
        adj, vol = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")

    assert dl.call_count == 3  # 100 + 100 + 50
    sizes = sorted(len(c.args[0] or c.kwargs["tickers"]) for c in dl.call_args_list)
    assert sizes == [50, 100, 100]
    assert list(adj.columns) == tickers
    assert list(vol.columns) == tickers


def test_batches_are_concatenated_on_a_shared_date_index() -> None:
    tickers = _universe(30)
    provider = yp.YFinanceProvider(batch_size=10)
    with (
        patch.object(yp.yf, "download", side_effect=_serve()),
        patch.object(yp.time, "sleep"),
    ):
        adj, _ = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")

    assert isinstance(adj.index, pd.DatetimeIndex)
    assert list(adj.index) == list(IDX)
    assert adj.notna().all().all()


def test_single_ticker_batch_is_named_after_the_ticker() -> None:
    """A universe of 101 with batch_size 100 leaves a 1-ticker tail batch,
    which yfinance returns without a ticker column level."""
    flat = pd.DataFrame(
        {"Close": [10.0, 10.5, 11.0, 11.5], "Volume": [100, 110, 120, 130]}, index=IDX,
    )

    def _download(tickers, **kwargs):
        batch = [tickers] if isinstance(tickers, str) else list(tickers)
        return flat.copy() if len(batch) == 1 else _multi(batch)

    tickers = _universe(101)
    provider = yp.YFinanceProvider(batch_size=100)
    with (
        patch.object(yp.yf, "download", side_effect=_download),
        patch.object(yp.time, "sleep"),
    ):
        adj, vol = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")

    assert "T100" in adj.columns
    assert "Close" not in adj.columns
    assert "Volume" not in vol.columns
    assert adj["T100"].tolist() == [10.0, 10.5, 11.0, 11.5]


def test_batch_smaller_than_universe_preserves_requested_order() -> None:
    tickers = ["ZZZ", "AAA", "MMM", "BBB", "QQQ"]
    provider = yp.YFinanceProvider(batch_size=2)
    with (
        patch.object(yp.yf, "download", side_effect=_serve()),
        patch.object(yp.time, "sleep"),
    ):
        adj, _ = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")
    assert list(adj.columns) == tickers


def test_duplicate_tickers_are_requested_once() -> None:
    provider = yp.YFinanceProvider(batch_size=100)
    with (
        patch.object(yp.yf, "download", side_effect=_serve()) as dl,
        patch.object(yp.time, "sleep"),
    ):
        adj, _ = provider.download_equity_ohlcv(
            ["AAA", "BBB", "AAA", "BBB"], "2024-01-02", "2024-01-08",
        )
    assert list(dl.call_args_list[0].args[0]) == ["AAA", "BBB"]
    assert list(adj.columns) == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_failing_batch_is_retried_with_backoff() -> None:
    calls = {"n": 0}

    def _download(tickers, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated Yahoo rate limit")
        return _multi(list(tickers))

    provider = yp.YFinanceProvider(batch_size=100, max_retries=3)
    with (
        patch.object(yp.yf, "download", side_effect=_download),
        patch.object(yp.time, "sleep") as slept,
    ):
        adj, _ = provider.download_equity_ohlcv(_universe(10), "2024-01-02", "2024-01-08")

    assert calls["n"] == 2
    assert slept.call_count >= 1
    assert len(adj.columns) == 10


def test_one_dead_batch_does_not_lose_the_others() -> None:
    """Regression: a partial reply used to become the whole universe."""
    tickers = _universe(30)
    dead = set(tickers[10:20])
    provider = yp.YFinanceProvider(batch_size=10, max_retries=2)
    with (
        patch.object(yp.yf, "download", side_effect=_serve(fail_for=dead)),
        patch.object(yp.time, "sleep"),
    ):
        adj, vol = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")

    survived = set(adj.columns)
    assert survived == set(tickers) - dead
    assert len(survived) == 20
    assert set(vol.columns) == survived


def test_total_failure_returns_empty_frames_not_an_exception() -> None:
    """The coverage guard, not the provider, decides whether a run continues."""
    tickers = _universe(20)
    provider = yp.YFinanceProvider(batch_size=10, max_retries=2)
    with (
        patch.object(yp.yf, "download", side_effect=_serve(fail_for=set(tickers))),
        patch.object(yp.time, "sleep"),
    ):
        adj, vol = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")
    assert adj.empty and vol.empty


def test_tickers_yahoo_cannot_serve_are_simply_absent() -> None:
    tickers = _universe(20)
    unknown = {"T005", "T017"}
    provider = yp.YFinanceProvider(batch_size=10)
    with (
        patch.object(yp.yf, "download", side_effect=_serve(empty_for=unknown)),
        patch.object(yp.time, "sleep"),
    ):
        adj, _ = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")
    assert set(adj.columns) == set(tickers) - unknown


def test_missing_tickers_get_a_smaller_second_pass() -> None:
    """A batch that fails wholesale is retried in smaller chunks, so one bad
    symbol does not cost the whole batch."""
    state = {"first_pass_done": False}

    def _download(tickers, **kwargs):
        batch = [tickers] if isinstance(tickers, str) else list(tickers)
        # Big batches fail; small retry batches succeed.
        if len(batch) > 5:
            raise RuntimeError("simulated Yahoo rate limit")
        state["first_pass_done"] = True
        return _multi(batch)

    tickers = _universe(20)
    provider = yp.YFinanceProvider(batch_size=10, max_retries=1, retry_batch_size=5)
    with (
        patch.object(yp.yf, "download", side_effect=_download),
        patch.object(yp.time, "sleep"),
    ):
        adj, _ = provider.download_equity_ohlcv(tickers, "2024-01-02", "2024-01-08")

    assert state["first_pass_done"]
    assert set(adj.columns) == set(tickers), "second pass did not recover the universe"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_batch_size_is_bounded() -> None:
    provider = yp.YFinanceProvider()
    assert 1 <= provider.batch_size <= 200


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(ValueError):
        yp.YFinanceProvider(batch_size=0)
