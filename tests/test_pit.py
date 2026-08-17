from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.pit import filter_panel_to_pit, tickers_overlapping_window


def test_tickers_overlapping_window() -> None:
    stints = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-06-01", "2019-01-01"]),
            "end_date": pd.to_datetime(["2020-12-31", pd.NaT, "2020-06-01"]),
        }
    )
    out = tickers_overlapping_window(stints, "2020-06-01", "2021-01-01")
    assert "AAA" in out and "BBB" in out
    assert "CCC" not in out


def test_filter_panel_to_pit() -> None:
    stints = pd.DataFrame(
        {
            "ticker": ["X"],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date": pd.to_datetime(["2020-06-30"]),
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-01", "2020-07-01"]),
            "ticker": ["X", "X"],
            "v": [1.0, 2.0],
        }
    )
    got = filter_panel_to_pit(panel, stints)
    assert len(got) == 1
    assert got["date"].iloc[0] == pd.Timestamp("2020-03-01")


# ---------------------------------------------------------------------------
# Stints caching: every training run needs this file, and the host rate-limits
# ---------------------------------------------------------------------------

_CSV = "ticker,start_date,end_date\nAAA,2020-01-01,\nBBB,2019-01-01,2021-06-30\n"


def _patch_fetch(monkeypatch, *, body=None, exc=None, counter=None):
    from stock_predictor import pit

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return (body or _CSV).encode()

    def fake(req, timeout=None):
        if counter is not None:
            counter.append(1)
        if exc is not None:
            raise exc
        return _Resp()

    monkeypatch.setattr(pit.urllib.request, "urlopen", fake)
    monkeypatch.setattr(pit.time, "sleep", lambda s: None)


def test_stints_are_cached_and_reused(tmp_path, monkeypatch) -> None:
    from stock_predictor.pit import load_sp500_stints

    cache = tmp_path / "stints.csv"
    calls: list[int] = []
    _patch_fetch(monkeypatch, counter=calls)
    a = load_sp500_stints(cache_path=cache)
    b = load_sp500_stints(cache_path=cache)
    assert cache.exists()
    assert len(calls) == 1, "a fresh cache must not re-fetch"
    pd.testing.assert_frame_equal(a, b)


def test_stale_cache_is_refreshed(tmp_path, monkeypatch) -> None:
    import os

    from stock_predictor.pit import load_sp500_stints

    cache = tmp_path / "stints.csv"
    calls: list[int] = []
    _patch_fetch(monkeypatch, counter=calls)
    load_sp500_stints(cache_path=cache)
    os.utime(cache, (0, 0))  # ancient
    load_sp500_stints(cache_path=cache)
    assert len(calls) == 2


def test_rate_limited_fetch_falls_back_to_a_stale_cache(tmp_path, monkeypatch) -> None:
    """Regression: repeated runs earned HTTP 429 from the upstream host and
    every run then died before it started."""
    import os
    import urllib.error

    from stock_predictor.pit import load_sp500_stints

    cache = tmp_path / "stints.csv"
    _patch_fetch(monkeypatch)
    load_sp500_stints(cache_path=cache)
    os.utime(cache, (0, 0))

    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
    _patch_fetch(monkeypatch, exc=err)
    out = load_sp500_stints(cache_path=cache, retries=2)
    assert list(out["ticker"]) == ["AAA", "BBB"], "stale data beats no run"


def test_fetch_failure_without_a_cache_still_raises(tmp_path, monkeypatch) -> None:
    import urllib.error

    from stock_predictor.pit import load_sp500_stints

    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
    _patch_fetch(monkeypatch, exc=err)
    with pytest.raises(urllib.error.HTTPError):
        load_sp500_stints(cache_path=tmp_path / "absent.csv", retries=2)


def test_retries_before_giving_up(tmp_path, monkeypatch) -> None:
    import urllib.error

    from stock_predictor.pit import load_sp500_stints

    calls: list[int] = []
    err = urllib.error.HTTPError("u", 429, "rate", {}, None)
    _patch_fetch(monkeypatch, exc=err, counter=calls)
    with pytest.raises(urllib.error.HTTPError):
        load_sp500_stints(cache_path=tmp_path / "absent.csv", retries=3)
    assert len(calls) == 3
