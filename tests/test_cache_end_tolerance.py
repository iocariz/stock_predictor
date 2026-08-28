"""A cache that expires every midnight is not a cache.

The coverage check added with the manifest was right in principle -- a file
fetched for 2024 must not answer a request for 2010 -- but it compared the
requested *end* date exactly. Requests default to "today", so every cached
ticker went stale the moment the date rolled over, and the next run tried to
refetch all of them.

Against a rate-limited vendor that is not a slow cache, it is a silent data
loss. It happened: a baseline rebuild on 2026-08-27 invalidated files whose
manifest said ``end: 2026-08-26``, hit Tiingo's daily limit 83 tickers in, and
produced a panel missing **24 delisted names** -- MRO, HES, JNPR, JWN, K, KSU,
IPG, HOLX and others. Those are exactly the acquisitions that make the panel
survivorship-free, so the run "succeeded" while quietly reintroducing the bias
the hybrid provider exists to remove.

The start date stays strict, because missing history is missing data. The end
date gets a tolerance: the vendor cannot serve sessions that have not happened,
and for a training panel a few days of tail is immaterial.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.providers.hybrid_provider import HybridProvider


def _bars(start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"date": idx, "close": 1.0, "volume": 100.0})


def _provider(tmp_path, responses):
    p = HybridProvider(tiingo_api_key="x", cache_dir=tmp_path)
    calls = []

    def fake(ticker, start, end):
        calls.append((ticker, start, end))
        return responses.get(ticker, pd.DataFrame())

    p._fetch_one = fake                                    # type: ignore[method-assign]
    return p, calls


def _seed(tmp_path, ticker="MRO", start="2010-01-01", end="2026-08-26"):
    p, _ = _provider(tmp_path, {ticker: _bars(start, end)})
    p.fetch_missing([ticker], start, end)
    return p


# ---------------------------------------------------------------------------


def test_the_next_day_does_not_invalidate_the_cache(tmp_path) -> None:
    """The exact failure: cached through 2026-08-26, asked for 2026-08-27."""
    _seed(tmp_path)
    p2, calls = _provider(tmp_path, {})
    out = p2.fetch_missing(["MRO"], "2010-01-01", "2026-08-27")
    assert calls == [], "one extra day must not trigger a refetch"
    assert not out["MRO"].empty


def test_a_delisted_name_survives_a_later_request(tmp_path) -> None:
    """MRO stopped trading in 2024, so its *data* can never reach 2026 -- but
    the pipeline asked for the full window and got everything that exists.

    Coverage is therefore about the range that was *requested*, not the range
    that came back. Judging by the data would refetch every delisted name on
    every run and burn the quota that recovering them depends on.
    """
    p, _ = _provider(tmp_path, {"MRO": _bars("2010-01-01", "2024-11-22")})
    p.fetch_missing(["MRO"], "2010-01-01", "2026-08-26")   # asked wide, got short

    p2, calls = _provider(tmp_path, {})
    out = p2.fetch_missing(["MRO"], "2010-01-01", "2026-08-27")
    assert calls == [], "the request was already covered; the data just ends"
    assert not out["MRO"].empty
    assert pd.to_datetime(out["MRO"]["date"]).max() == pd.Timestamp("2024-11-22")


def test_a_materially_staler_cache_is_refetched(tmp_path) -> None:
    """The tolerance is for a rolling 'today', not for a cache months behind."""
    _seed(tmp_path, "MRO", "2010-01-01", "2026-01-01")
    p2, calls = _provider(tmp_path, {"MRO": _bars("2010-01-01", "2026-08-27")})
    p2.fetch_missing(["MRO"], "2010-01-01", "2026-08-27")
    assert len(calls) == 1


def test_an_earlier_start_is_still_a_miss(tmp_path) -> None:
    """Missing history is missing data; the start date stays strict."""
    _seed(tmp_path, "MRO", "2020-01-01", "2026-08-26")
    p2, calls = _provider(tmp_path, {"MRO": _bars("2010-01-01", "2026-08-26")})
    p2.fetch_missing(["MRO"], "2010-01-01", "2026-08-26")
    assert len(calls) == 1


def test_the_tolerance_is_configurable(tmp_path) -> None:
    _seed(tmp_path, "MRO", "2010-01-01", "2026-08-01")
    p2, calls = _provider(tmp_path, {"MRO": _bars("2010-01-01", "2026-08-27")})
    p2.end_tolerance_days = 60
    p2.fetch_missing(["MRO"], "2010-01-01", "2026-08-27")
    assert calls == [], "inside a widened tolerance, still a hit"


@pytest.mark.parametrize("gap_days", [1, 2, 3])
def test_a_long_weekend_never_invalidates(tmp_path, gap_days: int) -> None:
    _seed(tmp_path, "MRO", "2010-01-01", "2026-08-26")
    asked = (pd.Timestamp("2026-08-26") + pd.Timedelta(days=gap_days)).date()
    p2, calls = _provider(tmp_path, {})
    p2.fetch_missing(["MRO"], "2010-01-01", str(asked))
    assert calls == []
