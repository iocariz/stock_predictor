"""yfinance for the bulk, Tiingo for the names it drops.

Survivorship is the largest un-fixed bias in this project: Yahoo stops
serving most companies once they are acquired, renamed or taken private, so
187 of 342 departed S&P members are simply absent from a 2010+ panel. Their
absence is not random — they are disproportionately the failures — so every
backtest built on Yahoo alone is flattered.

Tiingo keeps them. Spot-checked against the real corporate actions:

    ABMD  ends 2023-01-03  (J&J close)        ATVI  ends 2023-10-13  (Microsoft)
    AET   ends 2018-11-28  (CVS)              XLNX  ends 2022-02-14  (AMD)
    CERN  ends 2022-09-08  (Oracle)           CTXS  ends 2022-11-02  (take-private)

Tiingo's free tier is rate-limited per hour and per day, so this fetches only
what Yahoo could not supply and caches one parquet per ticker. A run that hits
the limit keeps what it recovered and the next run resumes rather than
restarting.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from stock_predictor.providers.yfinance_provider import (
    YFinanceProvider,
    _order_columns,
    _usable_columns,
)

TIINGO_PRICES = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
DEFAULT_CACHE = Path("artifacts/tiingo_cache")

CACHE_SCHEMA_VERSION = 1
"""Bump when the cached frame's columns or adjustment basis change, so old
files are refetched instead of silently mixing two schemas in one panel."""

MANIFEST_NAME = "_manifest.json"
"""Records the range each cached file was *asked* for, which is the only thing
that answers "does this file cover the new request?". The data's own range
cannot: a ticker that IPO'd in 2015 legitimately has no 2010 rows, and
comparing against requested dates alone would refetch it forever."""

DEFAULT_END_TOLERANCE_DAYS = 7
"""How far behind the requested end a cached file may sit and still be used.

Requests default to "today", so comparing the end date exactly expired every
cached ticker at midnight. Against a rate-limited vendor that is silent data
loss, not a slow cache: a rebuild on 2026-08-27 invalidated files recorded as
ending 2026-08-26, hit Tiingo's daily limit, and produced a panel missing 24
delisted names -- the acquisitions that make the panel survivorship-free.

The start date stays strict, because missing history is missing data. The end
gets slack, because the vendor cannot serve sessions that have not happened and
a few days of tail does not change a training panel."""

DEFAULT_EMPTY_TTL_DAYS = 7
"""How long a known miss is trusted. Caching misses stops a rate-limited run
from re-asking for names Tiingo does not have; caching them forever turns one
bad afternoon into a permanently absent ticker."""


class HybridProvider:
    """Yahoo first, Tiingo for the gaps.

    Only the equity path is hybrid; macro and benchmark come from the
    yfinance provider, which serves both reliably.
    """

    def __init__(
        self,
        *,
        tiingo_api_key: str,
        cache_dir: Path | None = None,
        batch_size: int = 100,
        max_retries: int = 4,
        backoff_s: float = 20.0,
        pause_s: float = 0.15,
        yf_provider: object | None = None,
    ) -> None:
        if not tiingo_api_key:
            raise ValueError("tiingo_api_key is required for the hybrid provider")
        self._key = tiingo_api_key
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.pause_s = pause_s
        self._yf = yf_provider or YFinanceProvider(batch_size=batch_size)
        self._session = None
        self.empty_ttl_days = DEFAULT_EMPTY_TTL_DAYS
        self.end_tolerance_days = DEFAULT_END_TOLERANCE_DAYS

    # -- Tiingo ------------------------------------------------------------

    def _get_session(self):
        if self._session is None:
            import requests

            s = requests.Session()
            s.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Token {self._key}",
            })
            self._session = s
        return self._session

    def _fetch_one(self, ticker: str, start: str, end: str | None) -> pd.DataFrame:
        """One ticker's adjusted OHLCV, or an empty frame. Never raises."""
        params = {"startDate": start}
        if end:
            params["endDate"] = end
        for attempt in range(self.max_retries):
            try:
                resp = self._get_session().get(
                    TIINGO_PRICES.format(ticker=ticker), params=params, timeout=30,
                )
            except Exception as exc:  # noqa: BLE001 - one symbol must not kill the run
                print(f"    Tiingo {ticker}: {type(exc).__name__}: {exc}")
                return pd.DataFrame()
            if resp.status_code == 200:
                rows = resp.json()
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
                # adjClose/adjVolume are split- and dividend-adjusted, matching
                # yfinance auto_adjust=True.
                return df[["date", "adjClose", "adjVolume"]].rename(
                    columns={"adjClose": "close", "adjVolume": "volume"},
                )
            if resp.status_code == 429:
                if attempt < self.max_retries - 1:
                    wait = self.backoff_s * 2**attempt
                    print(f"    Tiingo rate-limited on {ticker}; waiting {wait:.0f}s…")
                    time.sleep(wait)
                    continue
                raise TiingoRateLimited(ticker)
            return pd.DataFrame()
        return pd.DataFrame()

    # -- cache bookkeeping -------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.cache_dir / MANIFEST_NAME

    def _read_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}          # a corrupt manifest costs a refetch, not a crash

    def _write_manifest(self, manifest: dict) -> None:
        try:
            self._manifest_path().write_text(json.dumps(manifest, indent=2, default=str))
        except OSError:
            pass               # the cache is an optimisation; failing to note it is not fatal

    def _cache_is_usable(
        self, ticker: str, cached: Path, df: pd.DataFrame, start: str, end: str,
        manifest: dict,
    ) -> bool:
        """Does this cached file answer *this* request?

        The old key was the ticker alone, so the first call's range became the
        answer to every later one: ask for 2010-2026 after a 2024-only fetch
        and you silently got two years of history for a sixteen-year request.
        """
        entry = manifest.get(ticker)
        if entry is not None:
            if int(entry.get("schema", 0)) != CACHE_SCHEMA_VERSION:
                return False
            if entry.get("empty"):
                fetched = pd.to_datetime(entry.get("fetched"), errors="coerce", utc=True)
                if pd.isna(fetched):
                    return False
                age = datetime.now(timezone.utc) - fetched.to_pydatetime()
                return age < timedelta(days=self.empty_ttl_days)
            if str(entry.get("start", "9999")) > str(start):
                return False       # missing history is missing data
            cached_end = pd.to_datetime(entry.get("end"), errors="coerce")
            wanted_end = pd.to_datetime(end, errors="coerce")
            if pd.isna(cached_end) or pd.isna(wanted_end):
                return False
            if cached_end >= wanted_end:
                return True
            return (wanted_end - cached_end).days <= self.end_tolerance_days
        # Legacy file written before the manifest existed. Its own dates are a
        # conservative lower bound on what was requested, so a sub-range of the
        # data it holds is still safe to serve.
        if df.empty:
            return False
        dates = pd.to_datetime(df["date"])
        return dates.min() <= pd.Timestamp(start) and dates.max() >= pd.Timestamp(end)

    def fetch_missing(
        self, tickers: list[str], start: str, end: str | None,
    ) -> dict[str, pd.DataFrame]:
        """Cached per-ticker fetch. Stops cleanly when the rate limit is hit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        end_key = str(end) if end else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        manifest = self._read_manifest()
        out: dict[str, pd.DataFrame] = {}
        fetched = 0
        for i, t in enumerate(sorted(set(tickers)), 1):
            cached = self.cache_dir / f"{t}.parquet"
            if cached.exists():
                df = pd.read_parquet(cached)
                if self._cache_is_usable(t, cached, df, str(start), end_key, manifest):
                    if not df.empty:
                        out[t] = df
                    continue
            try:
                df = self._fetch_one(t, start, end)
            except TiingoRateLimited:
                print(f"  Tiingo daily/hourly limit reached after {fetched} new "
                      f"tickers ({i - 1}/{len(set(tickers))} considered). "
                      "Cached so far; re-run to resume.")
                break
            df.to_parquet(cached, index=False)  # cache misses too, to avoid re-asking
            manifest[t] = {
                "start": str(start),
                "end": end_key,
                "empty": bool(df.empty),
                "schema": CACHE_SCHEMA_VERSION,
                "fetched": datetime.now(timezone.utc).isoformat(),
            }
            self._write_manifest(manifest)
            fetched += 1
            if not df.empty:
                out[t] = df
            if self.pause_s:
                time.sleep(self.pause_s)
        if fetched:
            print(f"  Tiingo: fetched {fetched} new tickers, "
                  f"{len(out)} of {len(set(tickers))} recovered")
        return out

    # -- Provider protocol -------------------------------------------------

    def download_equity_ohlcv(
        self, tickers: list[str], start: str, end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        wanted = list(dict.fromkeys(tickers))
        adj_close, volume = self._yf.download_equity_ohlcv(wanted, start, end)
        have = _usable_columns(adj_close)
        missing = [t for t in wanted if t not in have]
        if not missing:
            return adj_close, volume

        print(f"  Yahoo missing {len(missing)} tickers; trying Tiingo…")
        recovered = self.fetch_missing(missing, start, end)
        if not recovered:
            return adj_close, volume

        close_add = pd.DataFrame({
            t: df.set_index("date")["close"] for t, df in recovered.items()
        })
        vol_add = pd.DataFrame({
            t: df.set_index("date")["volume"] for t, df in recovered.items()
        })
        # Yahoo hands back an all-NaN placeholder column for a name it cannot
        # serve. Those names are exactly the ones Tiingo just supplied, so the
        # placeholder must go before concatenating or the ticker ends up
        # duplicated — which then produces duplicate rows on stack().
        dead = [c for c in adj_close.columns if str(c) in recovered]
        if dead:
            adj_close = adj_close.drop(columns=dead)
            volume = volume.drop(columns=[c for c in volume.columns if str(c) in recovered],
                                 errors="ignore")
        # Tiingo carries session dates Yahoo never returned for these names, so
        # union the index rather than reindexing onto Yahoo's calendar.
        adj_close = pd.concat([adj_close, close_add], axis=1, sort=True).sort_index()
        volume = pd.concat([volume, vol_add], axis=1, sort=True).sort_index()
        assert not adj_close.columns.duplicated().any(), "duplicate ticker columns"
        return _order_columns(adj_close, wanted), _order_columns(volume, wanted)

    def download_macro(self, start: str, end: str | None) -> pd.DataFrame:
        return self._yf.download_macro(start, end)

    def download_benchmark(self, ticker: str, start: str, end: str) -> pd.Series:
        return self._yf.download_benchmark(ticker, start, end)


class TiingoRateLimited(RuntimeError):
    """Tiingo returned 429 after exhausting retries."""
