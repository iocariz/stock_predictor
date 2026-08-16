"""YFinance data provider (default backend)."""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from stock_predictor.training import wide_field

MACRO_YF = ["^VIX", "^TNX", "^IRX"]

# Yahoo intermittently rate-limits single-ticker downloads (empty frame or
# exception); retry with exponential backoff before giving up.
_BENCHMARK_RETRIES = 4
_BENCHMARK_BACKOFF_S = 2.0

# Equity downloads are chunked: one request for ~1000 symbols reliably trips
# Yahoo's rate limiter, which answers with a *partial* frame rather than an
# error. That partial reply used to become the traded universe silently.
DEFAULT_BATCH_SIZE = 100
DEFAULT_RETRY_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 2.0
DEFAULT_PAUSE_S = 0.2


def _empty_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(), pd.DataFrame()


def _split_fields(raw: pd.DataFrame, batch: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract (close, volume) wide frames for one downloaded batch."""
    if raw is None or raw.empty:
        return _empty_pair()
    close = wide_field(raw, "Close")
    volume = wide_field(raw, "Volume")
    # A one-symbol batch comes back without a ticker column level, so
    # wide_field yields the field name ("Close") as the column — rename it to
    # the ticker so downstream stacking sees a real symbol.
    if len(batch) == 1:
        if list(close.columns) == ["Close"]:
            close.columns = [batch[0]]
        if list(volume.columns) == ["Volume"]:
            volume.columns = [batch[0]]
    return close, volume


def _usable_columns(frame: pd.DataFrame) -> set[str]:
    """Columns that carry at least one real observation."""
    if frame is None or frame.empty:
        return set()
    return {str(c) for c in frame.columns[frame.notna().any(axis=0)]}


def _order_columns(frame: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    """Restore the caller's ticker order, keeping only symbols we actually got."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    present = [t for t in wanted if t in frame.columns]
    return frame.loc[:, present]


def _concat_wide(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Join per-batch wide frames on the union of their date indexes."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame()
    out = pd.concat(usable, axis=1).sort_index()
    # Later batches (the recovery pass) win on any overlapping symbol.
    return out.loc[:, ~out.columns.duplicated(keep="last")]


class YFinanceProvider:
    """Download equity, macro, and benchmark data via yfinance.

    Equity requests are split into batches of *batch_size*. Each batch is
    retried with exponential backoff, and symbols still missing after the
    first pass get a second pass in *retry_batch_size* chunks — so one bad
    or throttled batch costs only that batch, not the universe.

    Symbols Yahoo genuinely cannot serve are simply absent from the result.
    Deciding whether that is acceptable belongs to
    :func:`stock_predictor.universe.check_download_coverage`, not here.
    """

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        retry_batch_size: int = DEFAULT_RETRY_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        pause_s: float = DEFAULT_PAUSE_S,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if retry_batch_size < 1:
            raise ValueError(f"retry_batch_size must be >= 1, got {retry_batch_size}")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.batch_size = batch_size
        self.retry_batch_size = retry_batch_size
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.pause_s = pause_s

    # -- batching ---------------------------------------------------------

    def _download_batch(
        self, batch: list[str], start: str, end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One batch, retried with exponential backoff. Never raises."""
        for attempt in range(self.max_retries):
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    group_by="ticker",
                    threads=True,
                    auto_adjust=True,
                    progress=False,
                )
                close, volume = _split_fields(raw, batch)
                if not close.empty:
                    return close, volume
                reason = "empty frame"
            except Exception as exc:  # noqa: BLE001 - a bad batch must not kill the run
                reason = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries - 1:
                wait = self.backoff_s * 2**attempt
                print(
                    f"    batch of {len(batch)} failed ({reason}); "
                    f"retry {attempt + 2}/{self.max_retries} in {wait:.0f}s…"
                )
                time.sleep(wait)
            else:
                print(f"    batch of {len(batch)} failed after "
                      f"{self.max_retries} attempts ({reason}); continuing")
        return _empty_pair()

    def _run_pass(
        self, tickers: list[str], start: str, end: str | None, size: int, label: str,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        batches = [tickers[i : i + size] for i in range(0, len(tickers), size)]
        closes: list[pd.DataFrame] = []
        volumes: list[pd.DataFrame] = []
        for i, batch in enumerate(batches, 1):
            close, volume = self._download_batch(batch, start, end)
            if not close.empty:
                closes.append(close)
            if not volume.empty:
                volumes.append(volume)
            got = len(_usable_columns(close))
            print(f"  {label} {i}/{len(batches)}: {got}/{len(batch)} tickers")
            if self.pause_s > 0 and i < len(batches):
                time.sleep(self.pause_s)
        return closes, volumes

    def download_equity_ohlcv(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        wanted = list(dict.fromkeys(tickers))  # de-duplicate, keep order
        if not wanted:
            return _empty_pair()

        closes, volumes = self._run_pass(
            wanted, start, end, self.batch_size, "batch",
        )
        adj_close = _concat_wide(closes)
        volume = _concat_wide(volumes)

        # Second pass in smaller chunks for anything still missing: a whole
        # batch lost to throttling is usually recoverable, whereas a symbol
        # Yahoo has genuinely delisted is not.
        have = _usable_columns(adj_close)
        missing = [t for t in wanted if t not in have]
        if missing and self.retry_batch_size < self.batch_size:
            print(f"  Retrying {len(missing)} missing tickers in smaller batches…")
            more_c, more_v = self._run_pass(
                missing, start, end, self.retry_batch_size, "retry",
            )
            if more_c:
                adj_close = _concat_wide([adj_close, *more_c])
                volume = _concat_wide([volume, *more_v])

        return _order_columns(adj_close, wanted), _order_columns(volume, wanted)

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
            except Exception as exc:  # noqa: BLE001 - retried below, then degraded
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
