"""Rebuild a run from its snapshot instead of from the network.

The pipeline wrote hashed snapshots of everything it downloaded and could not
read one back. So "rerun the baseline" meant "download again and hope", and it
did not hold: four rebuilds from one commit, one pinned window and one seed
produced cohort CAGRs spanning 17.20% to 23.12%. The panels agreed to 2e-6 --
float noise in the vendor's adjustment arithmetic, not revised data -- and
LightGBM splits flip on near-ties, so a different fifteen names got held.

That made every comparison in this project a comparison of two draws. A fix
worth two points of CAGR could not be told from noise worth four.

Replay closes it. A :class:`SnapshotProvider` serves the recorded prices and
macro series in place of the vendors, membership comes from the recorded
stints, and the sector map from the recorded copy. Feature engineering,
labelling, training and the walk-forward all re-run as normal — this replaces
the *inputs*, not the pipeline, so a code change still shows up.

Verify before use, always: a snapshot whose bytes no longer match the manifest
is not the run it claims to be.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SNAPSHOT_DIRNAME = "snapshot"

REQUIRED = ("equity_prices_long", "stints")
"""Without these there is nothing to replay."""

OPTIONAL = ("macro", "sector_map", "execution_prices", "benchmark")
"""Recorded when the run used them. A snapshot without ``macro`` predates
macro capture and cannot be replayed exactly; the caller is told rather than
left to wonder why the numbers moved."""


class SnapshotIncomplete(RuntimeError):
    """The snapshot cannot reproduce the run it came from."""


def snapshot_dir(baseline_dir: Path | str) -> Path:
    return Path(baseline_dir) / SNAPSHOT_DIRNAME


def read_manifest(baseline_dir: Path | str) -> dict:
    path = snapshot_dir(baseline_dir) / "manifest.json"
    if not path.exists():
        raise SnapshotIncomplete(f"no manifest at {path}")
    return json.loads(path.read_text())


def verify(baseline_dir: Path | str) -> dict[str, str]:
    """Recompute every recorded artifact's hash. Raises on any mismatch.

    Replaying from bytes that no longer match the manifest would reproduce
    something, just not the run that was recorded.
    """
    man = read_manifest(baseline_dir)
    snaps = man.get("snapshots", {})
    if not snaps:
        raise SnapshotIncomplete("manifest records no snapshots")

    out: dict[str, str] = {}
    for name, meta in sorted(snaps.items()):
        path = snapshot_dir(baseline_dir) / f"{name}.parquet"
        if not path.exists():
            raise SnapshotIncomplete(f"{name}: recorded but missing from disk")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(meta.get("sha256", "")):
            raise SnapshotIncomplete(
                f"{name}: sha256 {actual[:16]} does not match the recorded "
                f"{str(meta.get('sha256', ''))[:16]}"
            )
        out[name] = actual
    for name in REQUIRED:
        if name not in snaps:
            raise SnapshotIncomplete(f"snapshot has no {name}; cannot replay")
    return out


def _read(baseline_dir: Path | str, name: str) -> pd.DataFrame | None:
    path = snapshot_dir(baseline_dir) / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_stints(baseline_dir: Path | str) -> pd.DataFrame:
    st = _read(baseline_dir, "stints")
    if st is None:
        raise SnapshotIncomplete("snapshot has no stints")
    return st


def load_sector_map(baseline_dir: Path | str) -> pd.DataFrame | None:
    return _read(baseline_dir, "sector_map")


def load_macro(baseline_dir: Path | str) -> pd.DataFrame | None:
    return _read(baseline_dir, "macro")


def recorded_recycled_symbols(baseline_dir: Path | str) -> list[str] | None:
    """The reused symbols the *source* run detected, or ``None`` if unrecorded.

    Detection reads the evidence -- a departed ticker whose prices resume after
    a long dead period -- and cleaning removes that evidence. A snapshot is
    already cleaned, so re-running detection over it finds nothing: the real
    baseline records 46 and re-detection from its own snapshot finds 0. Without
    carrying the recorded list forward, a replay overwrites the manifest with
    an empty one and the survivorship gate loses the classification it needs to
    tell "another issuer holds this symbol now" from "nobody ever fetched it".
    """
    try:
        man = read_manifest(baseline_dir)
    except SnapshotIncomplete:
        return None
    val = man.get("recycled_symbols")
    return list(val) if val is not None else None


def missing_for_exact_replay(baseline_dir: Path | str) -> list[str]:
    """Optional inputs this snapshot lacks, and so cannot reproduce."""
    return [n for n in OPTIONAL
            if not (snapshot_dir(baseline_dir) / f"{n}.parquet").exists()]


class SnapshotProvider:
    """A :class:`~stock_predictor.data_provider.DataProvider` backed by files.

    Serves exactly what was recorded and nothing else. A ticker or date the
    snapshot does not hold comes back absent rather than being fetched, because
    silently reaching for the network is the behaviour this exists to remove.
    """

    def __init__(self, baseline_dir: Path | str):
        self.baseline_dir = Path(baseline_dir)
        long_px = _read(baseline_dir, "equity_prices_long")
        if long_px is None:
            raise SnapshotIncomplete("snapshot has no equity_prices_long")
        px = long_px.copy()
        px["date"] = pd.to_datetime(px["date"])
        value_col = "close" if "close" in px.columns else "adj_close"
        self._adj = px.pivot_table(index="date", columns="ticker",
                                   values=value_col, aggfunc="first").sort_index()
        self._vol = px.pivot_table(index="date", columns="ticker",
                                   values="volume", aggfunc="first").sort_index()
        self._macro = load_macro(baseline_dir)
        self._bench = _read(baseline_dir, "benchmark")

    def _window(self, frame: pd.DataFrame, tickers: list[str],
                start: str, end: str | None) -> pd.DataFrame:
        idx = frame.index
        lo = pd.Timestamp(start) if start else idx.min()
        hi = pd.Timestamp(end) if end else idx.max()
        cols = [t for t in tickers if t in frame.columns]
        return frame.loc[(idx >= lo) & (idx <= hi), cols]

    def download_equity_ohlcv(self, tickers, start, end):
        names = [str(t) for t in tickers]
        return (self._window(self._adj, names, start, end),
                self._window(self._vol, names, start, end))

    def download_macro(self, start, end):
        if self._macro is None:
            raise SnapshotIncomplete(
                "snapshot has no macro series; this run cannot be replayed "
                "exactly. Rebuild to record it."
            )
        m = self._macro.copy()
        m["date"] = pd.to_datetime(m["date"])
        lo = pd.Timestamp(start) if start else m["date"].min()
        hi = pd.Timestamp(end) if end else m["date"].max()
        return m[(m["date"] >= lo) & (m["date"] <= hi)].reset_index(drop=True)

    def download_benchmark(self, ticker: str, start: str, end: str):
        """The recorded benchmark series.

        Every published beta, alpha and HAC t was measured against whatever the
        vendor returned for SPY at report time -- the last of the five external
        inputs still fetched mid-report, and the reason the verifier could not
        check any of those figures offline. A recorded ``benchmark.parquet``
        takes precedence; the price panel is a fallback for the rare snapshot
        that carries the ticker as an ordinary column.

        A snapshot holding neither raises rather than quietly fetching, because
        silently reaching for the network is the behaviour replay removes.
        """
        if self._bench is not None:
            b = self._bench.copy()
            b["date"] = pd.to_datetime(b["date"])
            if "ticker" in b.columns:
                b = b[b["ticker"].astype(str) == str(ticker)]
            if len(b):
                s = pd.Series(b["close"].to_numpy(dtype=float),
                              index=pd.DatetimeIndex(b["date"]), name=ticker)
                s = s.sort_index()
                lo = pd.Timestamp(start) if start else s.index.min()
                hi = pd.Timestamp(end) if end else s.index.max()
                return s.loc[(s.index >= lo) & (s.index <= hi)]
        if ticker not in self._adj.columns:
            raise SnapshotIncomplete(
                f"snapshot does not carry benchmark {ticker}"
            )
        s = self._adj[ticker].dropna()
        lo = pd.Timestamp(start) if start else s.index.min()
        hi = pd.Timestamp(end) if end else s.index.max()
        return s.loc[(s.index >= lo) & (s.index <= hi)].rename(ticker)
