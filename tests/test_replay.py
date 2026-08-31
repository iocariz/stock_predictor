"""Replaying a run must use the run's inputs, or refuse.

The pipeline wrote hashed snapshots of everything it downloaded and could not
read one back, so "rerun the baseline" meant "download again and hope". It did
not hold: four rebuilds from one commit, one pinned window and one seed gave
cohort CAGRs from 17.20% to 23.12%, on panels agreeing to 2e-6. Every
comparison in the project was a comparison of two draws.

These pin the contract that makes replay worth having: serve exactly what was
recorded, verify it first, and refuse rather than quietly reach for the network
when something is missing. A replay that silently fell back to a live vendor
would be the original problem wearing a new name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from stock_predictor.replay import (
    SnapshotIncomplete,
    SnapshotProvider,
    load_stints,
    missing_for_exact_replay,
    verify,
)

DATES = pd.bdate_range("2024-01-01", periods=30)


def _long_prices() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "ticker": t, "close": 100.0 + i + di, "volume": 1e6}
        for di, d in enumerate(DATES) for i, t in enumerate(["AAA", "BBB"])
    ])


def _stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "start_date": [pd.Timestamp("2010-01-01")] * 2,
        "end_date": [pd.NaT, pd.NaT],
    })


def _macro() -> pd.DataFrame:
    return pd.DataFrame({"date": DATES, "vix": 15.0,
                         "tnx_yield": 3.0, "irx_yield": 4.5})


def _build(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": "t", "git_commit": "abc", "snapshots": {}}
    for name, df in frames.items():
        path = snap / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["snapshots"][name] = {
            "rows": len(df), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (snap / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def _complete(tmp_path: Path) -> Path:
    return _build(tmp_path, {
        "equity_prices_long": _long_prices(),
        "stints": _stints(),
        "macro": _macro(),
        "sector_map": pd.DataFrame({"ticker": ["AAA", "BBB"],
                                    "sector": ["Tech", "Health"]}),
        "execution_prices": _long_prices(),
    })


# ---------------------------------------------------------------------------
# Verification comes first
# ---------------------------------------------------------------------------


def test_a_complete_snapshot_verifies(tmp_path: Path) -> None:
    assert len(verify(_complete(tmp_path))) == 5


def test_tampered_bytes_are_refused(tmp_path: Path) -> None:
    """Replaying bytes that no longer match reproduces something, just not the
    run that was recorded."""
    d = _complete(tmp_path)
    bad = _long_prices()
    bad.loc[0, "close"] = 1.0
    bad.to_parquet(d / "snapshot" / "equity_prices_long.parquet", index=False)
    with pytest.raises(SnapshotIncomplete, match="sha256"):
        verify(d)


def test_a_snapshot_without_prices_cannot_replay(tmp_path: Path) -> None:
    d = _build(tmp_path, {"stints": _stints()})
    with pytest.raises(SnapshotIncomplete, match="equity_prices_long"):
        verify(d)


def test_missing_optional_inputs_are_named(tmp_path: Path) -> None:
    """A snapshot predating macro capture can still be replayed, but not
    exactly, and the caller is told which inputs are absent."""
    d = _build(tmp_path, {"equity_prices_long": _long_prices(),
                          "stints": _stints()})
    assert set(missing_for_exact_replay(d)) == {"macro", "sector_map",
                                                "execution_prices"}


def test_a_complete_snapshot_is_missing_nothing(tmp_path: Path) -> None:
    assert missing_for_exact_replay(_complete(tmp_path)) == []


# ---------------------------------------------------------------------------
# The provider serves the snapshot and only the snapshot
# ---------------------------------------------------------------------------


def test_prices_come_back_as_the_pipeline_expects(tmp_path: Path) -> None:
    p = SnapshotProvider(_complete(tmp_path))
    adj, vol = p.download_equity_ohlcv(["AAA", "BBB"], "2024-01-01", "2024-02-09")
    assert list(adj.columns) == ["AAA", "BBB"]
    assert isinstance(adj.index, pd.DatetimeIndex)
    assert adj.shape == vol.shape


def test_a_ticker_not_in_the_snapshot_is_absent_not_fetched(tmp_path: Path) -> None:
    """Silently reaching for the network is the behaviour replay removes."""
    p = SnapshotProvider(_complete(tmp_path))
    adj, _ = p.download_equity_ohlcv(["AAA", "NOPE"], "2024-01-01", None)
    assert list(adj.columns) == ["AAA"]


def test_the_window_is_respected(tmp_path: Path) -> None:
    p = SnapshotProvider(_complete(tmp_path))
    adj, _ = p.download_equity_ohlcv(["AAA"], "2024-01-15", "2024-01-31")
    assert adj.index.min() >= pd.Timestamp("2024-01-15")
    assert adj.index.max() <= pd.Timestamp("2024-01-31")


def test_macro_is_served_from_the_snapshot(tmp_path: Path) -> None:
    m = SnapshotProvider(_complete(tmp_path)).download_macro("2024-01-01", None)
    assert {"date", "vix", "tnx_yield", "irx_yield"} <= set(m.columns)


def test_a_snapshot_without_macro_refuses_rather_than_downloads(tmp_path: Path) -> None:
    d = _build(tmp_path, {"equity_prices_long": _long_prices(),
                          "stints": _stints()})
    with pytest.raises(SnapshotIncomplete, match="macro"):
        SnapshotProvider(d).download_macro("2024-01-01", None)


def test_a_benchmark_the_snapshot_lacks_refuses(tmp_path: Path) -> None:
    with pytest.raises(SnapshotIncomplete, match="SPY"):
        SnapshotProvider(_complete(tmp_path)).download_benchmark(
            "SPY", "2024-01-01", "2024-02-09")


def test_stints_come_from_the_snapshot(tmp_path: Path) -> None:
    st = load_stints(_complete(tmp_path))
    assert set(st["ticker"]) == {"AAA", "BBB"}


def test_serving_is_repeatable(tmp_path: Path) -> None:
    """Two reads of one snapshot must be identical, or replay proves nothing."""
    d = _complete(tmp_path)
    a, _ = SnapshotProvider(d).download_equity_ohlcv(["AAA", "BBB"], "2024-01-01", None)
    b, _ = SnapshotProvider(d).download_equity_ohlcv(["AAA", "BBB"], "2024-01-01", None)
    pd.testing.assert_frame_equal(a, b)
