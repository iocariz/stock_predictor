from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_predictor import repro


def test_sha256_file_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello")
    h1 = repro.sha256_file(p)
    assert len(h1) == 64
    p.write_bytes(b"world")
    h2 = repro.sha256_file(p)
    assert h1 != h2


def test_snapshot_parquet_meta(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "t.parquet"
    meta = repro.snapshot_parquet(df, path)
    assert meta["rows"] == 2
    assert meta["sha256"] == repro.sha256_file(path)
    pd.testing.assert_frame_equal(df, pd.read_parquet(path))


def test_manifest_json(tmp_path: Path) -> None:
    m = repro.build_base_manifest(run_id="test_run", argv=["python", "-m", "x"], extra={"k": 1})
    repro.register_snapshot(m, "s1", {"path": "p", "sha256": "abc", "rows": 0, "columns": []})
    repro.write_manifest(tmp_path / "m.json", m)
    loaded = json.loads((tmp_path / "m.json").read_text())
    assert loaded["run_id"] == "test_run"
    assert "s1" in loaded["snapshots"]


def test_wide_prices_to_long() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    adj = pd.DataFrame({"A": [1.0, 1.1], "B": [2.0, 2.0]}, index=idx)
    vol = pd.DataFrame({"A": [100, 110], "B": [200, 200]}, index=idx)
    long = repro.wide_prices_to_long(adj, vol)
    assert len(long) == 4
    assert set(long.columns) == {"date", "ticker", "close", "volume"}
