"""A verifier that reads live state does not verify a baseline.

The point of a baseline is that it is a fixed object: the same artifacts must
reach the same verdict tomorrow. The first verifier did not manage that.

* It loaded **current** index membership rather than ``snapshot/stints.parquet``
  — which the baseline already contains — so the verdict moved whenever the
  membership table or the rename map moved. The rename work alone changed 12
  stints; that would have silently re-scored every earlier baseline.
* Its survivorship gate classified names against the **live Tiingo cache**, a
  directory any run mutates.
* It printed the manifest's snapshot hashes without ever recomputing them, so a
  corrupted or swapped artifact passed.
* It *wrote* ``survivorship_gap.json`` into the baseline while verifying it.

So it could pass or fail an unchanged baseline depending on what had happened
elsewhere on the machine, and it could not detect the one thing a manifest of
hashes exists to detect.

These cover the integrity check and the snapshot-first loading. The gate
functions themselves are exercised end to end against the real baseline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_baseline import (  # noqa: E402
    load_snapshot_stints,
    verify_snapshot_hashes,
)


def _write_snapshot(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": "test", "git_commit": "abc", "snapshots": {}}
    for name, df in frames.items():
        path = snap / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["snapshots"][name] = {
            "path": str(path),
            "rows": len(df),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return snap


def _stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "start_date": [pd.Timestamp("2010-01-01")] * 2,
        "end_date": [pd.NaT, pd.Timestamp("2020-01-01")],
    })


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def test_intact_snapshots_verify(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, {"stints": _stints()})
    g = verify_snapshot_hashes(tmp_path)
    assert g.passed, g.report()


def test_a_tampered_artifact_is_caught(tmp_path: Path) -> None:
    """The whole reason a manifest records hashes."""
    snap = _write_snapshot(tmp_path, {"stints": _stints()})
    bad = _stints()
    bad.loc[0, "ticker"] = "ZZZ"
    bad.to_parquet(snap / "stints.parquet", index=False)

    g = verify_snapshot_hashes(tmp_path)
    assert not g.passed
    assert any("stints" in n for n in g.notes)


def test_a_missing_artifact_is_caught(tmp_path: Path) -> None:
    snap = _write_snapshot(tmp_path, {"stints": _stints()})
    (snap / "stints.parquet").unlink()
    assert not verify_snapshot_hashes(tmp_path).passed


def test_a_missing_manifest_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """Absent evidence must not read as evidence of integrity."""
    assert not verify_snapshot_hashes(tmp_path).passed


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


def test_stints_come_from_the_snapshot(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, {"stints": _stints()})
    st = load_snapshot_stints(tmp_path)
    assert list(st["ticker"]) == ["AAA", "BBB"]


def test_a_baseline_without_snapshot_stints_is_refused(tmp_path: Path) -> None:
    """Falling back to live membership would make the verdict depend on when
    it was run, which is exactly what this is meant to stop."""
    _write_snapshot(tmp_path, {})
    with pytest.raises(FileNotFoundError, match="stints"):
        load_snapshot_stints(tmp_path)


def test_verification_does_not_write_into_the_baseline(tmp_path: Path) -> None:
    """Reading an artifact must not modify it."""
    _write_snapshot(tmp_path, {"stints": _stints()})
    before = {p: p.stat().st_mtime_ns for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    verify_snapshot_hashes(tmp_path)
    load_snapshot_stints(tmp_path)
    after = {p: p.stat().st_mtime_ns for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert before == after
    assert not (tmp_path / "survivorship_gap.json").exists()
