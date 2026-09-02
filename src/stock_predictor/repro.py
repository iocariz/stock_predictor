"""Reproducible run manifests and on-disk data snapshots (hashed parquet)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(cwd: Path | None = None) -> dict[str, Any]:
    root = cwd or REPO_ROOT
    out: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        out["commit"] = commit
        dirty_proc = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        out["dirty"] = dirty_proc.returncode != 0
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def snapshot_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "bytes": st.st_size,
    }


def build_base_manifest(
    *,
    run_id: str,
    argv: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = git_revision()
    m: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "argv": argv,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("UV_", "PYTHON", "CI"))
        },
        "snapshots": {},
    }
    if extra:
        m["config"] = extra
    return m


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def register_snapshot(manifest: dict[str, Any], name: str, snapshot_meta: dict[str, Any]) -> None:
    manifest.setdefault("snapshots", {})[name] = snapshot_meta


def register_output(manifest: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    """Hash an artifact this run *produced*, under ``manifest["outputs"]``.

    Inputs were hashed from the start; outputs were not, and the verifier reads
    outputs. A forged ``wf_scored.parquet`` therefore passed every gate while
    reporting a 98% CAGR. ``specs.md:334`` asks for both, and they are separate
    claims: a snapshot says "this is what we downloaded", an output says "this
    is what we computed from it".

    ``provenance`` distinguishes a hash taken as the file was written from one
    added to an older baseline afterwards, which is verifiable going forward but
    says nothing about where the bytes came from.
    """
    meta = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "provenance": "recorded-at-write",
    }
    manifest.setdefault("outputs", {})[name] = meta
    return meta


def wide_prices_to_long(adj_close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Stack wide OHLCV panels into a long table for compact parquet snapshots."""
    ac = adj_close.stack(future_stack=True).rename("close").reset_index()
    ac.columns = ["date", "ticker", "close"]
    vol = volume.stack(future_stack=True).rename("volume").reset_index()
    vol.columns = ["date", "ticker", "volume"]
    return ac.merge(vol, on=["date", "ticker"], how="outer").sort_values(["ticker", "date"])
