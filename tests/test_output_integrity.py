"""A verifier that never hashes what it evaluates is checking the wrong files.

The manifest records a sha256 for every *input* snapshot, and since the last
round the verifier recomputes them. But the two artifacts it actually reads --
``wf_scored.parquet`` and ``execution_prices.parquet`` at the baseline root --
were never hashed by anything. The manifest had no ``outputs`` key at all.

That is not theoretical. Replacing ``wf_scored.parquet`` with a forgery whose
scores are nudged 12% of the way toward the realised forward return prints::

    cohort     CAGR  98.34%   (the real baseline: ~20%)
    rank-hold  CAGR 116.70%
    All 12 gates passed.

Every gate is honest about what it measures; none of them measured whether the
scores came out of the model. ``specs.md:334`` requires output hashes in the
manifest, and they were the one class of hash missing.

Two mechanisms close it, and they are not the same strength:

* ``wf_scored.parquet`` can only be checked against a hash recorded when it was
  written. Nothing else in the baseline can re-derive a model's scores.
* ``execution_prices.parquet`` is better off: the root file is the wide pivot of
  ``snapshot/execution_prices.parquet``, which *is* hashed. Checking the
  derivation ties the output to an input hash rather than to a promise, and
  works on baselines built before output hashing existed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_baseline import (  # noqa: E402
    gate_execution_derivation,
    gate_output_hashes,
)

from stock_predictor import repro  # noqa: E402

DATES = pd.bdate_range("2024-01-01", periods=12)
TICKERS = ["AAA", "BBB", "CCC"]


def _scored() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"date": d, "ticker": t, "prob": float(rng.random()), "adj_close": 100.0 + i}
        for d in DATES for i, t in enumerate(TICKERS)
    ])


def _long() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "ticker": t, "close": 100.0 + i + di, "volume": 1e6}
        for di, d in enumerate(DATES) for i, t in enumerate(TICKERS)
    ])


def _wide(long: pd.DataFrame) -> pd.DataFrame:
    return long.pivot_table(index="date", columns="ticker",
                            values="close", aggfunc="first").sort_index()


def _baseline(tmp_path: Path, *, record_outputs: bool = True,
              extra_empty: int = 0) -> Path:
    """A miniature baseline: hashed snapshot, plus the two root outputs."""
    snap = tmp_path / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    long = _long()
    long.to_parquet(snap / "execution_prices.parquet", index=False)

    wide = _wide(long)
    for j in range(extra_empty):
        wide[f"EMPTY{j}"] = np.nan
    wide.to_parquet(tmp_path / "execution_prices.parquet")

    scored = _scored()
    scored.to_parquet(tmp_path / "wf_scored.parquet", index=False)

    manifest: dict = {
        "run_id": "test",
        "snapshots": {
            "execution_prices": {
                "sha256": hashlib.sha256(
                    (snap / "execution_prices.parquet").read_bytes()).hexdigest(),
                "rows": len(long),
            },
        },
    }
    if record_outputs:
        manifest["outputs"] = {
            name: {
                "path": str(tmp_path / f"{name}.parquet"),
                "sha256": hashlib.sha256(
                    (tmp_path / f"{name}.parquet").read_bytes()).hexdigest(),
                "provenance": "recorded-at-write",
            }
            for name in ("wf_scored", "execution_prices")
        }
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return tmp_path


# ---------------------------------------------------------------------------
# The recorded-hash gate
# ---------------------------------------------------------------------------


def test_untouched_outputs_pass(tmp_path: Path) -> None:
    assert gate_output_hashes(_baseline(tmp_path)).passed


def test_a_forged_score_file_is_caught(tmp_path: Path) -> None:
    """The whole point. Swapping the scores must not verify clean."""
    d = _baseline(tmp_path)
    forged = _scored()
    forged["prob"] = 0.99
    forged.to_parquet(d / "wf_scored.parquet", index=False)

    g = gate_output_hashes(d)
    assert not g.passed
    assert any("wf_scored" in n for n in g.notes)


def test_a_forged_execution_panel_is_caught(tmp_path: Path) -> None:
    d = _baseline(tmp_path)
    px = pd.read_parquet(d / "execution_prices.parquet")
    px.iloc[0, 0] = 1.0
    px.to_parquet(d / "execution_prices.parquet")
    assert not gate_output_hashes(d).passed


def test_a_baseline_that_records_no_outputs_fails(tmp_path: Path) -> None:
    """Absence is not innocence: an unhashed output cannot be vouched for,
    and passing it silently is how this went unnoticed."""
    g = gate_output_hashes(_baseline(tmp_path, record_outputs=False))
    assert not g.passed
    assert any("no output hashes" in n.lower() for n in g.notes)


def test_a_missing_output_file_fails(tmp_path: Path) -> None:
    d = _baseline(tmp_path)
    (d / "wf_scored.parquet").unlink()
    assert not gate_output_hashes(d).passed


def test_sealed_provenance_passes_but_says_so(tmp_path: Path) -> None:
    """A baseline sealed after the fact is verifiable going forward but proves
    nothing about where the bytes came from. The gate must not blur the two."""
    d = _baseline(tmp_path)
    man = json.loads((d / "snapshot" / "manifest.json").read_text())
    for meta in man["outputs"].values():
        meta["provenance"] = "sealed-after-the-fact"
    (d / "snapshot" / "manifest.json").write_text(json.dumps(man))

    g = gate_output_hashes(d)
    assert g.passed
    assert any("sealed" in n.lower() for n in g.notes)


# ---------------------------------------------------------------------------
# The derivation gate: stronger, and retroactive
# ---------------------------------------------------------------------------


def test_the_execution_panel_derives_from_the_hashed_snapshot(tmp_path: Path) -> None:
    assert gate_execution_derivation(_baseline(tmp_path)).passed


def test_columns_with_no_data_do_not_break_the_derivation(tmp_path: Path) -> None:
    """The real panel carries 70 all-empty columns -- names the vendor never
    served. ``pivot_table`` drops them, so they are absent from the snapshot
    and present in the output, which is not a discrepancy."""
    assert gate_execution_derivation(_baseline(tmp_path, extra_empty=70)).passed


def test_a_tampered_price_breaks_the_derivation(tmp_path: Path) -> None:
    """This catches a forged execution panel with no recorded output hash at
    all, which is what makes it worth having on top of the hash gate."""
    d = _baseline(tmp_path, record_outputs=False)
    px = pd.read_parquet(d / "execution_prices.parquet")
    px.iloc[3, 1] = px.iloc[3, 1] * 1.5
    px.to_parquet(d / "execution_prices.parquet")
    assert not gate_execution_derivation(d).passed


def test_an_invented_column_breaks_the_derivation(tmp_path: Path) -> None:
    """A column carrying data that the snapshot never held is fabricated."""
    d = _baseline(tmp_path)
    px = pd.read_parquet(d / "execution_prices.parquet")
    px["ZZZ"] = 42.0
    px.to_parquet(d / "execution_prices.parquet")
    assert not gate_execution_derivation(d).passed


def test_a_dropped_session_breaks_the_derivation(tmp_path: Path) -> None:
    d = _baseline(tmp_path)
    px = pd.read_parquet(d / "execution_prices.parquet")
    px.iloc[:-1].to_parquet(d / "execution_prices.parquet")
    assert not gate_execution_derivation(d).passed


def test_a_snapshot_without_execution_prices_is_reported_not_assumed(
    tmp_path: Path,
) -> None:
    d = _baseline(tmp_path)
    (d / "snapshot" / "execution_prices.parquet").unlink()
    g = gate_execution_derivation(d)
    assert not g.passed


# ---------------------------------------------------------------------------
# The producer side
# ---------------------------------------------------------------------------


def test_register_output_records_a_hash(tmp_path: Path) -> None:
    p = tmp_path / "wf_scored.parquet"
    _scored().to_parquet(p, index=False)
    man: dict = {"snapshots": {}}
    repro.register_output(man, "wf_scored", p)

    meta = man["outputs"]["wf_scored"]
    assert meta["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert meta["provenance"] == "recorded-at-write"
    assert meta["bytes"] == p.stat().st_size


def test_register_output_leaves_snapshots_alone(tmp_path: Path) -> None:
    """Inputs and outputs are different claims and live in different places."""
    p = tmp_path / "wf_scored.parquet"
    _scored().to_parquet(p, index=False)
    man: dict = {"snapshots": {"stints": {"sha256": "x"}}}
    repro.register_output(man, "wf_scored", p)
    assert man["snapshots"] == {"stints": {"sha256": "x"}}
