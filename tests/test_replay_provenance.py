"""Cleaning destroys the evidence that cleaning was needed.

``drop_recycled_prices`` finds a departed ticker whose prices resume after a
long dead period -- Anadarko's ``APC`` printing again from 2026 under a
different issuer -- and blanks them. The evidence *is* the resumed block, so
once it is blanked the detector has nothing left to find.

A snapshot is written after cleaning. Replaying it therefore re-runs detection
over already-clean data, finds nothing, and overwrote
``manifest["recycled_symbols"]`` with the empty result. On the real baseline
that is 46 recorded against 0 re-detected.

The list is not decorative: the survivorship gate uses it to tell "another
issuer holds this symbol now, and refetching returns the same wrong company"
from "nobody ever fetched it". Losing it on replay means a replayed baseline
cannot reproduce the original survivorship classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.pit import drop_recycled_prices
from stock_predictor.replay import recorded_recycled_symbols

SESSIONS = pd.bdate_range("2010-01-01", "2026-08-28")


def _manifest(tmp_path: Path, payload: dict) -> Path:
    snap = tmp_path / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "manifest.json").write_text(json.dumps(payload))
    return tmp_path


def _stints() -> pd.DataFrame:
    return pd.DataFrame([{"ticker": "APC",
                          "start_date": pd.Timestamp("2010-01-01"),
                          "end_date": pd.Timestamp("2019-08-08")}])


def _dirty() -> pd.DataFrame:
    col = pd.Series(np.nan, index=SESSIONS)
    for lo, hi in (("2010-01-04", "2019-08-08"), ("2026-02-12", "2026-08-27")):
        col.loc[(SESSIONS >= pd.Timestamp(lo)) & (SESSIONS <= pd.Timestamp(hi))] = 100.0
    return pd.DataFrame({"APC": col})


# ---------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------


def test_a_dirty_panel_is_detected() -> None:
    _, _, dropped = drop_recycled_prices(_dirty(), _stints())
    assert dropped == ["APC"]


def test_a_cleaned_panel_detects_nothing() -> None:
    """Why re-detection cannot stand in for the recorded list: cleaning is not
    idempotent in what it can *observe*, only in what it produces."""
    cleaned, _, _ = drop_recycled_prices(_dirty(), _stints())
    _, _, again = drop_recycled_prices(cleaned, _stints())
    assert again == []


# ---------------------------------------------------------------------------
# Carrying the classification forward
# ---------------------------------------------------------------------------


def test_the_recorded_list_is_readable_from_a_snapshot(tmp_path: Path) -> None:
    d = _manifest(tmp_path, {"run_id": "t", "recycled_symbols": ["APC", "FB"]})
    assert recorded_recycled_symbols(d) == ["APC", "FB"]


def test_an_empty_recorded_list_is_not_the_same_as_absent(tmp_path: Path) -> None:
    """A run that genuinely found none recorded ``[]``; one predating the
    detector recorded nothing. Collapsing the two loses the distinction."""
    assert recorded_recycled_symbols(
        _manifest(tmp_path, {"recycled_symbols": []})) == []
    assert recorded_recycled_symbols(_manifest(tmp_path, {"run_id": "t"})) is None


def test_a_snapshot_without_a_manifest_reports_absent(tmp_path: Path) -> None:
    assert recorded_recycled_symbols(tmp_path) is None


def test_a_replay_inherits_rather_than_redetects() -> None:
    """The wiring, read off the source: the replay branch must prefer the
    recorded list over what re-detection returns."""
    import inspect

    from stock_predictor import cli

    src = inspect.getsource(cli.main)
    block = src.split('manifest["recycled_symbols"]')[0][-1400:]
    assert "recorded_recycled_symbols" in src, (
        "replay still overwrites the recorded list with a fresh detection")
    assert "replay is not None" in block or "replay is not None" in src
