"""A recovery pass has to terminate.

``recover_delisted.py`` promises convergence: "Names Tiingo genuinely does not
serve are recorded as empty and not retried inside the TTL, so repeated passes
converge instead of looping forever on the same failures."

That covers names the vendor returns *nothing* for. It does not cover names it
returns almost nothing for. A ticker with 1-19 rows is written to the cache with
``empty: False``, so the provider serves it from cache and counts it recovered,
while ``outstanding()`` rejects it against MIN_ROWS and hands it back. Observed
on the real cache: BK and DF sat at 5 rows and WRK at 1, fetched days apart and
still outstanding, so every pass reported "recovered 3" and "3 still
outstanding" together and the run never reached "Nothing left to fetch".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recover_delisted import MIN_ROWS, outstanding  # noqa: E402


def _cache(tmp_path: Path, ticker: str, rows: int, **entry) -> Path:
    """A cached ticker with *rows* sessions and a manifest entry to match."""
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=rows),
        "adj_close": [100.0] * rows,
    })
    df.to_parquet(tmp_path / f"{ticker}.parquet", index=False)
    manifest = {}
    path = tmp_path / "_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    manifest[ticker] = {
        "start": "2010-01-01", "end": "2026-09-11", "empty": False,
        "schema": 1, "fetched": "2026-09-11T00:00:00+00:00", **entry,
    }
    path.write_text(json.dumps(manifest))
    return tmp_path


def test_a_thin_ticker_is_not_offered_forever(tmp_path: Path) -> None:
    """The loop that did not terminate."""
    _cache(tmp_path, "WRK", rows=1)
    todo, _, _ = outstanding(["WRK"], tmp_path)
    assert todo == [], "a thin ticker was handed back for another pass"


def test_a_thin_ticker_counts_as_unavailable_not_recovered(tmp_path: Path) -> None:
    """Reporting it as recovered would overstate the panel's coverage."""
    _cache(tmp_path, "BK", rows=5)
    _, have, absent = outstanding(["BK"], tmp_path)
    assert have == 0, "5 rows is not a recovered ticker"
    assert absent == 1


def test_a_full_ticker_is_still_recovered(tmp_path: Path) -> None:
    _cache(tmp_path, "GOOD", rows=MIN_ROWS + 10)
    todo, have, absent = outstanding(["GOOD"], tmp_path)
    assert (todo, have, absent) == ([], 1, 0)


def test_an_unfetched_ticker_is_still_offered(tmp_path: Path) -> None:
    """Convergence must not come from giving up on names never tried."""
    todo, have, absent = outstanding(["NEVER"], tmp_path)
    assert todo == ["NEVER"]
    assert (have, absent) == (0, 0)


def test_a_thin_cache_from_a_narrower_window_is_retried(tmp_path: Path) -> None:
    """A short file because the *request* was short is not the vendor's answer,
    so it must still be refetched for the wider window."""
    _cache(tmp_path, "NARROW", rows=5, start="2026-01-01")
    todo, _, _ = outstanding(["NARROW"], tmp_path, start="2010-01-01")
    assert todo == ["NARROW"]
