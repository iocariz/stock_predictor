"""Ticker -> CIK resolution.

`company_tickers.json` is a *current* registrant list, and not even a complete
one: AEP, EA, DFS and ANSS are all absent from it. Every ticker it misses is a
company whose fundamentals silently become NaN, and the misses skew toward
acquired and delisted names — the survivorship-relevant tail. EDGAR's company
browser resolves tickers server-side and recovers most of them.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from stock_predictor.fundamentals import (
    build_cik_map,
    load_cik_fallback_cache,
    resolve_cik_via_edgar,
    save_cik_fallback_cache,
)

ATOM = "<feed><company-info><cik>{cik}</cik></company-info></feed>"
NO_MATCH = "<feed><company-info></company-info></feed>"


# ---------------------------------------------------------------------------
# The single lookup
# ---------------------------------------------------------------------------


def test_a_ticker_resolves_to_a_padded_cik() -> None:
    with patch("stock_predictor.fundamentals._get_text", return_value=ATOM.format(cik="4904")):
        assert resolve_cik_via_edgar("AEP") == "0000004904"


def test_an_unknown_ticker_resolves_to_none() -> None:
    with patch("stock_predictor.fundamentals._get_text", return_value=NO_MATCH):
        assert resolve_cik_via_edgar("NOTATICKER") is None


def test_a_network_error_is_not_fatal() -> None:
    """One bad lookup must not lose the other 600 tickers."""
    with patch("stock_predictor.fundamentals._get_text", side_effect=OSError("reset")):
        assert resolve_cik_via_edgar("AEP") is None


# ---------------------------------------------------------------------------
# The layered map
# ---------------------------------------------------------------------------


def test_the_primary_map_is_used_and_not_re_queried() -> None:
    base = {"AAPL": "0000320193"}
    with patch("stock_predictor.fundamentals._get_text",
               side_effect=AssertionError("must not hit the network")):
        out, stats = build_cik_map(["AAPL"], primary=base, cache={})
    assert out["AAPL"] == "0000320193"
    assert stats["primary"] == 1 and stats["resolved"] == 0


def test_missing_tickers_fall_through_to_edgar() -> None:
    with patch("stock_predictor.fundamentals._get_text", return_value=ATOM.format(cik="712515")):
        out, stats = build_cik_map(["EA"], primary={}, cache={}, pause_s=0.0)
    assert out["EA"] == "0000712515"
    assert stats["resolved"] == 1


def test_lookups_are_case_insensitive() -> None:
    with patch("stock_predictor.fundamentals._get_text", return_value=ATOM.format(cik="4904")):
        out, _ = build_cik_map(["aep"], primary={}, cache={}, pause_s=0.0)
    assert out["AEP"] == "0000004904"


def test_the_cache_prevents_a_second_network_call() -> None:
    cache: dict[str, str | None] = {}
    with patch("stock_predictor.fundamentals._get_text",
               return_value=ATOM.format(cik="4904")) as g:
        build_cik_map(["AEP"], primary={}, cache=cache, pause_s=0.0)
        assert g.call_count == 1
        build_cik_map(["AEP"], primary={}, cache=cache, pause_s=0.0)
        assert g.call_count == 1, "a resolved ticker must not be re-queried"


def test_negative_results_are_cached_too() -> None:
    """Otherwise every run re-spends the SEC rate limit on the same 31 ghosts."""
    cache: dict[str, str | None] = {}
    with patch("stock_predictor.fundamentals._get_text", return_value=NO_MATCH) as g:
        build_cik_map(["AGN"], primary={}, cache=cache, pause_s=0.0)
        build_cik_map(["AGN"], primary={}, cache=cache, pause_s=0.0)
    assert g.call_count == 1
    assert cache["AGN"] is None
    assert "AGN" not in build_cik_map(["AGN"], primary={}, cache=cache, pause_s=0.0)[0]


def test_unresolved_tickers_are_reported_not_hidden() -> None:
    with patch("stock_predictor.fundamentals._get_text", return_value=NO_MATCH):
        out, stats = build_cik_map(["AGN", "CELG"], primary={}, cache={}, pause_s=0.0)
    assert out == {}
    assert stats["unresolved"] == 2


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def test_cache_round_trips_through_disk(tmp_path) -> None:
    p = tmp_path / "cik_fallback.json"
    save_cik_fallback_cache(p, {"AEP": "0000004904", "AGN": None})
    back = load_cik_fallback_cache(p)
    assert back == {"AEP": "0000004904", "AGN": None}


def test_a_missing_cache_file_is_empty_not_an_error(tmp_path) -> None:
    assert load_cik_fallback_cache(tmp_path / "nope.json") == {}


def test_a_corrupt_cache_file_is_ignored(tmp_path) -> None:
    """A half-written cache must not brick every future run."""
    p = tmp_path / "cik_fallback.json"
    p.write_text("{not json")
    assert load_cik_fallback_cache(p) == {}


def test_cache_survives_an_unwritable_directory(tmp_path) -> None:
    """Persisting the cache is an optimisation, never a reason to fail a run."""
    save_cik_fallback_cache(tmp_path / "missing_dir" / "c.json", {"AEP": "0000004904"})


def test_saved_cache_is_readable_json(tmp_path) -> None:
    p = tmp_path / "c.json"
    save_cik_fallback_cache(p, {"AEP": "0000004904"})
    assert json.loads(p.read_text())["AEP"] == "0000004904"


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_tickers_are_skipped(bad: str) -> None:
    with patch("stock_predictor.fundamentals._get_text",
               side_effect=AssertionError("must not query a blank ticker")):
        out, _ = build_cik_map([bad], primary={}, cache={}, pause_s=0.0)
    assert out == {}


# ---------------------------------------------------------------------------
# Reached through the real entry point
# ---------------------------------------------------------------------------
#
# The row-role fix was inert for a full regeneration cycle because the wiring
# never carried it to the caller. These tests exercise fetch_fundamentals
# itself rather than the resolver in isolation.


def _facts(cik: str = "0000004904") -> dict:
    return {"cik": int(cik), "entityName": "X", "facts": {}}


def test_fetch_fundamentals_recovers_a_ticker_the_sec_map_lacks(tmp_path) -> None:
    from stock_predictor import fundamentals as F

    with (
        patch.object(F, "load_cik_map", return_value={}),
        patch.object(F, "_get_text", return_value=ATOM.format(cik="4904")),
        patch.object(F, "_get_json", return_value=_facts()) as facts,
    ):
        F.fetch_fundamentals(["AEP"], cache_dir=tmp_path, min_interval_s=0.0)

    assert facts.called, "a recovered ticker must actually be fetched"
    assert "CIK0000004904" in facts.call_args[0][0]


def test_fetch_fundamentals_persists_the_fallback_between_runs(tmp_path) -> None:
    from stock_predictor import fundamentals as F

    with (
        patch.object(F, "load_cik_map", return_value={}),
        patch.object(F, "_get_text", return_value=ATOM.format(cik="4904")) as text,
        patch.object(F, "_get_json", return_value=_facts()),
    ):
        F.fetch_fundamentals(["AEP"], cache_dir=tmp_path, min_interval_s=0.0)
        first = text.call_count
        # Expire the per-ticker parquet so the CIK path is exercised again.
        for p in tmp_path.glob("*.parquet"):
            p.unlink()
        F.fetch_fundamentals(["AEP"], cache_dir=tmp_path, min_interval_s=0.0)
        assert text.call_count == first, "the resolved CIK must come from disk"

    assert (tmp_path / F.CIK_FALLBACK_CACHE).exists()


def test_a_permanently_unresolvable_ticker_is_not_retried(tmp_path) -> None:
    from stock_predictor import fundamentals as F

    with (
        patch.object(F, "load_cik_map", return_value={}),
        patch.object(F, "_get_text", return_value=NO_MATCH) as text,
        patch.object(F, "_get_json", return_value=_facts()),
    ):
        F.fetch_fundamentals(["AGN"], cache_dir=tmp_path, min_interval_s=0.0)
        F.fetch_fundamentals(["AGN"], cache_dir=tmp_path, min_interval_s=0.0)
    assert text.call_count == 1, "negative results must persist too"
