"""Universe selection and download-coverage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.universe import (
    DownloadCoverageError,
    check_download_coverage,
    sample_tickers,
)


def _alphabet_universe(n: int = 400) -> list[str]:
    """Deterministic tickers spread across the alphabet."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return sorted(f"{letters[i % 26]}{i:03d}" for i in range(n))


# ---------------------------------------------------------------------------
# sample_tickers
# ---------------------------------------------------------------------------


def test_sample_returns_everything_when_cap_exceeds_universe() -> None:
    tickers = _alphabet_universe(50)
    assert sample_tickers(tickers, 100, seed=42) == tickers
    assert sample_tickers(tickers, 50, seed=42) == tickers


def test_sample_is_not_an_alphabetical_prefix() -> None:
    """Regression: `tickers[:n]` truncated the universe at the alphabet.

    A 200-of-400 draw must reach past the middle of the alphabet, otherwise
    cross-sectional ranks and 'market' regime medians are computed on a
    biased slice of the index.
    """
    tickers = _alphabet_universe(400)
    picked = sample_tickers(tickers, 200, seed=42)
    assert len(picked) == 200
    assert picked != tickers[:200]
    # Half the draw should land in the back half of the alphabet.
    back_half = [t for t in picked if t > tickers[len(tickers) // 2]]
    assert len(back_half) > 60, "sample is skewed toward the front of the alphabet"


def test_sample_is_deterministic_for_a_seed() -> None:
    tickers = _alphabet_universe(400)
    a = sample_tickers(tickers, 100, seed=7)
    b = sample_tickers(tickers, 100, seed=7)
    c = sample_tickers(tickers, 100, seed=8)
    assert a == b
    assert a != c


def test_sample_output_is_sorted_and_unique() -> None:
    picked = sample_tickers(_alphabet_universe(400), 100, seed=1)
    assert picked == sorted(picked)
    assert len(set(picked)) == len(picked)


def test_sample_rejects_bad_cap() -> None:
    with pytest.raises(ValueError):
        sample_tickers(_alphabet_universe(10), 0, seed=1)


# ---------------------------------------------------------------------------
# check_download_coverage
# ---------------------------------------------------------------------------


def _wide(tickers: list[str], n_rows: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n_rows)
    return pd.DataFrame(
        np.arange(n_rows * len(tickers), dtype=float).reshape(n_rows, len(tickers)),
        index=idx,
        columns=tickers,
    )


def test_coverage_passes_when_everything_returned() -> None:
    req = ["A", "B", "C", "D"]
    cov = check_download_coverage(req, _wide(req), min_coverage=1.0)
    assert cov == pytest.approx(1.0)


def test_coverage_raises_on_alphabetically_truncated_download() -> None:
    """Regression: a rate-limited yfinance reply returned only a prefix and
    the pipeline carried on silently."""
    req = _alphabet_universe(100)
    got = _wide(req[:40])  # Yahoo returned only the front of the alphabet
    with pytest.raises(DownloadCoverageError) as exc:
        check_download_coverage(req, got, min_coverage=0.9)
    msg = str(exc.value)
    assert "40" in msg and "100" in msg


def test_coverage_counts_all_nan_columns_as_missing() -> None:
    req = ["A", "B", "C", "D"]
    got = _wide(req)
    got["D"] = np.nan  # column present but no bars
    with pytest.raises(DownloadCoverageError):
        check_download_coverage(req, got, min_coverage=0.9)


def test_coverage_below_threshold_but_allowed_only_warns(capsys) -> None:
    req = _alphabet_universe(100)
    cov = check_download_coverage(req, _wide(req[:80]), min_coverage=0.0)
    assert cov == pytest.approx(0.8)
    assert "missing" in capsys.readouterr().out.lower()


def test_coverage_rejects_empty_download() -> None:
    with pytest.raises(DownloadCoverageError):
        check_download_coverage(["A", "B"], pd.DataFrame(), min_coverage=0.9)


# ---------------------------------------------------------------------------
# Two-tier coverage: current members vs departed members
# ---------------------------------------------------------------------------


def test_departed_members_do_not_fail_the_run() -> None:
    """Yahoo drops most acquired/renamed symbols. On the real S&P universe
    that is ~97 of 691 tickers — all departed, with 100% of current members
    served. A flat threshold would block every full-universe run."""
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    requested = sorted(current + departed)
    # Every current member returned; only half the departed ones.
    got = _wide(current + departed[:10])

    cov = check_download_coverage(
        requested, got, min_coverage=0.98, active=set(current),
    )
    assert cov == pytest.approx(90 / 100)


def test_missing_current_member_still_fails() -> None:
    """A gap among current members means a broken or throttled download."""
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    got = _wide(current[:40] + departed)  # half the current members lost

    with pytest.raises(DownloadCoverageError, match="current"):
        check_download_coverage(
            sorted(current + departed), got, min_coverage=0.98, active=set(current),
        )


def test_survivorship_gap_is_reported(capsys) -> None:
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    check_download_coverage(
        sorted(current + departed), _wide(current), min_coverage=0.98,
        active=set(current),
    )
    out = capsys.readouterr().out.lower()
    assert "survivorship" in out
    assert "20" in out


def test_active_set_that_is_empty_falls_back_to_flat_threshold() -> None:
    req = _alphabet_universe(100)
    with pytest.raises(DownloadCoverageError):
        check_download_coverage(req, _wide(req[:40]), min_coverage=0.9, active=set())


# ---------------------------------------------------------------------------
# Deliberate removals are not vendor gaps
# ---------------------------------------------------------------------------


def test_recycled_symbols_are_not_counted_as_a_vendor_gap(capsys) -> None:
    """A dropped reused symbol is a cleaning decision, not missing data.

    Once a member is acquired its symbol can be reassigned, so the panel picks
    up a different issuer's prices under a departed member's name -- Qwest's Q
    priced from 2025, Anadarko's APC from 2026. ``drop_recycled_prices``
    removes those blocks *before* this check runs, so they arrive here looking
    exactly like names the vendor never served.

    Measured on the real panel: 65 departed members absent, of which 44 were
    recycled symbols the cache holds good data for and only 21 were genuinely
    unavailable. The message attributed all 65 to the vendor and called the
    result flattered, when for those 44 the removal made it *more* honest.
    """
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    recycled = departed[:12]
    check_download_coverage(
        sorted(current + departed), _wide(current), min_coverage=0.98,
        active=set(current), recycled=recycled,
    )
    out = capsys.readouterr().out
    assert "8/20" in out, f"vendor gap not reported net of recycled names: {out}"
    assert "12" in out, "the recycled count is not reported at all"


def test_recycled_symbols_are_reported_as_their_own_line(capsys) -> None:
    """Reported, not silently netted off: a reader has to be able to see both
    causes, because they call for opposite responses. A vendor gap wants a
    refetch; a recycled symbol must never be refetched -- it would return the
    same wrong company."""
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    check_download_coverage(
        sorted(current + departed), _wide(current), min_coverage=0.98,
        active=set(current), recycled=departed[:12],
    )
    out = capsys.readouterr().out.lower()
    assert "reused" in out or "recycled" in out


def test_an_all_recycled_gap_is_not_called_survivorship(capsys) -> None:
    """If every absent name was deliberately removed there is no vendor gap,
    and saying results are flattered by survivorship would be false."""
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    check_download_coverage(
        sorted(current + departed), _wide(current), min_coverage=0.98,
        active=set(current), recycled=departed,
    )
    out = capsys.readouterr().out.lower()
    # Not a bare word check: the reused-symbols line names survivorship in
    # order to deny it ("not a survivorship gap"). What must be absent is the
    # claim itself.
    assert "survivorship gap:" not in out, (
        f"claimed a survivorship gap with no vendor gap: {out}")
    assert "flatters results" not in out


def test_recycled_defaults_to_nothing(capsys) -> None:
    """Callers that do not drop recycled symbols keep the old reading."""
    current = _alphabet_universe(80)
    departed = [f"OLD{i:03d}" for i in range(20)]
    check_download_coverage(
        sorted(current + departed), _wide(current), min_coverage=0.98,
        active=set(current),
    )
    out = capsys.readouterr().out
    assert "20/20" in out
