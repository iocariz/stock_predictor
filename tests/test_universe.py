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
