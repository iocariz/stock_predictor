"""A name can be "covered" overall and still be unrankable today.

``check_download_coverage`` asks whether a ticker came back from the vendor at
all. That is not the same question as whether it can be *ranked*, which depends
on the recent sessions its cross-sectional features are computed from.

Measured on this repo's panel: AVB and EQR — both continuous S&P 500 members —
were present on 1 and 2 of the 20 most recent sessions, while counting as fully
covered. A momentum feature computed across a three-week hole is not a small
error in a good number; it is a different number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.universe import (
    DEFAULT_MIN_RECENT_COVERAGE,
    recent_coverage,
    thin_recent_names,
)

SESSIONS = pd.bdate_range("2026-01-01", periods=60)


def _panel(**holes: tuple[int, int]) -> pd.DataFrame:
    """Full history for AAA/BBB/CCC, with optional holes as (start, stop) index."""
    df = pd.DataFrame(100.0, index=SESSIONS, columns=["AAA", "BBB", "CCC"])
    for ticker, (lo, hi) in holes.items():
        df.iloc[lo:hi, df.columns.get_loc(ticker)] = np.nan
    return df


# ---------------------------------------------------------------------------
# The measure
# ---------------------------------------------------------------------------


def test_a_complete_history_is_fully_covered() -> None:
    cov = recent_coverage(_panel(), sessions=20)
    assert (cov == 1.0).all()


def test_only_the_recent_window_counts() -> None:
    """A hole long ago is already priced in; a hole now corrupts the features
    the next decision is made on."""
    old = recent_coverage(_panel(AAA=(0, 30)), sessions=20)
    recent = recent_coverage(_panel(AAA=(40, 60)), sessions=20)
    assert old["AAA"] == 1.0, "an old gap does not affect recent coverage"
    assert recent["AAA"] == 0.0


def test_partial_holes_are_measured_as_fractions() -> None:
    cov = recent_coverage(_panel(BBB=(50, 60)), sessions=20)
    assert cov["BBB"] == pytest.approx(0.5), "10 of the last 20 missing"


def test_a_ticker_absent_entirely_reads_as_zero() -> None:
    df = _panel()
    df["DDD"] = np.nan
    assert recent_coverage(df, sessions=20)["DDD"] == 0.0


def test_a_window_longer_than_the_history_uses_what_exists() -> None:
    cov = recent_coverage(_panel(), sessions=999)
    assert (cov == 1.0).all()


def test_zero_prices_do_not_count_as_coverage() -> None:
    """A zero is not a quote; it is a placeholder that would divide badly."""
    df = _panel()
    df.iloc[55:60, df.columns.get_loc("CCC")] = 0.0
    assert recent_coverage(df, sessions=20)["CCC"] < 1.0


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_names_below_the_floor_are_named() -> None:
    thin = thin_recent_names(_panel(AAA=(45, 60)), sessions=20, min_fraction=0.8)
    assert thin == ["AAA"]


def test_a_name_just_above_the_floor_is_kept() -> None:
    # 3 of 20 missing = 0.85 coverage, above a 0.8 floor
    assert thin_recent_names(_panel(AAA=(57, 60)), sessions=20, min_fraction=0.8) == []


def test_the_floor_is_inclusive() -> None:
    """Exactly at the threshold is acceptable; the guard is for names below it."""
    cov = recent_coverage(_panel(AAA=(56, 60)), sessions=20)
    assert cov["AAA"] == pytest.approx(0.8)
    assert thin_recent_names(_panel(AAA=(56, 60)), sessions=20, min_fraction=0.8) == []


def test_results_are_sorted_so_reports_are_stable() -> None:
    thin = thin_recent_names(_panel(BBB=(40, 60), AAA=(40, 60)), sessions=20,
                             min_fraction=0.8)
    assert thin == ["AAA", "BBB"]


def test_a_clean_panel_yields_no_warnings() -> None:
    assert thin_recent_names(_panel(), sessions=20, min_fraction=0.8) == []


def test_the_default_floor_is_documented_and_sane() -> None:
    assert 0.5 <= DEFAULT_MIN_RECENT_COVERAGE <= 1.0


def test_the_guard_reproduces_the_observed_failure() -> None:
    """AVB was present on 1 of the last 20 sessions while counting as covered."""
    df = _panel()
    df.iloc[41:60, df.columns.get_loc("AAA")] = np.nan   # 1 of last 20 present
    assert recent_coverage(df, sessions=20)["AAA"] == pytest.approx(0.05)
    assert "AAA" in thin_recent_names(df, sessions=20,
                                      min_fraction=DEFAULT_MIN_RECENT_COVERAGE)
