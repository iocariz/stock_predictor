"""Future sessions come from the exchange calendar, not from business days.

Live order generation has to place entry and expiry dates beyond the end of
the downloaded price history. Those were projected with ``pd.bdate_range``,
which counts every weekday as a session. Measured on this repo's own panel,
**156 of 4,337 business days were not sessions — 3.6%, about 9.1 a year**. So
an entry date could land on a closed market, and a 63-session expiry drifted
roughly two sessions early.

The cases below are the ones a hand-rolled rule set gets wrong: Good Friday is
a market holiday but not a federal one, Juneteenth only became a holiday in
2022, July 4th shifts when it falls at a weekend, the day after Thanksgiving
is open (early close, but open), and 2018-12-05 was an ad-hoc closure for a
national day of mourning that no rule predicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.execution_calendar import (
    exchange_sessions,
    extend_calendar,
    is_trading_session,
    offset_trading_days,
)

PAST = pd.DatetimeIndex(pd.bdate_range("2026-01-02", "2026-03-31"))


def _dates(arr) -> list[str]:
    return [str(pd.Timestamp(d).date()) for d in arr]


# ---------------------------------------------------------------------------
# Sessions the exchange actually holds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "why"),
    [
        ("2026-04-03", "Good Friday — a market holiday but not a federal one"),
        ("2025-04-18", "Good Friday, a different year"),
        ("2026-06-19", "Juneteenth, only a holiday since 2022"),
        ("2026-07-03", "observed July 4th, because the 4th is a Saturday"),
        ("2026-09-07", "Labor Day"),
        ("2026-11-26", "Thanksgiving"),
        ("2026-12-25", "Christmas"),
        ("2018-12-05", "national day of mourning — an ad-hoc closure no rule predicts"),
    ],
)
def test_known_closures_are_not_sessions(day: str, why: str) -> None:
    assert not is_trading_session(day), why


@pytest.mark.parametrize(
    ("day", "why"),
    [
        ("2026-11-27", "the day after Thanksgiving is open — early close, but open"),
        ("2026-12-24", "Christmas Eve is a session"),
        ("2026-01-02", "an ordinary weekday"),
    ],
)
def test_open_days_are_sessions(day: str, why: str) -> None:
    assert is_trading_session(day), why


def test_weekends_are_never_sessions() -> None:
    assert not is_trading_session("2026-08-22")   # Saturday
    assert not is_trading_session("2026-08-23")   # Sunday


def test_sessions_in_a_range_exclude_the_holidays_inside_it() -> None:
    sess = exchange_sessions("2026-08-21", "2026-12-31")
    got = set(_dates(sess))
    assert "2026-09-07" not in got and "2026-11-26" not in got
    assert len(got) < len(pd.bdate_range("2026-08-21", "2026-12-31"))


# ---------------------------------------------------------------------------
# Extending past the end of the price history
# ---------------------------------------------------------------------------


def test_extension_returns_exactly_the_sessions_asked_for() -> None:
    out = extend_calendar(PAST.to_numpy(), 10)
    assert len(out) == len(PAST) + 10


def test_extension_skips_a_holiday_instead_of_counting_it() -> None:
    """History ends the day before Good Friday 2026; the next session is the
    Monday, not the Friday."""
    hist = pd.DatetimeIndex(["2026-04-01", "2026-04-02"]).to_numpy()
    out = _dates(extend_calendar(hist, 3))
    assert "2026-04-03" not in out, "Good Friday is not a session"
    assert out[2] == "2026-04-06", "the next session is the following Monday"


def test_a_business_day_projection_would_have_been_wrong_here() -> None:
    """Pins the defect: bdate_range proposes the closed Friday."""
    naive = pd.bdate_range("2026-04-03", periods=1)
    assert str(naive[0].date()) == "2026-04-03"
    assert not is_trading_session("2026-04-03")


def test_a_long_hold_no_longer_expires_early() -> None:
    """A 63-session expiry projected on business days lands ~2 sessions early,
    because roughly two holidays fall inside any given quarter."""
    hist = pd.DatetimeIndex(["2026-01-02"]).to_numpy()
    real = pd.Timestamp(extend_calendar(hist, 63)[-1])
    naive = pd.bdate_range("2026-01-05", periods=63)[-1]
    assert real > naive, "the true 63rd session is later than the 63rd weekday"
    assert (real - naive).days >= 2


def test_extension_preserves_history_and_stays_sorted() -> None:
    out = pd.DatetimeIndex(extend_calendar(PAST.to_numpy(), 5))
    assert out.is_monotonic_increasing
    assert not out.duplicated().any()
    assert list(out[: len(PAST)]) == list(PAST)


def test_extension_never_repeats_the_last_known_session() -> None:
    hist = pd.DatetimeIndex(["2026-04-01", "2026-04-02"]).to_numpy()
    out = _dates(extend_calendar(hist, 2))
    assert out.count("2026-04-02") == 1


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_no_extension_requested_is_a_no_op() -> None:
    assert len(extend_calendar(PAST.to_numpy(), 0)) == len(PAST)


def test_an_empty_history_stays_empty() -> None:
    assert len(extend_calendar(np.array([], dtype="datetime64[ns]"), 5)) == 0


def test_offsets_land_on_real_sessions_after_extension() -> None:
    """The property the live path depends on: an expiry date must be a day the
    market is actually open."""
    cal = extend_calendar(PAST.to_numpy(), 80)
    exit_ts = offset_trading_days(pd.Timestamp("2026-03-31"), 63, cal)
    assert exit_ts is not None
    assert is_trading_session(exit_ts)


# ---------------------------------------------------------------------------
# Data lag must not eat the placement margin
# ---------------------------------------------------------------------------


def test_the_extension_can_be_anchored_past_the_data_end() -> None:
    """The calendar is extended from the last *data* session. When a run
    happens several sessions later — stale vendor, a weekend backlog — that
    lag consumed the margin and expiry placement silently returned None,
    surfacing as "no available cohort slots" rather than as an error."""
    hist = pd.DatetimeIndex(pd.bdate_range(end="2026-08-21", periods=200)).to_numpy()
    lagged = extend_calendar(hist, 26, anchor="2026-09-04")
    entry = pd.Timestamp("2026-09-08")
    assert offset_trading_days(entry, 21, lagged) is not None


def test_anchoring_is_a_floor_not_a_shift() -> None:
    """An anchor at or before the data end changes nothing."""
    hist = pd.DatetimeIndex(pd.bdate_range(end="2026-08-21", periods=50)).to_numpy()
    plain = extend_calendar(hist, 10)
    anchored = extend_calendar(hist, 10, anchor="2026-01-01")
    assert list(pd.DatetimeIndex(plain)) == list(pd.DatetimeIndex(anchored))


def test_an_anchored_calendar_still_holds_real_sessions() -> None:
    hist = pd.DatetimeIndex(pd.bdate_range(end="2026-08-21", periods=50)).to_numpy()
    out = pd.DatetimeIndex(extend_calendar(hist, 30, anchor="2026-09-04"))
    assert out.is_monotonic_increasing and not out.duplicated().any()
    assert all(is_trading_session(d) for d in out[-10:])
