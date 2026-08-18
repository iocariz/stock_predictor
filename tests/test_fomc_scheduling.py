"""FOMC countdown must only know what was on the published calendar (P2-2).

The Fed publishes its meeting calendar about a year ahead, so a countdown to
a *scheduled* meeting is legitimately knowable. Emergency inter-meeting
actions are not: the 3 March 2020 cut was announced that morning, and the
15 March cut on a Sunday evening.
"""

from __future__ import annotations

import pandas as pd

from stock_predictor.calendar_features import (
    FOMC_SCHEDULED_DATES,
    FOMC_STATEMENT_DATES,
    FOMC_UNSCHEDULED_DATES,
    add_calendar_features,
)


def _days_to(date: str) -> int:
    out = add_calendar_features(pd.DataFrame({"date": pd.to_datetime([date])}))
    return int(out["cal_days_to_next_fomc"].iloc[0])


def test_the_2020_emergency_actions_are_classified_unscheduled() -> None:
    assert "2020-03-03" in FOMC_UNSCHEDULED_DATES
    assert "2020-03-15" in FOMC_UNSCHEDULED_DATES
    assert "2020-03-03" not in FOMC_SCHEDULED_DATES
    assert "2020-03-15" not in FOMC_SCHEDULED_DATES


def test_the_scheduled_march_2020_meeting_is_restored() -> None:
    """It was on the published calendar all along and only cancelled after the
    15 March action. Dropping it made the countdown wrong in both directions."""
    assert "2020-03-18" in FOMC_SCHEDULED_DATES


def test_late_february_2020_cannot_see_the_emergency_cut() -> None:
    """The regression: on 28 Feb the model was told an FOMC event was four days
    away. Nobody knew that until the morning of 3 March."""
    assert _days_to("2020-02-28") == 19, "next *scheduled* meeting is 18 March"


def test_a_normal_countdown_is_unaffected() -> None:
    assert _days_to("2024-01-30") == 1     # 31 Jan 2024 meeting
    assert _days_to("2024-01-31") == 0


def test_unscheduled_dates_are_still_exposed_for_backward_looking_use() -> None:
    """They are real events; only *anticipating* them is the leak."""
    for d in FOMC_UNSCHEDULED_DATES:
        assert d in FOMC_STATEMENT_DATES


def test_statement_dates_remain_the_sorted_union() -> None:
    assert list(FOMC_STATEMENT_DATES) == sorted(
        set(FOMC_SCHEDULED_DATES) | set(FOMC_UNSCHEDULED_DATES)
    )


def test_explicit_dates_still_override() -> None:
    import numpy as np

    out = add_calendar_features(
        pd.DataFrame({"date": pd.to_datetime(["2020-02-28"])}),
        fomc_dates=np.array(["2020-03-03"], dtype="datetime64[D]"),
    )
    assert int(out["cal_days_to_next_fomc"].iloc[0]) == 4, "caller wins"


def test_within_5d_flag_follows_the_scheduled_calendar() -> None:
    out = add_calendar_features(
        pd.DataFrame({"date": pd.to_datetime(["2020-02-28", "2020-03-16"])}),
    )
    assert out["cal_fomc_within_5d"].tolist() == [0, 1]
