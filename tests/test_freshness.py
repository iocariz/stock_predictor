"""Refuse to trade on inputs that have gone stale.

This is not hypothetical. The model deployed in this repo until 2026-08-21 had
`train_end` of **2022-12-31 — 3.64 years before it was used to pick trades** —
at a 10-day horizon while the strategy traded 63. Nothing in the system said a
word; `predict-sp500` printed the feature count and the horizon and went ahead.

Market data goes stale the same way: a vendor outage that serves yesterday's
snapshot looks exactly like a quiet market until you check the dates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.freshness import (
    FreshnessPolicy,
    check_freshness,
    describe,
)

TODAY = pd.Timestamp("2026-08-21")


def _meta(train_end: str = "2024-12-31") -> dict:
    return {"train_end": train_end, "horizon": 63, "feature_cols": ["ret_1d"]}


def _sessions(last: str, n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range(end=last, periods=n)


# ---------------------------------------------------------------------------
# Model artifact age
# ---------------------------------------------------------------------------


def test_a_fresh_model_raises_nothing() -> None:
    assert check_freshness(_meta("2026-06-30"), _sessions("2026-08-20"),
                           as_of=TODAY) == []


def test_a_model_past_the_age_limit_is_flagged() -> None:
    out = check_freshness(_meta("2020-01-01"), _sessions("2026-08-20"), as_of=TODAY)
    assert [f.kind for f in out] == ["model_age"]
    assert out[0].value > out[0].limit


def test_the_actual_incident_would_have_been_caught() -> None:
    """The model that was live until 2026-08-21: train_end 2022-12-31."""
    out = check_freshness(_meta("2022-12-31"), _sessions("2026-08-20"), as_of=TODAY)
    kinds = {f.kind for f in out}
    assert "model_age" in kinds
    stale = next(f for f in out if f.kind == "model_age")
    assert stale.value == pytest.approx(3.64, abs=0.05)


def test_the_model_currently_deployed_passes_the_default_policy() -> None:
    """train_end 2024-12-31 is 1.64 years old — inside the 2.0 default."""
    out = check_freshness(_meta("2024-12-31"), _sessions("2026-08-20"), as_of=TODAY)
    assert [f.kind for f in out] == []


def test_metadata_without_a_train_end_is_itself_a_finding() -> None:
    """Unknown age is not the same as fresh."""
    out = check_freshness({"horizon": 63}, _sessions("2026-08-20"), as_of=TODAY)
    assert any(f.kind == "model_age" for f in out)


def test_an_unparseable_train_end_is_a_finding_not_a_crash() -> None:
    out = check_freshness(_meta("not-a-date"), _sessions("2026-08-20"), as_of=TODAY)
    assert any(f.kind == "model_age" for f in out)


# ---------------------------------------------------------------------------
# Market data age
# ---------------------------------------------------------------------------


def test_data_through_the_previous_session_is_fine() -> None:
    assert check_freshness(_meta(), _sessions("2026-08-20"), as_of=TODAY) == []


def test_a_long_weekend_does_not_trip_the_gate() -> None:
    """Counting calendar days would flag every Monday after a holiday."""
    out = check_freshness(_meta(), _sessions("2026-09-04"),
                          as_of=pd.Timestamp("2026-09-08"))
    assert [f.kind for f in out] == [], "2026-09-07 is Labor Day"


def test_data_many_sessions_behind_is_flagged() -> None:
    out = check_freshness(_meta(), _sessions("2026-07-01"), as_of=TODAY)
    assert [f.kind for f in out] == ["data_age"]
    assert out[0].value > out[0].limit


def test_an_empty_calendar_is_a_finding() -> None:
    out = check_freshness(_meta(), pd.DatetimeIndex([]), as_of=TODAY)
    assert any(f.kind == "data_age" for f in out)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_limits_are_configurable() -> None:
    lenient = FreshnessPolicy(max_model_age_years=10.0)
    assert check_freshness(_meta("2020-01-01"), _sessions("2026-08-20"),
                           as_of=TODAY, policy=lenient) == []


def test_a_zero_limit_disables_a_check() -> None:
    off = FreshnessPolicy(max_model_age_years=0.0, max_data_age_sessions=0)
    assert check_freshness(_meta("2000-01-01"), _sessions("2010-01-01"),
                           as_of=TODAY, policy=off) == []


def test_both_checks_can_fire_at_once() -> None:
    out = check_freshness(_meta("2019-01-01"), _sessions("2026-01-02"), as_of=TODAY)
    assert {f.kind for f in out} == {"model_age", "data_age"}


def test_findings_describe_themselves_for_an_operator() -> None:
    out = check_freshness(_meta("2019-01-01"), _sessions("2026-08-20"), as_of=TODAY)
    text = describe(out)
    assert "model_age" in text
    assert "2019-01-01" in text or "7." in text
    assert describe([]) == ""


def test_a_policy_rejects_negative_limits() -> None:
    for bad in (dict(max_model_age_years=-1.0), dict(max_data_age_sessions=-1)):
        with pytest.raises(ValueError):
            FreshnessPolicy(**bad)
