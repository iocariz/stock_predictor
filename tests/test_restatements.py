"""Restatements must become visible once filed (P2-4).

`keep="first"` per period kept only the original filing, so a revision was
discarded entirely rather than merely delayed. The existing test asserted a
restatement does not leak *backwards* — and passed — while the value it was
guarding never appeared at all.
"""

from __future__ import annotations

import pandas as pd

from stock_predictor.fundamentals import (
    asof_join_fundamentals,
    extract_concepts,
    trailing_twelve_months,
)


def _facts(tag="Assets"):
    obs = [
        {"end": "2024-03-31", "val": 1000.0, "filed": "2024-05-10",
         "form": "10-Q", "fy": 2024, "fp": "Q1"},
        {"end": "2024-06-30", "val": 1100.0, "filed": "2024-08-09",
         "form": "10-Q", "fy": 2024, "fp": "Q2"},
    ]
    return {"facts": {"us-gaap": {tag: {"units": {"USD": obs}}}}}


def _restated(value=555.0, filed="2024-11-08") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": "AAA", "concept": "assets",
        "period_end": pd.Timestamp("2024-03-31"), "period_start": pd.NaT,
        "filed": pd.Timestamp(filed), "value": value,
        "form": "10-Q/A", "fy": 2024, "fp": "Q1",
    }])


def _joined(dates, fund):
    return asof_join_fundamentals(
        pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "AAA"}), fund,
    )["raw_assets"].tolist()


def test_a_restatement_is_invisible_before_it_is_filed() -> None:
    fund = trailing_twelve_months(
        pd.concat([extract_concepts(_facts(), "AAA"), _restated()], ignore_index=True)
    )
    # 2024-05-13, not the 2024-05-10 filing date itself: EDGAR dates carry no
    # time and filings routinely land after the close, so a figure is usable
    # from the next session.
    seen = _joined(["2024-05-13", "2024-07-01", "2024-11-07"], fund)
    assert seen[0] == 1000.0
    assert seen[1] == 1000.0
    assert seen[2] == 1100.0, "Q2 filing supersedes Q1 by July"
    assert 555.0 not in seen, "the revision must not leak backwards"


def test_a_restatement_becomes_visible_after_it_is_filed() -> None:
    """The half that was missing: the revision must actually arrive."""
    fund = trailing_twelve_months(
        pd.concat([extract_concepts(_facts(), "AAA"), _restated()], ignore_index=True)
    )
    q1_after = asof_join_fundamentals(
        pd.DataFrame({"date": pd.to_datetime(["2024-12-01"]), "ticker": "AAA"}),
        fund[fund["period_end"] == pd.Timestamp("2024-03-31")],
    )["raw_assets"].iloc[0]
    assert q1_after == 555.0, "December must see the November revision of Q1"


def test_the_newest_filing_wins_at_any_date() -> None:
    fund = trailing_twelve_months(pd.concat([
        extract_concepts(_facts(), "AAA"),
        _restated(value=555.0, filed="2024-11-08"),
        _restated(value=333.0, filed="2025-02-01"),
    ], ignore_index=True))
    q1 = fund[fund["period_end"] == pd.Timestamp("2024-03-31")]
    seen = _joined(["2024-06-01", "2024-12-01", "2025-03-01"], q1)
    assert seen == [1000.0, 555.0, 333.0]


def test_every_filing_vintage_is_retained() -> None:
    fund = trailing_twelve_months(
        pd.concat([extract_concepts(_facts(), "AAA"), _restated()], ignore_index=True)
    )
    q1 = fund[fund["period_end"] == pd.Timestamp("2024-03-31")]
    assert len(q1) == 2, "original and revision must both survive"
    assert set(q1["value"]) == {1000.0, 555.0}


def test_edgar_cache_has_a_refresh_policy() -> None:
    import inspect

    from stock_predictor import fundamentals

    src = inspect.getsource(fundamentals.fetch_fundamentals)
    assert "max_age_days" in src, "a permanently frozen cache never sees new filings"
