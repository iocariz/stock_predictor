"""A restated figure has to become visible, and a filing has to wait a day.

Two defects on the ``--fundamentals`` path. It is off in the baseline
(``"fundamentals": false`` in the run manifest), so nothing published moves;
both are latent until somebody turns it on.

**Restatements were discarded, not delayed.** Flow concepts were collapsed to
the earliest filing per period twice over -- once across the whole frame and
again per ``(ticker, concept)`` before the rolling sum -- so a later revision
never entered the panel at all. The module already knew this was wrong and
said so about balance-sheet concepts:

    Balances keep every filing vintage. Collapsing to the earliest filing per
    period discarded revisions outright rather than merely delaying them, so a
    restatement never became visible at all.

Flows were simply not given the same treatment. They need more than a
different ``drop_duplicates``: a trailing-twelve-month sum has to be computed
*per filing vintage*, because what the TTM was worth on a given day depends on
which revisions had landed by then.

**Same-day filings could leak.** The as-of join used
``allow_exact_matches=True``, and EDGAR's ``filed`` field carries a date with no
time. A 10-Q filed after the close on day D was therefore visible to day D's
signal. ``specs.md:193`` asks for fundamentals joined on availability, and
availability before a close cannot be established from a date alone. A filing
now becomes usable the next session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.fundamentals import (
    asof_join_fundamentals,
    trailing_twelve_months,
)

TICKER = "AAA"


def _rows(specs) -> pd.DataFrame:
    """(fp, fy, period_start, period_end, filed, value) → a filing frame."""
    return pd.DataFrame([
        {"ticker": TICKER, "concept": "revenue", "fp": fp, "fy": fy,
         "period_start": pd.Timestamp(ps), "period_end": pd.Timestamp(pe),
         "filed": pd.Timestamp(f), "value": float(v)}
        for fp, fy, ps, pe, f, v in specs
    ])


def _four_quarters(*, restate: tuple | None = None) -> pd.DataFrame:
    """Four clean quarters, optionally with Q1 restated later."""
    specs = [
        ("Q1", 2023, "2023-01-01", "2023-03-31", "2023-04-20", 100),
        ("Q2", 2023, "2023-04-01", "2023-06-30", "2023-07-20", 110),
        ("Q3", 2023, "2023-07-01", "2023-09-30", "2023-10-20", 120),
        ("Q4", 2023, "2023-10-01", "2023-12-31", "2024-01-20", 130),
    ]
    if restate is not None:
        filed, value = restate
        specs.append(("Q1", 2023, "2023-01-01", "2023-03-31", filed, value))
    return _rows(specs)


def _ttm(fund: pd.DataFrame) -> pd.DataFrame:
    out = trailing_twelve_months(fund)
    return out[out["ttm"].notna()].sort_values("filed").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Restatements
# ---------------------------------------------------------------------------


def test_a_clean_four_quarters_sums_to_ttm() -> None:
    t = _ttm(_four_quarters())
    assert len(t) == 1
    assert t.loc[0, "ttm"] == pytest.approx(460.0)
    assert t.loc[0, "filed"] == pd.Timestamp("2024-01-20")


def test_a_restatement_becomes_visible() -> None:
    """The defect. A revised Q1 filed after the TTM was first computed has to
    produce a second, later figure -- not vanish."""
    t = _ttm(_four_quarters(restate=("2024-03-01", 150)))
    assert len(t) >= 2, "the restatement never produced a new TTM"
    assert t["ttm"].iloc[-1] == pytest.approx(510.0)   # 150 + 110 + 120 + 130


def test_the_restated_ttm_is_dated_when_it_was_filed() -> None:
    """Delayed, not backdated: the revision is knowable on its own filing
    date and not one day before it."""
    t = _ttm(_four_quarters(restate=("2024-03-01", 150)))
    assert t["filed"].iloc[-1] == pd.Timestamp("2024-03-01")


def test_the_original_figure_survives_the_restatement() -> None:
    """A backtest reading 2024-02-01 must still see what was public then."""
    t = _ttm(_four_quarters(restate=("2024-03-01", 150)))
    before = t[t["filed"] <= pd.Timestamp("2024-02-01")]
    assert len(before) == 1
    assert before["ttm"].iloc[0] == pytest.approx(460.0)


def test_a_restatement_of_a_stale_quarter_does_not_rewrite_history() -> None:
    """Revising a quarter that has aged out of the window changes nothing
    about the figures already published."""
    base = _four_quarters()
    later = _rows([
        ("Q1", 2024, "2024-01-01", "2024-03-31", "2024-04-20", 140),
    ])
    stale = _rows([
        ("Q1", 2023, "2023-01-01", "2023-03-31", "2024-05-01", 999),
    ])
    t = _ttm(pd.concat([base, later, stale], ignore_index=True))
    at_feb = t[t["filed"] <= pd.Timestamp("2024-02-01")]
    assert at_feb["ttm"].iloc[-1] == pytest.approx(460.0)


def test_balances_keep_every_vintage() -> None:
    """Regression guard: stocks already behaved and must keep behaving."""
    fund = pd.DataFrame([
        {"ticker": TICKER, "concept": "assets", "fp": "Q1", "fy": 2023,
         "period_start": pd.NaT, "period_end": pd.Timestamp("2023-03-31"),
         "filed": pd.Timestamp(f), "value": float(v)}
        for f, v in (("2023-04-20", 1000), ("2023-08-01", 1100))
    ])
    out = trailing_twelve_months(fund)
    assert len(out) == 2
    assert set(out["value"]) == {1000.0, 1100.0}


# ---------------------------------------------------------------------------
# The same-day leak
# ---------------------------------------------------------------------------


def _panel(dates) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(dates)),
                         "ticker": [TICKER] * len(list(dates))})


def _fund_at(filed: str, value: float = 460.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": TICKER, "concept": "revenue",
        "period_end": pd.Timestamp("2023-12-31"),
        "filed": pd.Timestamp(filed), "value": value, "ttm": value,
    }])


def test_a_filing_is_not_visible_on_its_own_filing_date() -> None:
    """EDGAR records a date, not a time, and filings routinely land after the
    close. Same-date visibility cannot be established, so it is not assumed."""
    joined = asof_join_fundamentals(
        _panel(["2024-01-19", "2024-01-20", "2024-01-22"]), _fund_at("2024-01-20"))
    got = joined["raw_revenue"].tolist()
    assert pd.isna(got[0]), "visible before it was filed"
    assert pd.isna(got[1]), "visible on the filing date itself"
    assert got[2] == pytest.approx(460.0), "never became visible at all"


def test_the_next_session_sees_it() -> None:
    joined = asof_join_fundamentals(_panel(["2024-01-21"]), _fund_at("2024-01-20"))
    assert joined["raw_revenue"].iloc[0] == pytest.approx(460.0)


def test_the_newest_filing_on_or_before_still_wins() -> None:
    fund = pd.concat([_fund_at("2024-01-20", 460.0), _fund_at("2024-03-01", 510.0)],
                     ignore_index=True)
    joined = asof_join_fundamentals(
        _panel(["2024-02-01", "2024-03-05"]), fund)
    assert joined["raw_revenue"].tolist() == pytest.approx([460.0, 510.0])


def test_the_reported_filing_date_is_the_one_used() -> None:
    joined = asof_join_fundamentals(_panel(["2024-01-25"]), _fund_at("2024-01-20"))
    assert pd.Timestamp(joined["fund_filed"].iloc[0]) == pd.Timestamp("2024-01-20")


def test_an_empty_frame_is_returned_unchanged() -> None:
    panel = _panel(["2024-01-25"])
    assert asof_join_fundamentals(panel, pd.DataFrame()).equals(panel)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_a_restatement_reaches_the_panel_on_the_right_day() -> None:
    """Both fixes together: the revision is computed, and it lands the session
    after it was filed."""
    fund = trailing_twelve_months(_four_quarters(restate=("2024-03-01", 150)))
    joined = asof_join_fundamentals(
        _panel(["2024-02-15", "2024-03-01", "2024-03-04"]), fund)
    got = joined["raw_revenue"].tolist()
    assert got[0] == pytest.approx(460.0), "pre-restatement figure wrong"
    assert got[1] == pytest.approx(460.0), "restatement leaked on its filing date"
    assert got[2] == pytest.approx(510.0), "restatement never arrived"


def test_no_row_ever_sees_a_figure_filed_later_than_its_date() -> None:
    """The invariant the whole join exists to hold."""
    fund = trailing_twelve_months(_four_quarters(restate=("2024-03-01", 150)))
    dates = pd.bdate_range("2023-06-01", "2024-06-01")
    joined = asof_join_fundamentals(_panel(dates), fund)
    have = joined.dropna(subset=["fund_filed"])
    assert (pd.to_datetime(have["fund_filed"]) < pd.to_datetime(have["date"])).all()


def test_values_are_never_invented() -> None:
    fund = trailing_twelve_months(_four_quarters())
    joined = asof_join_fundamentals(_panel(pd.bdate_range("2023-01-01", "2024-06-01")),
                                    fund)
    assert set(joined["raw_revenue"].dropna().unique()) <= {460.0}
    assert np.isnan(joined["raw_revenue"].iloc[0])
