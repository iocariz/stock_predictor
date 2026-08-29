"""One ticker, one score, one signal date.

``specs.md:240`` — *"Duplicate ticker scores on one signal date MUST fail
validation."* Nothing validated it, and two things went wrong quietly.

Selection took rows, not names, so a duplicated row became a duplicated
holding::

    picks  : ('AAA', 'AAA')
    weights: [0.5, 0.5]

Half the book in one name, described as two positions. Position sizing, the
per-name risk the top-N is meant to spread, and every diagnostic that counts
holdings are all wrong together, and nothing in the output says so.

The cross-section floor counted rows too, so five copies of one name satisfied
a floor of five. The floor exists to refuse dates too thin to rank; duplicates
let a date with *one* tradable name pass as if it had five.

The current baseline is clean — 952,329 rows, 952,329 unique pairs — so this is
a guard against a future panel, a merge that fans out, or a hand-built input,
rather than a fix to a number.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.execution import SelectionRules, eligible_candidates

DATES = pd.bdate_range("2024-01-01", periods=6)


def _day(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": t, "prob": p, "adj_close": px} for t, p, px in rows
    ])


def _panel(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "ticker": t, "prob": p, "adj_close": px}
        for d in DATES for t, p, px in rows
    ])


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_a_duplicated_ticker_is_refused() -> None:
    """The reported reproduction, stated as a requirement."""
    day = _day([("AAA", 0.9, 100.0), ("AAA", 0.9, 100.0), ("BBB", 0.5, 50.0)])
    with pytest.raises(ValueError, match="duplicate"):
        eligible_candidates(day, SelectionRules(top_n=2))


def test_the_error_names_the_offender() -> None:
    day = _day([("AAA", 0.9, 100.0), ("AAA", 0.8, 100.0), ("BBB", 0.5, 50.0)])
    with pytest.raises(ValueError, match="AAA"):
        eligible_candidates(day, SelectionRules(top_n=2))


def test_duplicates_are_refused_even_with_different_scores() -> None:
    """Same name twice is invalid whether or not the scores agree; picking one
    would be guessing which row the model meant."""
    day = _day([("AAA", 0.9, 100.0), ("AAA", 0.1, 100.0)])
    with pytest.raises(ValueError, match="duplicate"):
        eligible_candidates(day, SelectionRules(top_n=1))


def test_a_clean_cross_section_is_unaffected() -> None:
    day = _day([("AAA", 0.9, 100.0), ("BBB", 0.5, 50.0), ("CCC", 0.2, 20.0)])
    picks = eligible_candidates(day, SelectionRules(top_n=2))
    assert [c.ticker for c in picks[:2]] == ["AAA", "BBB"]


def test_duplicates_cannot_prop_up_the_cross_section_floor() -> None:
    """Five copies of one name is a one-name cross-section, not a five-name one."""
    day = _day([("AAA", 0.9, 100.0)] * 5)
    with pytest.raises(ValueError, match="duplicate"):
        eligible_candidates(day, SelectionRules(top_n=1, min_cross_section=5))


def test_an_empty_day_is_still_just_empty() -> None:
    assert eligible_candidates(_day([]), SelectionRules(top_n=2)) == []


# ---------------------------------------------------------------------------
# Panel ingestion
# ---------------------------------------------------------------------------


def test_a_panel_with_duplicate_pairs_is_refused() -> None:
    from stock_predictor.backtest import _prepare_scored

    panel = _panel([("AAA", 0.9, 100.0), ("AAA", 0.9, 100.0), ("BBB", 0.5, 50.0)])
    with pytest.raises(ValueError, match="duplicate"):
        _prepare_scored(panel, None)


def test_the_panel_error_names_a_date_and_ticker() -> None:
    from stock_predictor.backtest import _prepare_scored

    panel = _panel([("AAA", 0.9, 100.0), ("AAA", 0.9, 100.0)])
    with pytest.raises(ValueError, match="AAA"):
        _prepare_scored(panel, None)


def test_a_clean_panel_loads() -> None:
    from stock_predictor.backtest import _prepare_scored

    panel = _panel([("AAA", 0.9, 100.0), ("BBB", 0.5, 50.0)])
    df, dates, prices, actual = _prepare_scored(panel, None)
    assert len(dates) == len(DATES)
    assert set(prices.columns) == {"AAA", "BBB"}


def test_the_same_ticker_on_different_dates_is_fine() -> None:
    """Only *within* a signal date is a repeat a duplicate."""
    from stock_predictor.backtest import _prepare_scored

    panel = _panel([("AAA", 0.9, 100.0), ("BBB", 0.5, 50.0)])
    assert panel["ticker"].value_counts()["AAA"] == len(DATES)
    _prepare_scored(panel, None)          # must not raise
