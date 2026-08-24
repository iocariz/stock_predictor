"""One dict cannot answer three different questions.

`predict.py` built a single `latest_prices` from `adj_close.ffill().iloc[-1]`
and handed it to all three consumers:

* the **kill switch**, which values the book — forward fill is right here, and
  `specs.md:248` permits it for valuation
* `stale_positions`, which warns about holdings with no live quote
* `generate_orders`, which **executes**

Forward fill served the first and silently broke the other two. A holding whose
last real print was $41 three sessions ago arrived at order generation as a
clean $41, so `valid_quote()` — added specifically to refuse missing quotes —
never saw a missing quote, and `stale_positions` saw a populated dictionary and
said nothing.

So the roles are separated at the source: exact final-session quotes execute,
forward-filled marks value, and the gap between them is reported rather than
hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.quotes import (
    describe_quote_gaps,
    execution_quotes,
    last_quote_dates,
    quote_ages,
    valuation_marks,
)

SESSIONS = pd.bdate_range("2026-08-10", periods=5)


def _panel() -> pd.DataFrame:
    """LIVE prints every session; GAPPED stopped after the second."""
    return pd.DataFrame(
        {
            "LIVE": [10.0, 11.0, 12.0, 13.0, 14.0],
            "GAPPED": [40.0, 41.0, np.nan, np.nan, np.nan],
        },
        index=SESSIONS,
    )


# ---------------------------------------------------------------------------
# Execution: the exact final session, or nothing
# ---------------------------------------------------------------------------


def test_a_name_with_no_final_session_print_has_no_execution_quote() -> None:
    """The reported defect, stated as a requirement."""
    q = execution_quotes(_panel())
    assert q["LIVE"] == pytest.approx(14.0)
    assert "GAPPED" not in q


def test_the_last_historical_print_is_never_offered_for_execution() -> None:
    assert execution_quotes(_panel()).get("GAPPED") != 41.0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_degenerate_final_print_is_not_a_quote(bad: float) -> None:
    panel = _panel()
    panel.loc[SESSIONS[-1], "LIVE"] = bad
    assert "LIVE" not in execution_quotes(panel)


def test_an_empty_panel_yields_no_quotes() -> None:
    assert execution_quotes(pd.DataFrame()) == {}


# ---------------------------------------------------------------------------
# Valuation: forward fill is correct here, and only here
# ---------------------------------------------------------------------------


def test_valuation_marks_carry_the_last_observed_price() -> None:
    """Without this the kill switch cannot see a loss on an unquoted holding."""
    m = valuation_marks(_panel())
    assert m["GAPPED"] == pytest.approx(41.0)
    assert m["LIVE"] == pytest.approx(14.0)


def test_a_name_that_never_printed_has_no_mark_either() -> None:
    panel = _panel()
    panel["NEVER"] = np.nan
    assert "NEVER" not in valuation_marks(panel)
    assert "NEVER" not in execution_quotes(panel)


def test_marks_and_quotes_agree_when_nothing_is_stale() -> None:
    panel = _panel()[["LIVE"]]
    assert valuation_marks(panel) == execution_quotes(panel)


# ---------------------------------------------------------------------------
# The gap is reported, not hidden
# ---------------------------------------------------------------------------


def test_age_counts_sessions_since_the_last_real_print() -> None:
    ages = quote_ages(_panel())
    assert ages["LIVE"] == 0
    assert ages["GAPPED"] == 3, "printed on session 2 of 5"


def test_a_name_that_never_printed_is_infinitely_stale() -> None:
    panel = _panel()
    panel["NEVER"] = np.nan
    assert quote_ages(panel)["NEVER"] == np.iinfo(np.int64).max


def test_the_operator_is_told_the_date_and_the_age() -> None:
    panel = _panel()
    text = describe_quote_gaps(
        ["GAPPED"], quote_ages(panel), valuation_marks(panel),
        last_quote_dates(panel), last_session=SESSIONS[-1],
    )
    assert "GAPPED" in text
    assert "3" in text                    # sessions stale
    assert "2026-08-11" in text           # the date of the last real print
    assert "41" in text                   # the price it is marked at


def test_nothing_stale_says_nothing() -> None:
    assert describe_quote_gaps([], {}, {}, {}, last_session=SESSIONS[-1]) == ""
