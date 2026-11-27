"""An operator has to be able to say what a corporate action was.

``rebase_to_adjusted_history`` refuses to settle an adjustment of 3% or more on
its own, because a 2:1 split and a spin-off worth half the parent read the
same. It takes the operator's word through ``confirmed`` -- and for a while
nothing passed it. ``predict-sp500`` called it bare and had no flag, so a
pending action was reported every run and could never be settled.

SanDisk split 8:1 on 2026-11-20. The live book was long 1 share from $1,740.87
and went on holding 1 share at ~$79: a $1,662 loss on a leg that had really
lost about $1,107, half a point of drawdown that the kill switch was reading.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.corporate_actions import (
    Adjustment,
    parse_confirmations,
    unapplied_confirmations,
)

SESSION = pd.Timestamp("2026-09-04")


def test_a_confirmation_is_a_ticker_and_a_kind() -> None:
    assert parse_confirmations(["SNDK=split", "CTVA=distribution"]) == {
        "SNDK": "split",
        "CTVA": "distribution",
    }


def test_nothing_confirmed_is_an_empty_mapping() -> None:
    assert parse_confirmations([]) == {}


def test_case_and_padding_do_not_matter() -> None:
    assert parse_confirmations([" sndk = Split "]) == {"SNDK": "split"}


@pytest.mark.parametrize("bad", ["SNDK", "SNDK=", "=split", "SNDK=split=2", ""])
def test_a_malformed_confirmation_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="TICKER=KIND"):
        parse_confirmations([bad])


def test_an_unknown_kind_is_refused() -> None:
    """'dividend' is the tempting wrong word; the code calls it a distribution."""
    with pytest.raises(ValueError, match="distribution.*split|split.*distribution"):
        parse_confirmations(["SNDK=dividend"])


def test_the_same_ticker_cannot_be_two_things() -> None:
    with pytest.raises(ValueError, match="SNDK"):
        parse_confirmations(["SNDK=split", "SNDK=distribution"])


def test_repeating_a_confirmation_is_harmless() -> None:
    assert parse_confirmations(["SNDK=split", "SNDK=split"]) == {"SNDK": "split"}


def test_a_confirmation_that_settled_something_is_applied() -> None:
    notes = (Adjustment("SNDK", SESSION, 0.125, "split", 0.0, 1, 8),)
    assert unapplied_confirmations({"SNDK": "split"}, notes) == ()


def test_a_confirmation_for_a_name_with_nothing_pending_is_reported() -> None:
    """A flag that silently does nothing: a typo in the ticker, a name not
    held, or an action already settled must not look like success."""
    notes = (Adjustment("SNDK", SESSION, 0.125, "split", 0.0, 1, 8),)
    assert unapplied_confirmations({"SNDK": "split", "SNDX": "split"}, notes) == ("SNDX",)


def test_a_leg_left_pending_does_not_count_as_applied() -> None:
    notes = (Adjustment("SNDK", SESSION, 0.125, None),)
    assert unapplied_confirmations({"SNDK": "split"}, notes) == ("SNDK",)
