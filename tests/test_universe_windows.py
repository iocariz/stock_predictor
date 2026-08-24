"""Which historical companies train must not depend on who joined later.

The universe was formed over the whole download window and then sampled. With a
capped `--sample-n` that is transductive leakage: names admitted to the index
*after* the training period compete for slots with the historical ones, so
future membership decides which past companies the model ever sees. Measured on
the real stints at `--sample-n 500`, **352 of the historical names differ**
depending on which population you draw from.

Three universes, three purposes:

* **download** — everything fetched. Must stay wide: recent cross-sections and
  execution prices both need names outside the training window.
* **fitted** — drawn from the population that existed *during training*, so the
  draw depends only on information available then.
* **scoring** — what a live run may trade: fitted names the index still holds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.universe import UniverseSplit, split_universes

STINTS = pd.DataFrame({
    "ticker": ["OLD1", "OLD2", "OLD3", "LATE1", "LATE2"],
    "start_date": [pd.Timestamp("2010-01-01")] * 3 + [pd.Timestamp("2025-06-01")] * 2,
    "end_date": [pd.NaT] * 5,
})
START, TRAIN_END, END = "2010-01-01", "2024-12-31", "2026-08-01"


def _split(**kw) -> UniverseSplit:
    base = dict(stints=STINTS, start=START, train_end=TRAIN_END, end=END,
                sample_n=10_000, seed=42)
    base.update(kw)
    return split_universes(**base)


# ---------------------------------------------------------------------------
# The three populations
# ---------------------------------------------------------------------------


def test_the_fitted_universe_excludes_later_entrants() -> None:
    out = _split()
    assert set(out.fitted) == {"OLD1", "OLD2", "OLD3"}
    assert "LATE1" not in out.fitted


def test_the_download_universe_keeps_them() -> None:
    """Recent cross-sections and execution prices both need the wider set."""
    assert {"LATE1", "LATE2"} <= set(_split().download)


def test_the_download_universe_contains_the_fitted_one() -> None:
    out = _split()
    assert set(out.fitted) <= set(out.download)


def test_later_entrants_are_reported() -> None:
    out = _split()
    assert sorted(out.joined_after_training) == ["LATE1", "LATE2"]


# ---------------------------------------------------------------------------
# The leakage itself
# ---------------------------------------------------------------------------


def test_a_capped_draw_ignores_who_joined_later() -> None:
    """The property that matters: adding future entrants must not change which
    historical names are drawn."""
    out = _split(sample_n=2)
    more_stints = pd.concat([STINTS, pd.DataFrame({
        "ticker": ["LATE3", "LATE4", "LATE5"],
        "start_date": [pd.Timestamp("2026-01-01")] * 3,
        "end_date": [pd.NaT] * 3,
    })])
    after = _split(sample_n=2, stints=more_stints)
    assert set(out.fitted) == set(after.fitted), (
        "future membership changed the historical draw"
    )


def test_the_cap_applies_to_the_fitted_draw() -> None:
    assert len(_split(sample_n=2).fitted) == 2


def test_later_entrants_do_not_consume_fitted_slots() -> None:
    """They are added to the download, not competing for the sample."""
    out = _split(sample_n=2)
    assert len(out.fitted) == 2
    assert {"LATE1", "LATE2"} <= set(out.download)


def test_an_uncapped_draw_takes_the_whole_training_population() -> None:
    assert set(_split().fitted) == {"OLD1", "OLD2", "OLD3"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_the_scoring_universe_is_fitted_names_still_in_the_index() -> None:
    out = _split()
    assert set(out.scoring({"OLD1", "OLD2", "LATE1"})) == {"OLD1", "OLD2"}


def test_a_name_the_model_never_saw_is_not_scorable() -> None:
    assert "LATE1" not in _split().scoring({"LATE1", "OLD1"})


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_everything_comes_back_sorted() -> None:
    out = _split()
    for names in (out.fitted, out.download, out.joined_after_training):
        assert names == sorted(names)


def test_a_training_window_with_nothing_in_it_is_reported() -> None:
    """Before any stint begins. A window *after* they start is not empty:
    an open-ended membership overlaps every later window."""
    with pytest.raises(ValueError, match="training window"):
        _split(start="1990-01-01", train_end="1995-01-01")
