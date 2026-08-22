"""The live universe must be the one the model was fitted on.

Training samples from the union covering the whole training window; inference
sampled from a 400-day window. The same seed drawing from different
populations is a different draw:

    sample_n=500  training draw 500 | live draw 500 | overlap 307
                  current members: 305 in training, 473 live, 289 shared

Every cross-sectional feature is a rank within the universe, so a different
universe means different feature values for the same stock. `sample_n=10000`
draws everything and hides this, which is why it stayed latent.

`sample_mismatch_warning` compared sample *sizes* and would say nothing about
500 versus 500. The fix is to record what was actually drawn.
"""

from __future__ import annotations

import pytest

from stock_predictor.universe import (
    resolve_live_universe,
    universe_drift,
    universe_hash,
)

TRAIN = [f"T{i:02d}" for i in range(20)]          # what the model saw
CURRENT = {f"T{i:02d}" for i in range(10, 30)}    # today's index members


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_the_hash_is_order_independent() -> None:
    assert universe_hash(["B", "A"]) == universe_hash(["A", "B"])


def test_a_different_universe_hashes_differently() -> None:
    assert universe_hash(["A", "B"]) != universe_hash(["A", "C"])


def test_the_hash_is_stable_across_processes() -> None:
    import subprocess
    import sys

    code = ("from stock_predictor.universe import universe_hash;"
            "print(universe_hash(['AAA','BBB']))")
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True,
                       env={"PYTHONHASHSEED": s, "PATH": "/usr/bin:/bin"}).stdout.strip()
        for s in ("0", "1", "999")
    }
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Resolving the live universe
# ---------------------------------------------------------------------------


def test_the_recorded_universe_is_intersected_with_current_membership() -> None:
    """Trade what the model knows and the index still holds."""
    out = resolve_live_universe(TRAIN, CURRENT)
    assert set(out) == {f"T{i:02d}" for i in range(10, 20)}
    assert out == sorted(out), "stable order so the draw is reproducible"


def test_names_the_model_never_saw_are_excluded() -> None:
    """A member added after training was never in its cross-section; including
    it silently changes every rank the model reads."""
    assert "T25" not in resolve_live_universe(TRAIN, CURRENT)


def test_names_that_left_the_index_are_excluded() -> None:
    assert "T00" not in resolve_live_universe(TRAIN, CURRENT)


def test_an_empty_intersection_is_reported_not_returned_silently() -> None:
    with pytest.raises(ValueError, match="no overlap"):
        resolve_live_universe(["A", "B"], {"C", "D"})


# ---------------------------------------------------------------------------
# Drift reporting
# ---------------------------------------------------------------------------


def test_drift_counts_what_is_missed_and_what_is_dropped() -> None:
    d = universe_drift(TRAIN, CURRENT)
    assert d["tradable"] == 10
    assert d["current_not_in_training"] == 10   # T20..T29
    assert d["training_no_longer_current"] == 10  # T00..T09
    assert d["coverage_of_current"] == pytest.approx(0.5)


def test_no_drift_when_the_universe_still_matches() -> None:
    d = universe_drift(TRAIN, set(TRAIN))
    assert d["current_not_in_training"] == 0
    assert d["coverage_of_current"] == 1.0


def test_drift_is_reported_even_when_it_is_total() -> None:
    d = universe_drift(["A"], {"B", "C"})
    assert d["tradable"] == 0
    assert d["coverage_of_current"] == 0.0


# ---------------------------------------------------------------------------
# The failure this reproduces
# ---------------------------------------------------------------------------


def test_reseeding_a_smaller_population_does_not_reproduce_the_draw() -> None:
    """The mechanism itself: same seed, different population, different draw."""
    from stock_predictor.universe import sample_tickers

    big = [f"T{i:03d}" for i in range(845)]
    small = big[:531]
    assert sample_tickers(big, 500, seed=42) != sample_tickers(small, 500, seed=42)


def test_recording_the_universe_removes_the_dependence_on_reseeding() -> None:
    """With the draw recorded, the population it came from no longer matters."""
    from stock_predictor.universe import sample_tickers

    big = [f"T{i:03d}" for i in range(845)]
    drawn = sample_tickers(big, 500, seed=42)
    current = set(big[:531])
    a = resolve_live_universe(drawn, current)
    b = resolve_live_universe(drawn, current)
    assert a == b
    assert set(a) <= set(drawn), "never trades a name the model did not see"


# ---------------------------------------------------------------------------
# Reached through the real entry points
# ---------------------------------------------------------------------------


def test_training_metadata_records_the_drawn_universe() -> None:
    import argparse

    from stock_predictor.cli import build_model_meta

    args = argparse.Namespace(
        start="2010-01-01", end=None, train_end="2024-12-31",
        test_start="2025-01-01", sample_n=500, horizon=63, threshold=0.05,
        skip_earnings=True, label_target="raw", seed=42,
    )
    meta = build_model_meta(
        args, feature_cols=["ret_1d"], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        universe=["BBB", "AAA"],
    )
    assert meta["universe"] == ["AAA", "BBB"], "sorted, so it is comparable"
    assert meta["universe_hash"] == universe_hash(["AAA", "BBB"])


def test_a_model_without_a_universe_records_none_not_a_guess() -> None:
    import argparse

    from stock_predictor.cli import build_model_meta

    args = argparse.Namespace(
        start="2010-01-01", end=None, train_end="2024-12-31",
        test_start="2025-01-01", sample_n=500, horizon=63, threshold=0.05,
        skip_earnings=True, label_target="raw", seed=42,
    )
    meta = build_model_meta(
        args, feature_cols=[], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
    )
    assert meta["universe"] is None and meta["universe_hash"] is None
