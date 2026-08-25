"""Units carved out of ``cli.main``.

The training CLI was 32% covered because ~300 of its 340 lines were a single
``main``. What lived in there was not glue: it decided hyperparameters, what
the reproducibility manifest recorded, and what a saved model remembers about
how it was built. All of that is worth pinning.
"""

from __future__ import annotations

import argparse

import pytest

from stock_predictor.cli import (
    DEFAULT_LGBM_PARAMS,
    build_model_meta,
    build_run_extra,
    resolve_model_params,
    validate_train_test_window,
)


def _args(**kw) -> argparse.Namespace:
    base = dict(
        start="2010-01-01", end=None, train_end="2018-12-31",
        test_start="2019-01-01", sample_n=500, min_coverage=0.98, horizon=63,
        threshold=0.05, skip_earnings=True, no_optuna=True, no_macro_merge=False,
        strict_dropna=False, label_target="raw", fundamentals=False, seed=42,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Window validation
# ---------------------------------------------------------------------------


def test_a_valid_window_passes() -> None:
    validate_train_test_window("2018-12-31", "2019-01-01")


@pytest.mark.parametrize(
    ("train_end", "test_start"),
    [("2019-01-01", "2019-01-01"), ("2019-06-01", "2019-01-01")],
)
def test_an_overlapping_window_is_rejected(train_end, test_start) -> None:
    """Training past the test start leaks the period being measured."""
    with pytest.raises(ValueError, match="train-end"):
        validate_train_test_window(train_end, test_start)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


def test_the_defaults_are_a_complete_lgbm_config() -> None:
    assert {"n_estimators", "learning_rate", "num_leaves"} <= set(DEFAULT_LGBM_PARAMS)


def test_no_tuning_leaves_the_defaults_intact() -> None:
    assert resolve_model_params({}) == DEFAULT_LGBM_PARAMS


def test_tuned_values_override_the_defaults() -> None:
    out = resolve_model_params({"learning_rate": 0.01})
    assert out["learning_rate"] == 0.01
    assert out["num_leaves"] == DEFAULT_LGBM_PARAMS["num_leaves"], "untuned keys survive"


def test_resolving_params_does_not_mutate_the_defaults() -> None:
    """A module-level dict handed out by reference would leak tuning from one
    run into the next."""
    resolve_model_params({"learning_rate": 0.01})
    assert DEFAULT_LGBM_PARAMS["learning_rate"] == 0.05


def test_tuning_may_add_keys_the_defaults_lack() -> None:
    assert resolve_model_params({"min_child_samples": 30})["min_child_samples"] == 30


# ---------------------------------------------------------------------------
# The reproducibility manifest
# ---------------------------------------------------------------------------


def test_the_manifest_records_what_changes_a_result() -> None:
    extra = build_run_extra(_args(), objective="rank", tune_metric="topn_excess")
    for key in ("start", "train_end", "test_start", "horizon", "threshold",
                "sample_n", "seed", "objective", "label_target", "optuna_metric",
                "fundamentals", "skip_earnings"):
        assert key in extra, f"{key} changes the result and must be recorded"


def test_the_manifest_records_the_resolved_values_not_the_raw_flags() -> None:
    """``--objective auto`` resolves to something concrete; recording "auto"
    would make the run unreproducible once the default changes."""
    extra = build_run_extra(_args(), objective="rank", tune_metric="topn_ir")
    assert extra["objective"] == "rank"
    assert extra["optuna_metric"] == "topn_ir"


def test_fundamentals_are_recorded_as_a_plain_bool() -> None:
    extra = build_run_extra(_args(fundamentals="some/path"), objective="rank",
                            tune_metric="auto")
    assert extra["fundamentals"] is True


# ---------------------------------------------------------------------------
# Saved-model metadata
# ---------------------------------------------------------------------------


def test_saved_metadata_carries_what_scoring_needs() -> None:
    """predict.load_model rejects metadata without these two."""
    meta = build_model_meta(
        _args(), feature_cols=["ret_1d"], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params=DEFAULT_LGBM_PARAMS, n_trees=100,
        importance={"ret_1d": 1.0}, pr_auc=0.5, roc_auc=0.6,
        run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
    )
    assert meta["feature_cols"] == ["ret_1d"]
    assert meta["horizon"] == 63


def test_saved_metadata_carries_the_seed_and_sample_size() -> None:
    """The live path reuses both to rebuild the same universe."""
    meta = build_model_meta(
        _args(seed=7, sample_n=250), feature_cols=["ret_1d"], objective="rank",
        tune_metric="auto", optuna_best={}, manual_params={}, n_trees=1,
        importance={}, pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
    )
    assert meta["seed"] == 7
    assert meta["sample_n"] == 250


def test_a_run_without_a_snapshot_records_none_not_a_broken_path() -> None:
    meta = build_model_meta(
        _args(), feature_cols=[], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
    )
    assert meta["snapshot_dir"] is None


def test_metadata_round_trips_through_the_live_loader(tmp_path) -> None:
    """The two modules agree on the contract, not just on their own tests."""
    import pickle

    from stock_predictor.predict import load_model, resolve_universe_seed

    meta = build_model_meta(
        _args(seed=9), feature_cols=["ret_1d"], objective="rank",
        tune_metric="auto", optuna_best={}, manual_params={}, n_trees=1,
        importance={}, pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
    )
    p = tmp_path / "m.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"model": "M", "meta": meta}, fh)

    _, loaded = load_model(p)
    assert resolve_universe_seed(None, loaded) == 9
