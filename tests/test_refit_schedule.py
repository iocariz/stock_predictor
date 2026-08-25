"""A scheduled retrain has to actually learn something new.

Three faults, all mine, all from the same PR:

* training writes ``model_candidate.pkl`` while CI uploaded ``model.pkl`` with
  ``if-no-files-found: error`` — on a clean runner the monthly job fails;
* ``TRAIN_END`` defaulted to a *fixed* ``2024-12-31``, so every monthly run
  refit the same window and the cron learned nothing;
* purging removes another ``horizon`` sessions before ``test_start``, so the
  model's last real training signal was **2024-10-01** while its metadata
  reported 2024-12-31 — and the freshness gate reads the metadata.

Evaluation and production refit want different windows: one holds data back to
measure, the other uses everything it can label. They are separate stages now.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from stock_predictor.cli import build_model_meta, latest_trainable_end

SESSIONS = pd.bdate_range("2024-01-01", periods=500)


def _args(**kw) -> argparse.Namespace:
    base = dict(start="2010-01-01", end=None, train_end="2024-12-31",
                test_start="2025-01-01", sample_n=500, horizon=63,
                threshold=0.05, skip_earnings=True, label_target="raw", seed=42)
    base.update(kw)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# The effective fitted-through date
# ---------------------------------------------------------------------------


def test_metadata_records_what_the_model_actually_learned_through() -> None:
    """train_end is a request; fitted_through is what survived purging."""
    meta = build_model_meta(
        _args(), feature_cols=["ret_1d"], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
        fitted_through=pd.Timestamp("2024-10-01"),
    )
    assert meta["train_end"] == "2024-12-31"
    assert meta["fitted_through"] == "2024-10-01"


def test_fitted_through_is_none_when_unknown_rather_than_guessed() -> None:
    meta = build_model_meta(
        _args(), feature_cols=[], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=0.5, roc_auc=0.6, run_id="r1", snapshot_root=None,
        train_end="2024-12-31", test_start="2025-01-01",
    )
    assert meta["fitted_through"] is None


def test_the_freshness_gate_prefers_fitted_through() -> None:
    """The gate reported 1.64 years using train_end when the last real signal
    was a further quarter back."""
    from stock_predictor.freshness import FreshnessPolicy, check_freshness

    as_of = pd.Timestamp("2026-08-23")
    # A tight limit so both dates trip it and the ages are comparable; at the
    # 2.0-year default the optimistic date sneaks under, which is the bug.
    tight = FreshnessPolicy(max_model_age_years=1.0, max_data_age_sessions=0)
    optimistic = {"train_end": "2024-12-31", "horizon": 63}
    honest = {**optimistic, "fitted_through": "2024-10-01"}
    a = check_freshness(optimistic, SESSIONS, as_of=as_of, policy=tight)
    b = check_freshness(honest, SESSIONS, as_of=as_of, policy=tight)
    age_a = next(f.value for f in a if f.kind == "model_age")
    age_b = next(f.value for f in b if f.kind == "model_age")
    assert age_b > age_a, "the honest date must read as older"
    assert age_a == pytest.approx(1.64, abs=0.05)
    assert age_b == pytest.approx(1.89, abs=0.05)


# ---------------------------------------------------------------------------
# A refit window that moves
# ---------------------------------------------------------------------------


def test_the_trainable_end_tracks_the_run_date() -> None:
    """A label needs `horizon` sessions of future, so the newest trainable
    date is that far behind the last session — and it moves."""
    end = latest_trainable_end(SESSIONS, horizon=63)
    assert end == SESSIONS[-64]


def test_a_later_run_trains_through_a_later_date() -> None:
    early = latest_trainable_end(SESSIONS[:400], horizon=63)
    late = latest_trainable_end(SESSIONS, horizon=63)
    assert late > early, "a monthly refit must learn something new"


def test_a_longer_horizon_pulls_the_window_back() -> None:
    assert (latest_trainable_end(SESSIONS, horizon=126)
            < latest_trainable_end(SESSIONS, horizon=63))


def test_too_little_history_is_reported_not_guessed() -> None:
    with pytest.raises(ValueError, match="history"):
        latest_trainable_end(SESSIONS[:10], horizon=63)


def test_the_result_is_an_actual_session() -> None:
    assert latest_trainable_end(SESSIONS, horizon=63) in set(SESSIONS)


# ---------------------------------------------------------------------------
# The scheduled job uploads what training writes
# ---------------------------------------------------------------------------


def test_ci_uploads_the_artifact_training_produces() -> None:
    """Training writes model_candidate.pkl; the job uploaded model.pkl with
    if-no-files-found: error, so a clean runner failed."""
    from pathlib import Path

    wf = (Path(__file__).resolve().parents[1]
          / ".github" / "workflows" / "train-sp500.yml").read_text()
    upload = wf[wf.index("Upload model"):]
    assert "artifacts/model_candidate.pkl" in upload
    assert "\n            artifacts/model.pkl\n" not in upload


def test_the_uploaded_bundle_is_self_consistent() -> None:
    """A model without its scores or its execution panel cannot be verified."""
    from pathlib import Path

    wf = (Path(__file__).resolve().parents[1]
          / ".github" / "workflows" / "train-sp500.yml").read_text()
    upload = wf[wf.index("Upload model"):wf.index("Upload plots")]
    for part in ("model_candidate.pkl", "model_candidate.meta.json",
                 "wf_scored.parquet", "execution_prices.parquet"):
        assert part in upload, f"bundle is missing {part}"


def test_scheduled_runs_refit_rather_than_repeat() -> None:
    from pathlib import Path

    wf = (Path(__file__).resolve().parents[1]
          / ".github" / "workflows" / "train-sp500.yml").read_text()
    assert "REFIT:" in wf and "schedule" in wf
