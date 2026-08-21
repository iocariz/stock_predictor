"""Promoting a model is a decision, not a side effect of training.

`train-full` wrote straight to the deployed path, so a retrain replaced the
model being traded the moment it finished — with no check that the new one was
loadable, fresh, or even the right horizon. A stray `train-full` in this repo's
own history came within a download phase of doing exactly that.

Promotion is now explicit and validated, and the outgoing model is archived so
a bad promotion is reversible.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from stock_predictor.deploy import PromotionError, promote_model

SESSIONS = pd.bdate_range(end="2026-08-21", periods=300)


def _write(path: Path, *, horizon: int = 63, train_end: str = "2024-12-31",
           features=("ret_1d",)) -> Path:
    meta = {"feature_cols": list(features), "horizon": horizon,
            "train_end": train_end, "objective": "rank"}
    path.write_bytes(pickle.dumps({"model": object(), "meta": meta}))
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_valid_candidate_is_promoted(tmp_path) -> None:
    cand = _write(tmp_path / "candidate.pkl")
    live = tmp_path / "model.pkl"
    res = promote_model(cand, live, archive_dir=tmp_path / "archive",
                        sessions=SESSIONS)
    assert live.exists()
    assert res.deployed == live
    assert live.read_bytes() == cand.read_bytes()


def test_the_metadata_travels_with_the_model(tmp_path) -> None:
    cand = _write(tmp_path / "candidate.pkl", horizon=21)
    live = tmp_path / "model.pkl"
    promote_model(cand, live, archive_dir=tmp_path / "archive", sessions=SESSIONS)
    assert json.loads(live.with_suffix(".meta.json").read_text())["horizon"] == 21


def test_the_outgoing_model_is_archived(tmp_path) -> None:
    """A bad promotion has to be reversible."""
    live = _write(tmp_path / "model.pkl", horizon=10, train_end="2022-12-31")
    old = live.read_bytes()
    cand = _write(tmp_path / "candidate.pkl")
    res = promote_model(cand, live, archive_dir=tmp_path / "archive",
                        sessions=SESSIONS)
    assert res.archived is not None and res.archived.exists()
    assert res.archived.read_bytes() == old
    assert res.archived.with_suffix(".meta.json").exists()


def test_promoting_with_nothing_deployed_archives_nothing(tmp_path) -> None:
    cand = _write(tmp_path / "candidate.pkl")
    res = promote_model(cand, tmp_path / "model.pkl",
                        archive_dir=tmp_path / "archive", sessions=SESSIONS)
    assert res.archived is None


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


def test_a_missing_candidate_is_refused(tmp_path) -> None:
    with pytest.raises(PromotionError, match="not found"):
        promote_model(tmp_path / "nope.pkl", tmp_path / "model.pkl",
                      archive_dir=tmp_path / "archive", sessions=SESSIONS)


def test_an_unloadable_candidate_is_refused(tmp_path) -> None:
    bad = tmp_path / "candidate.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(PromotionError):
        promote_model(bad, tmp_path / "model.pkl",
                      archive_dir=tmp_path / "archive", sessions=SESSIONS)


def test_a_candidate_without_the_required_metadata_is_refused(tmp_path) -> None:
    bad = tmp_path / "candidate.pkl"
    bad.write_bytes(pickle.dumps({"model": object(), "meta": {"horizon": 63}}))
    with pytest.raises(PromotionError, match="feature_cols"):
        promote_model(bad, tmp_path / "model.pkl",
                      archive_dir=tmp_path / "archive", sessions=SESSIONS)


def test_a_stale_candidate_is_refused(tmp_path) -> None:
    cand = _write(tmp_path / "candidate.pkl", train_end="2015-01-01")
    with pytest.raises(PromotionError, match="stale"):
        promote_model(cand, tmp_path / "model.pkl",
                      archive_dir=tmp_path / "archive", sessions=SESSIONS)


def test_a_horizon_mismatch_is_refused(tmp_path) -> None:
    """The exact shape of the live defect this repo shipped: a horizon-10
    model traded on a 63-day holding rule."""
    cand = _write(tmp_path / "candidate.pkl", horizon=10)
    with pytest.raises(PromotionError, match="horizon"):
        promote_model(cand, tmp_path / "model.pkl",
                      archive_dir=tmp_path / "archive",
                      expected_horizon=63, sessions=SESSIONS)


def test_a_matching_horizon_passes(tmp_path) -> None:
    cand = _write(tmp_path / "candidate.pkl", horizon=63)
    promote_model(cand, tmp_path / "model.pkl", archive_dir=tmp_path / "archive",
                  expected_horizon=63, sessions=SESSIONS)


def test_a_refused_promotion_leaves_the_deployed_model_alone(tmp_path) -> None:
    """The property that makes this safe to run from a cron."""
    live = _write(tmp_path / "model.pkl", horizon=63)
    keep = live.read_bytes()
    cand = _write(tmp_path / "candidate.pkl", train_end="2015-01-01")
    with pytest.raises(PromotionError):
        promote_model(cand, live, archive_dir=tmp_path / "archive",
                      sessions=SESSIONS)
    assert live.read_bytes() == keep


def test_force_overrides_the_checks_but_still_archives(tmp_path) -> None:
    live = _write(tmp_path / "model.pkl", horizon=63)
    cand = _write(tmp_path / "candidate.pkl", train_end="2015-01-01")
    res = promote_model(cand, live, archive_dir=tmp_path / "archive",
                        sessions=SESSIONS, force=True)
    assert live.read_bytes() == cand.read_bytes()
    assert res.archived is not None
    assert res.findings, "force proceeds, but the findings are still reported"
