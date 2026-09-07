"""A model should say which code trained it.

Everything else in this pipeline gained provenance over the last few days: the
snapshot hashes its five external inputs, the run manifest records the commit
and whether the tree was dirty, outputs are hashed at write time, and published
metrics are pinned to the artifacts they came from. The model bundle was the
gap. It recorded ``run_id`` and ``snapshot_dir`` but not the revision, so the
deployed artifact could not answer the one question that matters when a number
looks wrong: *was this trained before or after the fix?*

That question came up repeatedly. The model deployed until 2026-09-07 was built
on 21 August and predated the cohort look-ahead fix, the execution-panel
authority fix, recycled-symbol cleaning, the disposal fixes and the feature-grid
fix — and none of that was recoverable from the artifact itself. It had to be
inferred from a file timestamp.

``git_dirty`` is recorded alongside, because a commit hash from a dirty tree
names code that was never committed anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_metadata_builder_records_the_revision() -> None:
    import inspect

    from stock_predictor import cli

    src = inspect.getsource(cli.build_model_meta)
    assert "git_commit" in src, "the model bundle records no code revision"
    assert "git_dirty" in src, (
        "a commit from a dirty tree names code that was never committed")


def test_the_revision_comes_from_git_not_from_a_flag() -> None:
    """Read from the repository, so it cannot be passed in wrong."""
    import inspect

    from stock_predictor import cli

    src = inspect.getsource(cli.build_model_meta)
    assert "git_revision" in src or "repro." in src


def test_a_real_revision_is_resolvable_here() -> None:
    from stock_predictor import repro

    rev = repro.git_revision()
    assert rev.get("commit"), "no commit resolvable in this checkout"
    assert len(rev["commit"]) == 40
    assert isinstance(rev.get("dirty"), bool)


@pytest.mark.skipif(not (ROOT / "artifacts" / "model.meta.json").exists(),
                    reason="no deployed model")
def test_the_deployed_model_is_readable() -> None:
    """The deployed artifact predates this field, so it is absent rather than
    wrong. Recorded here so the gap is visible rather than assumed closed: the
    next deployment carries it, this one cannot."""
    meta = json.loads((ROOT / "artifacts" / "model.meta.json").read_text())
    assert "run_id" in meta
    assert "feature_cols" in meta and "horizon" in meta
