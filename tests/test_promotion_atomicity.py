"""Promotion switches one pointer, so there is no in-between to crash in.

``specs.md:511`` — *"Promotion MUST be atomic. A crash MUST NOT leave the model
and metadata from different versions."*

The previous implementation staged both files and then issued two consecutive
``os.replace`` calls. Each rename is atomic on its own, and the comment said so
while admitting the gap; between them the deployed pair is mixed, and the
handler cleaned up temp files without restoring the first rename:

    PromotionError {'model': 'new', 'meta': 'old'}

Narrowing that window is not the same as closing it. A release is now an
immutable directory containing both files, and promotion swaps a single
symlink; the visible paths resolve through it, so one atomic rename moves both.

The load-bearing test here is :func:`test_no_injected_failure_can_mix_versions`
— it injects a failure at *every* filesystem call in turn and asserts the pair
on disk is internally consistent each time.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from stock_predictor.deploy import (
    PromotionError,
    current_release,
    list_releases,
    promote_model,
    rollback_release,
)

META = {"horizon": 63, "fitted_through": "2026-06-30", "feature_cols": ["a"]}


def _model():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    return lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(
        rng.normal(size=(200, 2)), rng.integers(0, 2, 200))


def _write(path: Path, tag: str) -> Path:
    with open(path, "wb") as f:
        pickle.dump({"model": _model(), "meta": {**META, "tag": tag}}, f)
    path.with_suffix(".meta.json").write_text(json.dumps({**META, "tag": tag}))
    return path


def _deployed_tags(live: Path) -> tuple[str | None, str | None]:
    """(tag inside the pickle, tag in the sidecar) as currently on disk."""
    model_tag = meta_tag = None
    if live.exists():
        with open(live, "rb") as f:
            model_tag = pickle.load(f)["meta"]["tag"]
    sidecar = live.with_suffix(".meta.json")
    if sidecar.exists():
        meta_tag = json.loads(sidecar.read_text())["tag"]
    return model_tag, meta_tag


@pytest.fixture
def rig(tmp_path):
    live = _write(tmp_path / "model.pkl", "old")
    cand = _write(tmp_path / "model_candidate.pkl", "new")
    return tmp_path, live, cand


def _promote(tmp_path, cand, live, **kw):
    return promote_model(cand, live, archive_dir=tmp_path / "arch", **kw)


# ---------------------------------------------------------------------------
# The happy path still works
# ---------------------------------------------------------------------------


def test_promotion_installs_the_new_bundle(rig) -> None:
    tmp_path, live, cand = rig
    _promote(tmp_path, cand, live)
    assert _deployed_tags(live) == ("new", "new")


def test_the_bundle_is_readable_through_the_deployed_path(rig) -> None:
    """Whatever the mechanism, `--model artifacts/model.pkl` must still load."""
    from stock_predictor.predict import load_model

    tmp_path, live, cand = rig
    _promote(tmp_path, cand, live)
    _, meta = load_model(live)
    assert meta["tag"] == "new"


def test_a_first_promotion_with_nothing_deployed_works(tmp_path) -> None:
    cand = _write(tmp_path / "model_candidate.pkl", "first")
    live = tmp_path / "model.pkl"
    promote_model(cand, live, archive_dir=tmp_path / "arch")
    assert _deployed_tags(live) == ("first", "first")


def test_successive_promotions_each_land(tmp_path) -> None:
    live = tmp_path / "model.pkl"
    for tag in ("v1", "v2", "v3"):
        cand = _write(tmp_path / "model_candidate.pkl", tag)
        promote_model(cand, live, archive_dir=tmp_path / "arch")
        assert _deployed_tags(live) == (tag, tag)


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------


def _count_fs_calls(tmp_path, cand, live) -> int:
    n = {"c": 0}
    real = os.replace

    def counting(src, dst):
        n["c"] += 1
        return real(src, dst)

    with patch("stock_predictor.deploy.os.replace", counting):
        _promote(tmp_path, cand, live)
    return n["c"]


def test_no_injected_failure_can_mix_versions(tmp_path) -> None:
    """Fail at every rename in turn; the pair must never be half-updated.

    This is the test the old implementation could not pass: failing at its
    second rename left the new model beside the old metadata.
    """
    probe_live = _write(tmp_path / "probe.pkl", "old")
    probe_cand = _write(tmp_path / "probe_candidate.pkl", "new")
    total = _count_fs_calls(tmp_path, probe_cand, probe_live)
    assert total >= 2, "a single-rename install would make this test vacuous"

    for fail_at in range(1, total + 1):
        case = tmp_path / f"case{fail_at}"
        case.mkdir()
        live = _write(case / "model.pkl", "old")
        cand = _write(case / "model_candidate.pkl", "new")

        seen = {"c": 0}
        real = os.replace

        def flaky(src, dst, _seen=seen, _at=fail_at):
            _seen["c"] += 1
            if _seen["c"] == _at:
                raise OSError(f"simulated failure at rename {_at}")
            return real(src, dst)

        with patch("stock_predictor.deploy.os.replace", flaky):
            try:
                promote_model(cand, live, archive_dir=case / "arch")
            except PromotionError:
                pass

        model_tag, meta_tag = _deployed_tags(live)
        assert model_tag == meta_tag, (
            f"failure at rename {fail_at} left model={model_tag!r} "
            f"meta={meta_tag!r} — different versions on disk"
        )
        assert model_tag in ("old", "new")


def test_a_failure_partway_leaves_a_loadable_model(tmp_path) -> None:
    """Consistent is not enough; it also has to still be usable."""
    from stock_predictor.predict import load_model

    live = _write(tmp_path / "model.pkl", "old")
    cand = _write(tmp_path / "model_candidate.pkl", "new")

    seen = {"c": 0}
    real = os.replace

    def flaky(src, dst):
        seen["c"] += 1
        if seen["c"] == 1:
            raise OSError("simulated")
        return real(src, dst)

    with patch("stock_predictor.deploy.os.replace", flaky):
        with pytest.raises(PromotionError):
            promote_model(cand, live, archive_dir=tmp_path / "arch")

    _, meta = load_model(live)
    assert meta["tag"] == "old"


# ---------------------------------------------------------------------------
# Releases are immutable and the previous one is still there
# ---------------------------------------------------------------------------


def test_the_previous_release_survives_a_new_promotion(tmp_path) -> None:
    """Rollback is only possible if the old bundle was not overwritten."""
    live = tmp_path / "model.pkl"
    first = _write(tmp_path / "model_candidate.pkl", "v1")
    r1 = promote_model(first, live, archive_dir=tmp_path / "arch")

    second = _write(tmp_path / "model_candidate.pkl", "v2")
    r2 = promote_model(second, live, archive_dir=tmp_path / "arch")

    assert r1.release != r2.release
    assert r1.release.exists(), "the superseded release must remain on disk"
    with open(r1.release / "model.pkl", "rb") as f:
        assert pickle.load(f)["meta"]["tag"] == "v1"


def test_a_legacy_plain_file_deployment_is_migrated_without_a_mixed_state(
    tmp_path,
) -> None:
    """Existing installs have real files at the deployed paths, not links."""
    live = _write(tmp_path / "model.pkl", "legacy")
    assert not live.is_symlink()

    cand = _write(tmp_path / "model_candidate.pkl", "new")
    promote_model(cand, live, archive_dir=tmp_path / "arch")
    assert _deployed_tags(live) == ("new", "new")


# ---------------------------------------------------------------------------
# Rollback is a supported operation, not a shell incantation
# ---------------------------------------------------------------------------


def test_rollback_restores_the_previous_release(tmp_path) -> None:
    live = tmp_path / "model.pkl"
    r1 = promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                       archive_dir=tmp_path / "arch")
    promote_model(_write(tmp_path / "c.pkl", "v2"), live,
                  archive_dir=tmp_path / "arch")
    assert _deployed_tags(live) == ("v2", "v2")

    rollback_release(live, r1.release)
    assert _deployed_tags(live) == ("v1", "v1")


def test_rollback_moves_both_paths_together(tmp_path) -> None:
    """Same guarantee as promotion: one pointer, so never a mixed pair."""
    live = tmp_path / "model.pkl"
    r1 = promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                       archive_dir=tmp_path / "arch")
    promote_model(_write(tmp_path / "c.pkl", "v2"), live,
                  archive_dir=tmp_path / "arch")
    rollback_release(live, r1.release)
    model_tag, meta_tag = _deployed_tags(live)
    assert model_tag == meta_tag == "v1"


def test_rollback_refuses_an_unloadable_release(tmp_path) -> None:
    """Rolling back usually means something is already wrong; a second broken
    artifact does not help."""
    live = tmp_path / "model.pkl"
    promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                  archive_dir=tmp_path / "arch")
    broken = tmp_path / "releases" / "broken"
    broken.mkdir(parents=True)
    (broken / "model.pkl").write_bytes(b"not a pickle")
    (broken / "model.meta.json").write_text("{}")
    with pytest.raises(PromotionError, match="not loadable"):
        rollback_release(live, broken)


def test_rollback_refuses_a_missing_release(tmp_path) -> None:
    live = tmp_path / "model.pkl"
    promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                  archive_dir=tmp_path / "arch")
    with pytest.raises(PromotionError, match="no such release"):
        rollback_release(live, tmp_path / "releases" / "nope")


def test_rollback_refuses_a_horizon_mismatch(tmp_path) -> None:
    live = tmp_path / "model.pkl"
    r1 = promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                       archive_dir=tmp_path / "arch")
    with pytest.raises(PromotionError, match="horizon"):
        rollback_release(live, r1.release, expected_horizon=21)


def test_releases_are_listed_newest_first(tmp_path) -> None:
    live = tmp_path / "model.pkl"
    for tag in ("v1", "v2", "v3"):
        promote_model(_write(tmp_path / "c.pkl", tag), live,
                      archive_dir=tmp_path / "arch")
    releases = list_releases(live)
    assert len(releases) == 3
    assert releases == sorted(releases, key=lambda p: p.name, reverse=True)
    assert current_release(live) == releases[0].resolve()


def test_mv_would_have_corrupted_the_release_which_is_why_this_exists(tmp_path) -> None:
    """A symlink pointing at a directory swallows ``mv``: the replacement lands
    *inside* the release instead of over the pointer, leaving the old version
    live and writing into a directory that is meant to be immutable. Pin the
    behaviour so nobody documents ``mv`` as the rollback command."""
    import subprocess

    live = tmp_path / "model.pkl"
    r1 = promote_model(_write(tmp_path / "c.pkl", "v1"), live,
                       archive_dir=tmp_path / "arch")
    r2 = promote_model(_write(tmp_path / "c.pkl", "v2"), live,
                       archive_dir=tmp_path / "arch")

    tmp_link = tmp_path / "ptr.tmp"
    os.symlink(os.path.relpath(r1.release, tmp_path), tmp_link)
    subprocess.run(["mv", "-f", str(tmp_link), str(tmp_path / ".current_release")],
                   check=True)

    assert _deployed_tags(live) == ("v2", "v2"), "mv did not roll anything back"
    assert (r2.release / "ptr.tmp").is_symlink(), "it wrote into the release"

    # The supported path does the right thing.
    (r2.release / "ptr.tmp").unlink()
    rollback_release(live, r1.release)
    assert _deployed_tags(live) == ("v1", "v1")
