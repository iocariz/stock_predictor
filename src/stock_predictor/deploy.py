"""Promote a trained model to the deployed path, deliberately.

``train-full`` used to write straight to the model the live path loads, so a
retrain replaced the model being traded the instant it finished — with no check
that the new one was loadable, fresh, or even the right horizon. In this repo's
own history a stray ``train-full`` came within a download phase of doing
exactly that to a live model.

Training now writes a *candidate*. Promotion is a separate, validated step:

* the candidate must load, expose a scoring method, and carry the metadata the
  live path requires — "unpickles without raising" is not the same as "can
  score a cross-section tomorrow morning"
* it must pass the same freshness policy the live run applies
* its horizon must match the holding rule it will be traded on — the exact
  shape of the defect this repo shipped, a horizon-10 model held for 63 days
* the outgoing model is archived, so a bad promotion is reversible
* the swap is **one** atomic operation, so a crash mid-promotion cannot leave a
  live model paired with the previous run's metadata — a mismatch that reads as
  a working deployment (``specs.md:511``)

A release is an immutable directory under ``releases/`` holding both files.
``.current_release`` is a symlink to the live one, and the deployed paths are
symlinks *through* it:

    model.pkl        -> .current_release/model.pkl
    model.meta.json  -> .current_release/model.meta.json
    .current_release -> releases/20260825T145959Z-abc1234/

Promotion renames one symlink. Both visible paths follow it, so there is no
interval in which they disagree. An earlier version staged both files and
issued two consecutive ``os.replace`` calls; each rename is atomic alone, but
failing on the second left the new model beside the old metadata. Narrowing
that window is not the same as closing it.

Superseded releases stay on disk, so rolling back is pointing the symlink at a
directory that is still there — use :func:`rollback_release`, not ``mv``. A
symlink whose target is a directory swallows ``mv``: the replacement lands
*inside* the release rather than over the pointer, silently leaving the old
version live and littering a directory that is meant to be immutable.
``os.replace`` operates on the link itself and does not have this problem.

A refused promotion leaves the deployed model untouched, which is what makes
this safe to run unattended.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_predictor.freshness import Finding, FreshnessPolicy, check_freshness, describe
from stock_predictor.predict import load_model


class PromotionError(RuntimeError):
    """The candidate was not fit to deploy; nothing was changed."""


RELEASES_DIRNAME = "releases"
POINTER_NAME = ".current_release"


@dataclass(frozen=True)
class PromotionResult:
    deployed: Path
    archived: Path | None
    findings: list[Finding]
    release: Path | None = None
    """The immutable directory this promotion installed. Still on disk after
    the next promotion, which is what makes a rollback a symlink swap."""


def _release_id(meta: dict) -> str:
    """Unique, sortable, and traceable back to the run that produced it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = str(meta.get("run_id") or "")[:12]
    return f"{stamp}-{run}" if run else stamp


def _swap_symlink(link: Path, target: str) -> None:
    """Point *link* at *target* in one atomic step.

    ``os.symlink`` cannot overwrite, so the link is built under a temporary
    name and renamed over the old one. ``os.replace`` on a symlink is atomic:
    readers see either the previous target or the new one.
    """
    tmp = link.with_name(link.name + ".swapping")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    try:
        os.replace(tmp, link)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _stage_release(releases_dir: Path, model_src: Path, meta_src: Path,
                   meta: dict) -> Path:
    """Build an immutable release directory, published by one directory rename."""
    releases_dir.mkdir(parents=True, exist_ok=True)
    rid = _release_id(meta)
    final = releases_dir / rid
    staging = releases_dir / f".{rid}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copy2(model_src, staging / "model.pkl")
        shutil.copy2(meta_src, staging / "model.meta.json")
        # Until this rename the release does not exist; after it, it is
        # complete. Nothing ever observes a partially written release.
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _archive(deployed: Path, archive_dir: Path) -> Path | None:
    """Copy the currently deployed model aside, timestamped. ``None`` if absent."""
    if not deployed.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = archive_dir / f"{deployed.stem}_{stamp}{deployed.suffix}"
    shutil.copy2(deployed, target)
    meta = deployed.with_suffix(".meta.json")
    if meta.exists():
        shutil.copy2(meta, target.with_suffix(".meta.json"))
    return target


def promote_model(
    candidate: Path | str,
    deployed: Path | str,
    *,
    archive_dir: Path | str,
    expected_horizon: int | None = None,
    policy: FreshnessPolicy | None = None,
    sessions: pd.DatetimeIndex | None = None,
    releases_dir: Path | str | None = None,
    force: bool = False,
) -> PromotionResult:
    """Validate *candidate* and put it in place of *deployed*.

    *sessions* is the price panel's session index, used for the data-age half
    of the freshness policy; pass ``None`` to check the model age only.
    Raises :class:`PromotionError` without touching anything if a check fails,
    unless *force*, which proceeds but still reports and still archives.
    """
    candidate, deployed = Path(candidate), Path(deployed)
    if not candidate.exists():
        raise PromotionError(f"candidate model not found: {candidate}")

    cand_meta = candidate.with_suffix(".meta.json")
    if not cand_meta.exists():
        # Promoting the model alone would leave the live .meta.json describing
        # the model it replaced: same filenames, different run.
        raise PromotionError(
            f"candidate has no metadata file at {cand_meta}; refusing to "
            "promote a model that would inherit the previous run's metadata"
        )

    try:
        model, meta = load_model(candidate)
    except Exception as exc:  # noqa: BLE001 - any failure to load disqualifies it
        raise PromotionError(f"candidate model is not loadable: {exc}") from exc

    if not any(callable(getattr(model, m, None)) for m in ("predict_proba", "predict")):
        raise PromotionError(
            f"candidate loaded as {type(model).__name__}, which exposes neither "
            "predict_proba nor predict; it cannot score a cross-section"
        )

    if expected_horizon is not None and int(meta.get("horizon", -1)) != expected_horizon:
        message = (
            f"candidate horizon {meta.get('horizon')} does not match the "
            f"holding rule ({expected_horizon}). Trading a "
            f"{meta.get('horizon')}-day signal on a {expected_horizon}-day exit "
            "is the mismatch this check exists to prevent."
        )
        if not force:
            raise PromotionError(message)

    findings = check_freshness(
        meta,
        pd.DatetimeIndex(sessions) if sessions is not None else pd.DatetimeIndex([]),
        policy=policy or FreshnessPolicy(
            max_data_age_sessions=0 if sessions is None else
            FreshnessPolicy().max_data_age_sessions,
        ),
    )
    if findings and not force:
        raise PromotionError(f"candidate is stale.\n{describe(findings)}")

    archived = _archive(deployed, Path(archive_dir))
    deployed.parent.mkdir(parents=True, exist_ok=True)

    live_meta = deployed.with_suffix(".meta.json")
    releases_dir = (Path(releases_dir) if releases_dir is not None
                    else deployed.parent / RELEASES_DIRNAME)
    pointer = deployed.parent / POINTER_NAME

    try:
        # An existing install has real files at the deployed paths. Seed a
        # release from what is already live and point at that first, so the
        # conversion to symlinks never changes what either path resolves to --
        # both sides are byte-identical to the bundle already deployed.
        if deployed.exists() and not deployed.is_symlink():
            _, live_meta_obj = load_model(deployed)
            legacy = _stage_release(
                releases_dir, deployed,
                live_meta if live_meta.exists() else cand_meta,
                live_meta_obj,
            )
            _swap_symlink(pointer, os.path.relpath(legacy, deployed.parent))
            _swap_symlink(deployed, f"{POINTER_NAME}/model.pkl")
            _swap_symlink(live_meta, f"{POINTER_NAME}/model.meta.json")

        release = _stage_release(releases_dir, candidate, cand_meta, meta)

        if not pointer.is_symlink():
            # Nothing was deployed. Publish the release, then link to it;
            # there is no previous version, so no mismatch is possible.
            _swap_symlink(pointer, os.path.relpath(release, deployed.parent))
            _swap_symlink(deployed, f"{POINTER_NAME}/model.pkl")
            _swap_symlink(live_meta, f"{POINTER_NAME}/model.meta.json")
        else:
            # The single visible switch. Both deployed paths resolve through
            # the pointer, so this one rename moves the model and its metadata
            # together -- there is no interval in which they disagree.
            _swap_symlink(pointer, os.path.relpath(release, deployed.parent))
    except Exception as exc:  # noqa: BLE001 - report rather than leave debris
        raise PromotionError(f"promotion failed while installing: {exc}") from exc

    return PromotionResult(deployed=deployed, archived=archived,
                           findings=findings, release=release)


def list_releases(deployed: Path | str,
                  releases_dir: Path | str | None = None) -> list[Path]:
    """Every release on disk, newest first. Release ids sort chronologically."""
    deployed = Path(deployed)
    root = (Path(releases_dir) if releases_dir is not None
            else deployed.parent / RELEASES_DIRNAME)
    if not root.is_dir():
        return []
    return sorted(
        (d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")),
        key=lambda d: d.name, reverse=True,
    )


def current_release(deployed: Path | str) -> Path | None:
    """The release the deployed paths currently resolve to, if any."""
    pointer = Path(deployed).parent / POINTER_NAME
    if not pointer.is_symlink():
        return None
    return pointer.resolve()


def rollback_release(
    deployed: Path | str,
    release: Path | str,
    *,
    expected_horizon: int | None = None,
) -> PromotionResult:
    """Point the deployed paths back at an earlier *release*, atomically.

    Validated the same way a promotion is: an unloadable or unscoreable
    release is refused rather than deployed, because the reason to roll back
    is usually that something is already wrong and a second broken artifact
    does not help.
    """
    deployed, release = Path(deployed), Path(release)
    if not release.is_dir():
        raise PromotionError(f"no such release: {release}")
    model_path = release / "model.pkl"
    meta_path = release / "model.meta.json"
    for required in (model_path, meta_path):
        if not required.exists():
            raise PromotionError(f"release is incomplete, missing {required.name}")

    try:
        model, meta = load_model(model_path)
    except Exception as exc:  # noqa: BLE001 - any failure to load disqualifies it
        raise PromotionError(f"release is not loadable: {exc}") from exc
    if not any(callable(getattr(model, m, None)) for m in ("predict_proba", "predict")):
        raise PromotionError(
            f"release holds a {type(model).__name__}, which cannot score"
        )
    if expected_horizon is not None and int(meta.get("horizon", -1)) != expected_horizon:
        raise PromotionError(
            f"release horizon {meta.get('horizon')} does not match the holding "
            f"rule ({expected_horizon})"
        )

    pointer = deployed.parent / POINTER_NAME
    _swap_symlink(pointer, os.path.relpath(release, deployed.parent))
    return PromotionResult(deployed=deployed, archived=None, findings=[],
                           release=release)
