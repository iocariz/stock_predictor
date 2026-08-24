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
* the model and its metadata are swapped in via ``os.replace``, so a crash
  mid-promotion cannot leave a live model paired with the previous run's
  metadata — a mismatch that reads as a working deployment

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


@dataclass(frozen=True)
class PromotionResult:
    deployed: Path
    archived: Path | None
    findings: list[Finding]


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

    # Stage both files beside their targets, then rename. os.replace is atomic
    # within a filesystem, so each target is either wholly old or wholly new --
    # never a half-copied pickle. The pair is swapped back to back, which is
    # as close to atomic as two paths get without a symlinked release dir.
    live_meta = deployed.with_suffix(".meta.json")
    tmp_model = deployed.with_name(deployed.name + ".promoting")
    tmp_meta = live_meta.with_name(live_meta.name + ".promoting")
    try:
        shutil.copy2(candidate, tmp_model)
        shutil.copy2(cand_meta, tmp_meta)
        os.replace(tmp_model, deployed)
        os.replace(tmp_meta, live_meta)
    except Exception as exc:  # noqa: BLE001 - clean up rather than leave debris
        for tmp in (tmp_model, tmp_meta):
            tmp.unlink(missing_ok=True)
        raise PromotionError(f"promotion failed while installing: {exc}") from exc

    return PromotionResult(deployed=deployed, archived=archived, findings=findings)
