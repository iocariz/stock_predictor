"""Promote a trained candidate model to the deployed path.

Training writes a candidate; this puts it in place, after checking that it
loads, carries the metadata the live path needs, is not stale, and matches the
holding rule it will be traded on. The outgoing model is archived so a bad
promotion is reversible, and a refused promotion changes nothing.

    uv run python scripts/deploy_model.py \\
        artifacts/model_candidate.pkl artifacts/model.pkl --expected-horizon 63

Each promotion writes an immutable release directory and swaps one symlink, so
the model and its metadata always move together. Superseded releases stay on
disk:

    uv run python scripts/deploy_model.py --list artifacts/model.pkl
    uv run python scripts/deploy_model.py --rollback-to <release> artifacts/model.pkl

Roll back with ``--rollback-to``, never with ``mv``: a symlink pointing at a
directory swallows the replacement into the directory instead of replacing the
link, which silently leaves the old version live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stock_predictor.deploy import (
    PromotionError,
    current_release,
    list_releases,
    promote_model,
    rollback_release,
)
from stock_predictor.freshness import describe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("candidate", type=Path, nargs="?", default=None)
    p.add_argument("deployed", type=Path)
    p.add_argument("--list", action="store_true", dest="list_releases",
                   help="Show releases on disk, newest first, and the live one")
    p.add_argument("--rollback-to", type=Path, default=None, dest="rollback_to",
                   help="Point the deployed paths at an earlier release")
    p.add_argument("--archive-dir", type=Path, default=Path("artifacts/archive"),
                   dest="archive_dir")
    p.add_argument("--expected-horizon", type=int, default=None,
                   dest="expected_horizon",
                   help="Refuse a model whose horizon differs from the holding rule")
    p.add_argument("--panel", type=Path, default=None,
                   help="Scored panel or price parquet, for the data-age check")
    p.add_argument("--force", action="store_true",
                   help="Promote despite findings; they are still reported")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.list_releases:
        live = current_release(args.deployed)
        releases = list_releases(args.deployed)
        if not releases:
            print("No releases on disk.")
            return
        for r in releases:
            marker = " <- live" if live and r.resolve() == live else ""
            print(f"{r}{marker}")
        return

    if args.rollback_to is not None:
        try:
            res = rollback_release(args.deployed, args.rollback_to,
                                   expected_horizon=args.expected_horizon)
        except PromotionError as exc:
            sys.exit(f"Refusing to roll back: {exc}")
        print(f"Rolled back {res.deployed} -> {res.release}")
        return

    if args.candidate is None:
        sys.exit("A candidate path is required unless --list or --rollback-to.")

    sessions = None
    if args.panel is not None and args.panel.exists():
        df = pd.read_parquet(args.panel)
        sessions = pd.DatetimeIndex(
            df["date"].unique() if "date" in df.columns else df.index
        )
    try:
        res = promote_model(
            args.candidate, args.deployed, archive_dir=args.archive_dir,
            expected_horizon=args.expected_horizon, sessions=sessions,
            force=args.force,
        )
    except PromotionError as exc:
        sys.exit(f"Refusing to promote: {exc}")
    if res.findings:
        print(describe(res.findings), file=sys.stderr)
        print("  --force: promoted anyway.", file=sys.stderr)
    print(f"Deployed {args.candidate} -> {res.deployed}")
    if res.archived:
        print(f"Previous model archived at {res.archived}")


if __name__ == "__main__":
    main()
