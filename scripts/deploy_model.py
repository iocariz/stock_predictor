"""Promote a trained candidate model to the deployed path.

Training writes a candidate; this puts it in place, after checking that it
loads, carries the metadata the live path needs, is not stale, and matches the
holding rule it will be traded on. The outgoing model is archived so a bad
promotion is reversible, and a refused promotion changes nothing.

    uv run python scripts/deploy_model.py \\
        artifacts/model_candidate.pkl artifacts/model.pkl --expected-horizon 63
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stock_predictor.deploy import PromotionError, promote_model
from stock_predictor.freshness import describe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("candidate", type=Path)
    p.add_argument("deployed", type=Path)
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
