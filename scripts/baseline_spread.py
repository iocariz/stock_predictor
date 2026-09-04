"""How much of a baseline's headline number is the draw rather than the strategy.

One commit, one pinned window, one seed, four fresh downloads. The panels agree
to vendor float noise -- 2e-6 relative, adjustment arithmetic rather than
revised data -- and LightGBM splits flip on near-ties, so a different fifteen
names get held and the curve lands somewhere else.

That makes a single artifact's CAGR a draw from a distribution, not a
measurement of the strategy, and the width of that distribution is the number
that decides whether any comparison between two runs means anything. A fix
worth two points cannot be told from noise worth four.

This reads the pinned metrics from several verified baselines and reports, per
engine and per metric, the mean, the spread, and the range. It reads only
``expected_metrics.json``, so it describes exactly what each baseline published
about itself rather than re-deriving anything.

    uv run python scripts/baseline_spread.py artifacts/baseline_v2 artifacts/baseline_v3 …
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PCT = {"cagr", "max_drawdown", "alpha_ann"}
ORDER = ("long-short", "cohort", "rank-hold")
METRICS = ("cagr", "sharpe", "max_drawdown", "beta", "alpha_ann", "alpha_t")


def _load(d: Path) -> dict:
    path = d / "expected_metrics.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; pin it with scripts/pin_baseline_metrics.py")
    return json.loads(path.read_text())


def _fmt(key: str, v: float) -> str:
    if v != v:
        return "—"
    return f"{v:+.2%}" if key in PCT else f"{v:+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baselines", nargs="+", type=Path)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    pins = [(d, _load(d)) for d in args.baselines]
    print(f"{len(pins)} draws, one commit, one pinned window, one seed:\n")
    for d, p in pins:
        print(f"  {d.name:22s} run {p.get('run_id')}  commit "
              f"{str(p.get('baseline_commit'))[:12]}")

    commits = {str(p.get("baseline_commit")) for _, p in pins}
    if len(commits) > 1:
        print("\nWARNING: these draws are not from one commit, so the spread "
              "below mixes code changes with vendor noise:")
        for c in sorted(commits):
            print(f"  {c}")

    rows = []
    for engine in ORDER:
        for key in METRICS:
            vals = pd.Series(
                [p["engines"].get(engine, {}).get(key) for _, p in pins],
                dtype="float64",
            ).dropna()
            if vals.empty:
                continue
            rows.append({
                "engine": engine, "metric": key, "n": int(len(vals)),
                "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "min": float(vals.min()), "max": float(vals.max()),
            })
    df = pd.DataFrame(rows)

    print(f"\n{'engine':11s} {'metric':13s} {'mean':>10s} {'sd':>10s} "
          f"{'min':>10s} {'max':>10s} {'range':>10s}")
    for r in df.itertuples():
        rng = r.max - r.min
        print(f"{r.engine:11s} {r.metric:13s} {_fmt(r.metric, r.mean):>10s} "
              f"{_fmt(r.metric, r.sd):>10s} {_fmt(r.metric, r.min):>10s} "
              f"{_fmt(r.metric, r.max):>10s} {_fmt(r.metric, rng):>10s}")

    if len(pins) < 2:
        print("\nOne draw is not a spread. Build more before quoting a range.")
    else:
        print("\nWhat a difference has to clear to mean anything "
              "(2 sd, per engine):")
        for engine in ORDER:
            sub = df[(df["engine"] == engine) & (df["metric"] == "cagr")]
            if sub.empty:
                continue
            print(f"  {engine:11s} CAGR  ±{2 * float(sub['sd'].iloc[0]):.2%}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "draws": [{"dir": str(d), "run_id": p.get("run_id"),
                       "commit": p.get("baseline_commit")} for d, p in pins],
            "spread": df.to_dict("records"),
        }, indent=2))
        print(f"\nReport -> {args.report}")


if __name__ == "__main__":
    main()
