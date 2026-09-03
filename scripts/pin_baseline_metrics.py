"""Pin a baseline's headline figures so drift cannot pass silently.

Every gate in ``verify_baseline.py`` checks the baseline against *itself*: that
it reconciles, that it fills honestly, that its bytes match what was recorded.
None checked it against what had been **published** about it, and that is how
this happened:

At ``c656df9`` the baseline artifact was rebuilt with reused ticker symbols
removed. New run id, new commit, every snapshot hash different, 37,156 fewer
labelled rows. The document's provenance table and survivorship section were
updated. Its two results tables were not -- they went on describing the
artifact that had just been replaced, including a headline CAGR spread measured
on the *contaminated* panel. All gates passed, because none of them was looking
at the published numbers.

Pinning closes it. The figures a reader quotes are recorded next to the
artifacts they came from, and the verifier recomputes and compares them. Change
an engine, swap an artifact, or regenerate the panel, and verification fails
until someone consciously re-pins -- which is the point. Re-pinning is a
deliberate act with a commit attached, not a silent drift.

    uv run python scripts/pin_baseline_metrics.py artifacts/baseline
    uv run python scripts/pin_baseline_metrics.py artifacts/baseline --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_baseline import (  # noqa: E402
    PINNED_METRICS,
    RISK_FREE,
    measure_engines,
    read_manifest_key,
)

from stock_predictor import repro  # noqa: E402
from stock_predictor.replay import SnapshotIncomplete, SnapshotProvider  # noqa: E402

ENGINE_ORDER = ("long-short", "cohort", "rank-hold")


def _table(engines: dict[str, dict]) -> str:
    """The results table, generated rather than typed.

    BASELINE.md's numbers drifted because they were maintained by hand next to
    artifacts that were replaced. Emitting them from the measurement removes
    the step where that goes wrong.
    """
    rows = ["| engine | CAGR | Sharpe | max drawdown | beta | alpha/yr | HAC t |",
            "|---|---|---|---|---|---|---|"]
    for label in ENGINE_ORDER:
        m = engines.get(label)
        if not m:
            continue

        def f(key: str, pct: bool = False, sign: bool = False) -> str:
            v = m.get(key)
            if v is None or not pd.notna(v):
                return "—"
            if pct:
                return f"{v:+.2%}" if sign else f"{v:.2%}"
            return f"{v:+.2f}" if sign else f"{v:.2f}"

        rows.append(
            f"| {label} | {f('cagr', pct=True)} | {f('sharpe')} | "
            f"{f('max_drawdown', pct=True)} | {f('beta', sign=True)} | "
            f"{f('alpha_ann', pct=True, sign=True)} | {f('alpha_t', sign=True)} |"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--max-cohorts", type=int, default=2)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--rebalance-day", default="Friday")
    ap.add_argument("--exit-rank", type=int, default=40)
    ap.add_argument("--markdown", action="store_true",
                    help="Print the results table for BASELINE.md and exit "
                         "without writing.")
    args = ap.parse_args()

    d = args.baseline_dir
    scored = pd.read_parquet(d / "wf_scored.parquet")
    scored["date"] = pd.to_datetime(scored["date"])
    execution = pd.read_parquet(d / "execution_prices.parquet")

    # The benchmark comes from the snapshot, never the network: pinning against
    # a series that moves is pinning against nothing.
    try:
        provider = SnapshotProvider(d)
        provider.download_benchmark("SPY", None, None)
    except (SnapshotIncomplete, FileNotFoundError) as exc:
        sys.exit(f"{exc}\n\nRecord it first: "
                 f"uv run python scripts/record_baseline_benchmark.py {d}")

    engines = measure_engines(
        scored, execution, provider=provider, top_n=args.top_n,
        horizon=args.horizon, max_cohorts=args.max_cohorts,
        slippage_bps=args.slippage_bps, rebalance_day=args.rebalance_day,
        exit_rank=args.exit_rank,
    )
    metrics = {k: v["metrics"] for k, v in engines.items()}

    if args.markdown:
        print(_table(metrics))
        return

    pin = {
        "run_id": read_manifest_key(d, "run_id"),
        "baseline_commit": read_manifest_key(d, "git_commit"),
        "pinned_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned_at_commit": repro.git_revision().get("commit"),
        "provenance": "measured-from-these-artifacts",
        "risk_free_rate": RISK_FREE,
        "config": {
            "top_n": args.top_n, "horizon": args.horizon,
            "max_cohorts": args.max_cohorts, "slippage_bps": args.slippage_bps,
            "rebalance_day": args.rebalance_day, "exit_rank": args.exit_rank,
            "benchmark": "SPY (from the snapshot)",
        },
        "engines": {
            label: {k: (float(m[k]) if k in m and pd.notna(m[k]) else None)
                    for k in PINNED_METRICS}
            for label, m in metrics.items()
        },
    }
    out = d / "expected_metrics.json"
    out.write_text(json.dumps(pin, indent=2) + "\n")

    print(_table(metrics))
    print(f"\nPinned -> {out}")
    print("verify_baseline.py now fails if any of these move.")


if __name__ == "__main__":
    main()
