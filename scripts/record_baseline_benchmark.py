"""Record the benchmark series for a baseline built before it was captured.

Four of a run's five external inputs are snapshotted. The fifth -- the
benchmark -- was fetched at report time, so every published beta, alpha and HAC
t-statistic was measured against whatever the vendor served that day, and the
verifier could not check any of them against the artifacts. It compensated by
turning the benchmark off entirely (``benchmark_ticker=None``), which made the
long-short book's headline numbers the one part of the baseline nothing gated.

The pipeline now records it. This backfills a baseline built earlier.

Like sealing an output, this is weaker than the real thing and is marked as
such: the series is today's view of a historical index, not the one the run
actually used. For a total-return index over a closed historical window those
are the same series up to vendor revisions -- unlike prices, there is no
survivorship or membership question here -- but "should be identical" is not
"was recorded", and the manifest says which it is.

    uv run python scripts/record_baseline_benchmark.py artifacts/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_predictor import repro
from stock_predictor.cli import BENCHMARK_TICKER, _download_benchmark_series
from stock_predictor.data_provider import get_provider


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--provider", default="yfinance",
                    help="Benchmark source. yfinance needs no key and serves "
                         "SPY adjusted closes.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    d = args.baseline_dir
    man_path = d / "snapshot" / "manifest.json"
    if not man_path.exists():
        sys.exit(f"no manifest at {man_path}")
    man = json.loads(man_path.read_text())

    if "benchmark" in man.get("snapshots", {}) and not args.force:
        sys.exit("benchmark already recorded; pass --force to replace it")

    execution = pd.read_parquet(d / "execution_prices.parquet")
    sessions = pd.DatetimeIndex(pd.to_datetime(execution.index)).sort_values()
    print(f"window {sessions.min().date()} .. {sessions.max().date()} "
          f"({len(sessions):,} sessions)")

    bench = _download_benchmark_series(get_provider(args.provider), sessions)
    if bench is None or bench.empty:
        sys.exit("no benchmark bars returned; nothing recorded")

    meta = repro.snapshot_parquet(bench, d / "snapshot" / "benchmark.parquet")
    meta["provenance"] = "recorded-after-the-fact"
    meta["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["provider"] = args.provider
    repro.register_snapshot(man, "benchmark", meta)
    repro.write_manifest(man_path, man)

    print(f"Recorded {BENCHMARK_TICKER}: {len(bench):,} sessions, "
          f"sha256 {meta['sha256'][:16]}")
    print("Marked recorded-after-the-fact: today's view of the index, not the "
          "series the original run fetched.")


if __name__ == "__main__":
    main()
