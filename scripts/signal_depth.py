#!/usr/bin/env python3
"""Does the ranking work where the strategy actually trades?

Regenerates the selection-depth diagnostic behind the README's results
disclosure, so that claim can be re-derived from any scored panel rather
than trusted. Three views, none of which depend on portfolio construction:

  1. Mean forward return by selection depth, vs the universe average.
  2. Rank IC (per-date Spearman correlation of score to forward return).
  3. Optionally, CAPM alpha vs a benchmark across a --top-n ladder.

Read the shape, not the level. A ranker with usable skill puts its best
forward returns at the top of the list, and gains alpha as you concentrate.
If deeper buckets beat the tightest one, the edge is not where you trade.

Every t-statistic is HAC-corrected: forward returns overlap, so ordinary
standard errors overstate significance here.

Example:
  uv run python scripts/signal_depth.py artifacts/wf_scored.parquet
  uv run python scripts/signal_depth.py artifacts/wf_scored.parquet \
      --benchmark-ticker RSP --until-date 2025-06-30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_predictor.backtest import (
    BacktestConfig,
    _align_benchmark_to_nav,
    _download_benchmark,
    _load_scored,
    run_backtest,
)
from stock_predictor.backtest_reporting import relative_metrics
from stock_predictor.signal_depth import (
    DEFAULT_BUCKETS,
    depth_frame,
    depth_table,
    format_depth_table,
    is_signal_monotone,
    rank_ic,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Selection-depth and rank-IC diagnostics for a scored panel.",
    )
    p.add_argument("scored_path", type=Path, help="Scored parquet/CSV with date,ticker,prob,fwd_ret")
    p.add_argument(
        "--buckets",
        default=",".join(str(b) for b in DEFAULT_BUCKETS),
        help="Comma-separated selection depths",
    )
    p.add_argument(
        "--horizon", type=int, default=10,
        help="Label horizon in sessions; sets the HAC lag floor (default 10)",
    )
    p.add_argument("--from-date", default=None, dest="from_date", help="YYYY-MM-DD")
    p.add_argument("--until-date", default=None, dest="until_date", help="YYYY-MM-DD")
    p.add_argument(
        "--top-n-ladder", default="5,10,15,25,50", dest="top_n_ladder",
        help="Comma-separated --top-n values for the alpha ladder ('' to skip)",
    )
    p.add_argument("--benchmark-ticker", default="SPY", dest="benchmark_ticker")
    p.add_argument(
        "--no-benchmark", action="store_true",
        help="Skip the alpha ladder (no network)",
    )
    p.add_argument("--rf-rate", type=float, default=0.045, dest="rf_rate")
    p.add_argument("--output-csv", type=Path, default=None, help="Write the depth table here")
    return p


def _slice(panel: pd.DataFrame, from_date, until_date) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    if from_date:
        out = out[out["date"] >= pd.Timestamp(from_date)]
    if until_date:
        out = out[out["date"] <= pd.Timestamp(until_date)]
    if out.empty:
        raise SystemExit("No rows left after --from-date / --until-date filtering.")
    return out


def _alpha_ladder(panel: pd.DataFrame, ladder: list[int], ticker: str, rf: float) -> None:
    raw, _ = _download_benchmark(panel["date"].min(), panel["date"].max(), ticker, 100_000.0)
    if raw.empty:
        print(f"  Benchmark {ticker} unavailable; skipping the alpha ladder.")
        return
    head = f"{'top-n':>6s}  {'total ret':>10s}  {'beta':>6s}  {'alpha (ann)':>12s}  {'HAC t':>7s}"
    print(head)
    print("-" * len(head))
    alphas: list[tuple[int, float]] = []
    for n in ladder:
        cfg = BacktestConfig(
            benchmark_ticker=None, top_n=n, risk_free_rate=rf,
            exit_rank=max(40, n),
        )
        result = run_backtest(panel, cfg)
        bench = _align_benchmark_to_nav(raw, result.daily_nav.index, cfg.initial_capital)
        rel = relative_metrics(bench_nav=bench, strategy_nav=result.daily_nav,
                               overlap_days=cfg.holding_days)
        a = rel.get("alpha_ann", float("nan"))
        alphas.append((n, a))
        print(
            f"{n:>6d}  {result.metrics['total_return']:>+10.1%}  {rel.get('beta', float('nan')):>6.2f}"
            f"  {a:>+12.1%}  {rel.get('alpha_t', float('nan')):>+7.2f}"
        )
    if len(alphas) >= 2 and alphas[0][1] == alphas[0][1] and alphas[-1][1] == alphas[-1][1]:
        tight, wide = alphas[0], alphas[-1]
        if tight[1] < wide[1]:
            print(
                f"\n  Alpha *falls* as you concentrate ({wide[1]:+.1%} at top-{wide[0]} "
                f"-> {tight[1]:+.1%} at top-{tight[0]}). A ranker with skill moves the "
                "other way; this one dilutes toward the index to look better."
            )


def main() -> None:
    args = build_parser().parse_args()
    panel = _slice(_load_scored(args.scored_path), args.from_date, args.until_date)
    if "fwd_ret" not in panel.columns:
        raise SystemExit(
            "Panel has no 'fwd_ret' column — re-export it with --wf-scores-path "
            "from a training run (the walk-forward writer includes it)."
        )
    buckets = tuple(int(b) for b in args.buckets.split(",") if b.strip())

    print(
        f"\nPanel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}"
    )
    print("=" * 62)
    print("FORWARD RETURN BY SELECTION DEPTH (equal-weighted, before costs)")
    print("=" * 62)
    rows = depth_table(panel, buckets=buckets, horizon=args.horizon)
    print(format_depth_table(rows))

    if not is_signal_monotone(rows):
        tightest = min((r for r in rows if r.label.startswith("top ")), key=lambda r: r.n_names)
        best = max((r for r in rows if r.label.startswith("top ")), key=lambda r: r.mean_fwd_ret)
        print(
            f"\n  Shape check FAILED: '{best.label}' beats '{tightest.label}'. The edge is "
            "not at the top of the ranking, which is the only part a top-N strategy trades."
        )
    else:
        print("\n  Shape check passed: the tightest bucket has the best forward return.")

    ic = rank_ic(panel, horizon=args.horizon)
    print(
        f"\nRank IC: mean={ic['mean']:+.4f}  std={ic['std']:.4f}  "
        f"HAC t={ic['t']:+.2f}  ({ic['n_days']} days)"
    )
    if abs(ic["t"]) < 2:
        print("  |t| < 2: the cross-sectional signal is not distinguishable from noise.")

    ladder = [int(x) for x in args.top_n_ladder.split(",") if x.strip()]
    if ladder and not args.no_benchmark:
        print()
        print("=" * 62)
        print(f"CAPM ALPHA BY CONCENTRATION (vs {args.benchmark_ticker}, rf={args.rf_rate:.1%})")
        print("=" * 62)
        _alpha_ladder(panel, ladder, args.benchmark_ticker, args.rf_rate)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        depth_frame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved depth table to {args.output_csv}")


if __name__ == "__main__":
    main()
