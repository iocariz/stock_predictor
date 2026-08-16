#!/usr/bin/env python3
"""Grid-search backtest configuration over one walk-forward scored panel.

Ranks by Sharpe but always reports CAPM alpha and its t-statistic against the
benchmark, because a grid's best Sharpe cell is a hypothesis, not a result:
with ~100 combinations, the top cell is expected to look good by chance. Check
the alpha t-stat and re-run the winner on a window it has never seen
(`--from-date` / `--until-date`).

Example:
  uv run python scripts/grid_search_sharpe.py artifacts/wf_scored.parquet
  uv run python scripts/grid_search_sharpe.py artifacts/wf_scored.parquet \
      --until-date 2024-12-31 --benchmark-ticker RSP
"""

from __future__ import annotations

import argparse
import itertools
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


def _parse_csv_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_optional_floats(raw: str) -> list[float | None]:
    return [
        None if x.strip().lower() == "none" else float(x)
        for x in raw.split(",")
        if x.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grid-search backtest configs; rank by Sharpe, report alpha t-stat.",
    )
    p.add_argument(
        "scored_path", type=Path,
        help="Scored panel parquet/csv with date,ticker,adj_close,prob",
    )
    p.add_argument(
        "--output-csv", type=Path,
        default=Path("artifacts/reports/grid_search_sharpe.csv"),
    )
    p.add_argument("--top-n", default="5,10,15", help="Comma-separated values")
    p.add_argument(
        "--holding-days", default="5,10", dest="holding_days",
        help="Comma-separated values",
    )
    p.add_argument(
        "--rebalance-day", default="Friday,last", dest="rebalance_day",
        help="Comma-separated values from Monday..Friday,last",
    )
    p.add_argument(
        "--min-prob", default="none", dest="min_prob",
        help="Comma-separated score floors; 'none' disables the floor. "
             "Only meaningful within one model family — classifier scores are "
             "uncalibrated and lambdarank scores are unbounded",
    )
    p.add_argument(
        "--vix-filter", default="none", dest="vix_filter",
        help="Comma-separated VIX percentile thresholds; 'none' disables. "
             "Requires a vix_percentile column in the panel",
    )
    p.add_argument("--weighting", choices=["equal", "probability"], default="equal")
    p.add_argument("--max-cohorts", type=int, default=2)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument(
        "--commission-per-share", type=float, default=0.0, dest="commission_per_share",
    )
    p.add_argument(
        "--commission-per-order", type=float, default=0.0, dest="commission_per_order",
    )
    p.add_argument("--benchmark-ticker", default="SPY", dest="benchmark_ticker")
    p.add_argument(
        "--no-benchmark", action="store_true",
        help="Skip the benchmark download (alpha/IR columns show N/A)",
    )
    p.add_argument("--from-date", default=None, dest="from_date", help="YYYY-MM-DD")
    p.add_argument("--until-date", default=None, dest="until_date", help="YYYY-MM-DD")
    p.add_argument(
        "--min-cohorts", type=int, default=10,
        help="Drop combos with fewer cohorts than this",
    )
    return p


def _slice_panel(scored: pd.DataFrame, from_date, until_date) -> pd.DataFrame:
    out = scored.copy()
    out["date"] = pd.to_datetime(out["date"])
    if from_date:
        out = out[out["date"] >= pd.Timestamp(from_date)]
    if until_date:
        out = out[out["date"] <= pd.Timestamp(until_date)]
    if out.empty:
        raise SystemExit("No rows left after --from-date / --until-date filtering.")
    return out


def main() -> None:
    args = build_parser().parse_args()
    scored = _slice_panel(_load_scored(args.scored_path), args.from_date, args.until_date)
    print(
        f"Loaded {len(scored)} rows, {scored['ticker'].nunique()} tickers, "
        f"{scored['date'].min().date()} → {scored['date'].max().date()}"
    )

    top_n_vals = _parse_csv_list(args.top_n, int)
    hold_vals = _parse_csv_list(args.holding_days, int)
    rebalance_vals = _parse_csv_list(args.rebalance_day, str)
    min_prob_vals = _parse_optional_floats(args.min_prob)
    vix_vals = _parse_optional_floats(args.vix_filter)

    # Download the benchmark once and reuse it for every variant.
    bench_raw = pd.Series(dtype=float)
    if not args.no_benchmark:
        bench_raw, _ = _download_benchmark(
            scored["date"].min(), scored["date"].max(),
            args.benchmark_ticker, args.capital,
        )
        if bench_raw.empty:
            print(f"Benchmark {args.benchmark_ticker} unavailable; alpha columns will be N/A.")

    combos = list(itertools.product(
        top_n_vals, hold_vals, rebalance_vals, min_prob_vals, vix_vals,
    ))
    total = len(combos)
    print(f"Running {total} combinations…\n")

    rows: list[dict] = []
    partial_csv = args.output_csv.with_suffix(args.output_csv.suffix + ".partial")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    for i, (top_n, holding_days, rebalance_day, min_prob, vix_filter) in enumerate(combos, 1):
        label = (
            f"top{top_n}/hold{holding_days}/{rebalance_day}"
            f"/min_prob={min_prob}/vix={vix_filter}"
        )
        try:
            cfg = BacktestConfig(
                top_n=top_n,
                holding_days=holding_days,
                rebalance_day=rebalance_day,
                weighting=args.weighting,
                slippage_bps=args.slippage_bps,
                initial_capital=args.capital,
                max_overlapping_cohorts=args.max_cohorts,
                min_prob=min_prob,
                vix_filter_percentile=vix_filter,
                benchmark_ticker=None,  # benchmark handled once, above
                commission_per_share=args.commission_per_share,
                commission_per_order=args.commission_per_order,
            )
            result = run_backtest(scored, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the grid
            # Printed, never swallowed: a skipped cell is absent from the
            # ranked CSV, so a silently-dropped combo cannot look like a
            # result. Config-validation errors (e.g. a VIX threshold on a
            # panel with no vix_percentile column) surface here.
            print(f"[{i}/{total}] SKIP {label}: {type(exc).__name__}: {exc}")
            continue

        m = result.metrics
        rel: dict[str, float] = {}
        if not bench_raw.empty:
            bench_nav = _align_benchmark_to_nav(
                bench_raw, result.daily_nav.index, args.capital,
            )
            rel = relative_metrics(result.daily_nav, bench_nav)

        n_cohorts = int(m.get("n_cohorts", 0))
        alpha_t = rel.get("alpha_t", float("nan"))
        print(
            f"[{i}/{total}] sharpe={m.get('sharpe', float('nan')):6.3f} "
            f"alpha_t={alpha_t:6.2f} cohorts={n_cohorts:4d}  {label}"
        )
        rows.append({
            "top_n": top_n,
            "holding_days": holding_days,
            "rebalance_day": rebalance_day,
            "min_prob": min_prob,
            "vix_filter": vix_filter,
            "weighting": args.weighting,
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "cagr": m.get("cagr"),
            "total_return": m.get("total_return"),
            "max_drawdown": m.get("max_drawdown"),
            "calmar": m.get("calmar"),
            "total_costs": m.get("total_costs"),
            "win_rate": m.get("win_rate"),
            "n_cohorts": n_cohorts,
            "active_return_ann": rel.get("active_return_ann"),
            "information_ratio": rel.get("information_ratio"),
            "beta": rel.get("beta"),
            "alpha_ann": rel.get("alpha_ann"),
            "alpha_t": alpha_t,
        })
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(partial_csv, index=False)

    if not rows:
        raise SystemExit("Every combination failed; nothing to rank.")

    out = pd.DataFrame(rows)
    dropped = int((out["n_cohorts"] < args.min_cohorts).sum())
    filtered = out[out["n_cohorts"] >= args.min_cohorts].copy()
    ranked = filtered.sort_values(["sharpe", "sortino"], ascending=False).reset_index(drop=True)
    ranked.to_csv(args.output_csv, index=False)
    partial_csv.unlink(missing_ok=True)

    print()
    if dropped:
        print(f"Dropped {dropped} combos with < {args.min_cohorts} cohorts.")
    print(f"Saved {len(ranked)} ranked rows to {args.output_csv}")
    if ranked.empty:
        print("No rows survived the min-cohorts filter.")
        return

    cols = [
        "sharpe", "sortino", "max_drawdown", "cagr", "alpha_ann", "alpha_t",
        "n_cohorts", "top_n", "holding_days", "rebalance_day", "min_prob", "vix_filter",
    ]
    print("\nTop 10 by Sharpe:")
    print(ranked[cols].head(10).to_string(index=False))

    best = ranked.iloc[0]
    print(
        f"\nBest cell: Sharpe {best['sharpe']:.3f}, alpha t-stat "
        f"{best['alpha_t'] if best['alpha_t'] == best['alpha_t'] else float('nan'):.2f}."
    )
    print(
        f"This is {len(out)} configurations searched on one panel — the top "
        "Sharpe is selection-biased upward. Treat it as a hypothesis: re-run "
        "it with --from-date/--until-date on a window it has never seen, and "
        "believe it only if |alpha t-stat| clears ~2 there too."
    )


if __name__ == "__main__":
    main()
