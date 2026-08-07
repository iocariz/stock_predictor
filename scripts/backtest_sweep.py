"""Backtest a grid of strategy variants over one walk-forward scored panel.

Runs each BacktestConfig variant on the same scores, downloads the benchmark
once (instead of once per variant), and prints a comparison table.

Example:
  uv run python scripts/backtest_sweep.py artifacts/wf_scored.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_predictor.backtest import (
    BacktestConfig,
    _align_benchmark_to_nav,
    _compute_metrics,
    _download_benchmark,
    _load_scored,
    run_backtest,
)
from stock_predictor.backtest_reporting import relative_metrics

CONFIGS: list[tuple[str, dict]] = [
    ("baseline top15/hold10/eq", {}),  # top_n defaults to 15
    ("top 10 (old default)", {"top_n": 10}),
    ("top 20", {"top_n": 20}),
    ("top 25", {"top_n": 25}),
    ("prob-weighted", {"weighting": "probability"}),
    ("vix filter <=0.90", {"vix_filter_percentile": 0.90}),
    ("hold 15d", {"holding_days": 15}),
    ("top20 + hold15", {"top_n": 20, "holding_days": 15}),
]


def _row(label: str, m: dict[str, float]) -> dict[str, object]:
    def pct(k: str) -> str:
        v = m.get(k, float("nan"))
        return f"{v:+.1%}" if v == v else "N/A"

    def num(k: str) -> str:
        v = m.get(k, float("nan"))
        return f"{v:.2f}" if v == v else "N/A"

    return {
        "variant": label,
        "total_ret": pct("total_return"),
        "cagr": pct("cagr"),
        "sharpe": num("sharpe"),
        "sortino": num("sortino"),
        "max_dd": pct("max_drawdown"),
        "calmar": num("calmar"),
        "win_rate": pct("win_rate"),
        "cohorts": int(m.get("n_cohorts", 0)),
        "costs": f"${m.get('total_costs', 0):,.0f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest a grid of config variants.")
    ap.add_argument("scored_path", type=Path)
    ap.add_argument("--benchmark-ticker", default="SPY")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--from-date", default=None, dest="from_date",
                    help="Restrict scored panel to dates >= this (YYYY-MM-DD)")
    ap.add_argument("--until-date", default=None, dest="until_date",
                    help="Restrict scored panel to dates <= this (YYYY-MM-DD)")
    args = ap.parse_args()

    scored = _load_scored(args.scored_path)
    print(f"Loaded {len(scored)} scored rows from {args.scored_path}")
    if args.from_date or args.until_date:
        d = pd.to_datetime(scored["date"])
        if args.from_date:
            scored = scored[d >= pd.Timestamp(args.from_date)]
            d = pd.to_datetime(scored["date"])
        if args.until_date:
            scored = scored[d <= pd.Timestamp(args.until_date)]
        print(f"  Date filter: {len(scored)} rows "
              f"({args.from_date or 'start'} -> {args.until_date or 'end'})")

    rows: list[dict[str, object]] = []
    navs: list[tuple[str, pd.Series]] = []
    for label, overrides in CONFIGS:
        cfg = BacktestConfig(
            benchmark_ticker=None, initial_capital=args.capital, **overrides,
        )
        res = run_backtest(scored, cfg)
        rows.append(_row(label, res.metrics))
        navs.append((label, res.daily_nav))
        print(f"  ran: {label} ({res.metrics.get('n_cohorts', 0):.0f} cohorts)")

    baseline_nav = navs[0][1]
    aligned = pd.Series(dtype=float)
    if args.benchmark_ticker:
        start = pd.Timestamp(baseline_nav.index[0])
        end = pd.Timestamp(baseline_nav.index[-1])
        raw_nav, _ = _download_benchmark(start, end, args.benchmark_ticker, args.capital)
        aligned = _align_benchmark_to_nav(raw_nav, baseline_nav.index, args.capital)
        if aligned.notna().sum() > 1:
            rows.append(_row(f"{args.benchmark_ticker} buy & hold", _compute_metrics(aligned, [])))
        else:
            print(f"Benchmark {args.benchmark_ticker} unavailable; table has no benchmark row.")

    table = pd.DataFrame(rows)
    print()
    print("=" * 110)
    print(f"STRATEGY SWEEP: {baseline_nav.index[0].date()} -> {baseline_nav.index[-1].date()}")
    print("=" * 110)
    print(table.to_string(index=False))

    # Relative-return framing: each variant as an overlay on the benchmark.
    if aligned.notna().sum() > 1:
        rel_rows = []
        for label, nav in navs:
            rm = relative_metrics(nav, aligned)
            if not rm:
                continue
            rel_rows.append({
                "variant": label,
                "active_ret_ann": f"{rm['active_return_ann']:+.1%}",
                "track_err": f"{rm['tracking_error']:.1%}",
                "info_ratio": f"{rm['information_ratio']:.2f}",
                "beta": f"{rm['beta']:.2f}",
                "alpha_ann": f"{rm['alpha_ann']:+.1%}",
                "up_capt": f"{rm['up_capture']:.2f}",
                "down_capt": f"{rm['down_capture']:.2f}",
                "overlay_ret": f"{rm['overlay_total_return']:+.1%}",
                "overlay_dd": f"{rm['overlay_max_drawdown']:+.1%}",
            })
        print()
        print("=" * 110)
        print(f"RELATIVE-RETURN FRAMING vs {args.benchmark_ticker} "
              "(active = long strategy / short benchmark, daily rebalance)")
        print("=" * 110)
        print(pd.DataFrame(rel_rows).to_string(index=False))


if __name__ == "__main__":
    main()
