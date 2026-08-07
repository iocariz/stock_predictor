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

CONFIGS: list[tuple[str, dict]] = [
    ("baseline top10/hold10/eq", {}),
    ("prob-weighted", {"weighting": "probability"}),
    ("vix filter <=0.80", {"vix_filter_percentile": 0.80}),
    ("vix filter <=0.90", {"vix_filter_percentile": 0.90}),
    ("hold 5d", {"holding_days": 5}),
    ("top 5", {"top_n": 5}),
    ("top 15", {"top_n": 15}),
    ("top 20", {"top_n": 20}),
    ("prob + vix<=0.80", {"weighting": "probability", "vix_filter_percentile": 0.80}),
    ("hold5 + top20", {"holding_days": 5, "top_n": 20}),
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
    args = ap.parse_args()

    scored = _load_scored(args.scored_path)
    print(f"Loaded {len(scored)} scored rows from {args.scored_path}")

    rows: list[dict[str, object]] = []
    baseline_nav: pd.Series | None = None
    for label, overrides in CONFIGS:
        cfg = BacktestConfig(
            benchmark_ticker=None, initial_capital=args.capital, **overrides,
        )
        res = run_backtest(scored, cfg)
        rows.append(_row(label, res.metrics))
        if baseline_nav is None:
            baseline_nav = res.daily_nav
        print(f"  ran: {label} ({res.metrics.get('n_cohorts', 0):.0f} cohorts)")

    if args.benchmark_ticker and baseline_nav is not None:
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


if __name__ == "__main__":
    main()
