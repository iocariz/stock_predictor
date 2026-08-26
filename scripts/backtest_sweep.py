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
    run_rank_hold_backtest,
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

# Low-beta / VIX-regime grid: hard-skip thresholds vs continuous de-risking
# (full size below the median VIX percentile, linearly to zero at the top).
VIX_CONFIGS: list[tuple[str, dict]] = [
    ("baseline top15 (no filter)", {}),
    ("vix skip >0.95", {"vix_filter_percentile": 0.95}),
    ("vix skip >0.90", {"vix_filter_percentile": 0.90}),
    ("vix skip >0.85", {"vix_filter_percentile": 0.85}),
    ("vix skip >0.80", {"vix_filter_percentile": 0.80}),
    ("vix skip >0.70", {"vix_filter_percentile": 0.70}),
    ("vix-scaled exposure", {"vix_scale_exposure": True}),
    ("scaled + skip >0.90", {"vix_scale_exposure": True, "vix_filter_percentile": 0.90}),
]

# Cohort (fixed 10d liquidation) vs rank-based holding (sell on rank decay).
# The special "mode" key selects the backtest engine; everything else is
# BacktestConfig overrides.
HOLD_CONFIGS: list[tuple[str, dict]] = [
    ("cohort top15/hold10 (baseline)", {}),
    ("rank-hold exit>25", {"mode": "rank_hold", "exit_rank": 25}),
    ("rank-hold exit>40", {"mode": "rank_hold", "exit_rank": 40}),
    ("rank-hold exit>60", {"mode": "rank_hold", "exit_rank": 60}),
    ("rank-hold exit>100", {"mode": "rank_hold", "exit_rank": 100}),
    ("rank-hold exit>40 top20", {"mode": "rank_hold", "exit_rank": 40, "top_n": 20}),
    ("rank-hold exit>40 + vix-scaled", {"mode": "rank_hold", "exit_rank": 40, "vix_scale_exposure": True}),
]

GRIDS: dict[str, list[tuple[str, dict]]] = {
    "default": CONFIGS,
    "vix": VIX_CONFIGS,
    "hold": HOLD_CONFIGS,
}


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
    ap.add_argument("--execution-prices", type=Path, default=None,
                    dest="execution_prices",
                    help="Unfiltered price panel that prices the fills. Without "
                         "it, fills come from the point-in-time scored panel, "
                         "which on the control panel overstates CAGR by ~6.6pp")
    ap.add_argument("--benchmark-ticker", default="SPY")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--from-date", default=None, dest="from_date",
                    help="Restrict scored panel to dates >= this (YYYY-MM-DD)")
    ap.add_argument("--until-date", default=None, dest="until_date",
                    help="Restrict scored panel to dates <= this (YYYY-MM-DD)")
    ap.add_argument("--grid", default="default", choices=sorted(GRIDS),
                    help="Which variant grid to run (default: general strategy grid)")
    args = ap.parse_args()
    configs = GRIDS[args.grid]

    scored = _load_scored(args.scored_path)
    # An absent execution panel is not a smaller measurement, it is a
    # different one: fills then come from the point-in-time scored panel,
    # which on the control panel overstates CAGR by ~6.6pp and understates
    # drawdown by ~18pp. Say so rather than producing a quiet wrong number.
    exec_px = None
    if args.execution_prices is not None:
        exec_px = pd.read_parquet(args.execution_prices)
        print(f"Execution prices: {args.execution_prices} "
              f"({exec_px.shape[0]} dates x {exec_px.shape[1]} tickers)")
    else:
        print("WARNING: no --execution-prices; fills come from the "
              "point-in-time scored panel and results are optimistic.")
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
    for label, overrides in configs:
        ov = dict(overrides)
        engine = run_rank_hold_backtest if ov.pop("mode", "cohort") == "rank_hold" else run_backtest
        cfg = BacktestConfig(
            benchmark_ticker=None, initial_capital=args.capital, **ov,
        )
        res = engine(scored, cfg, execution_prices=exec_px)
        rows.append(_row(label, res.metrics))
        navs.append((label, res.daily_nav))
        print(f"  ran: {label} ({res.metrics.get('n_cohorts', 0):.0f} closed trades/cohorts)")

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
                "alpha_t": f"{rm['alpha_t']:.2f}",
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
