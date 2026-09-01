"""Choose the configuration on one window; test it once on another.

The long-short book measures alpha of +8.90%/yr at HAC t = +2.76, every run
above |t| = 2. The objection to reading that as an edge is multiplicity: the
configuration it was measured at -- decile 0.1, 1.0x gross, 63-day rebalance --
was picked from a search over exactly those knobs, on the same data. The best of
dozens of correlated variants is not a pre-registered test, and that is true no
matter how carefully each individual variant was measured.

So: split the out-of-sample period. Search the grid on the **selection** window
only, commit to the single winner, and evaluate it **once** on the holdout. The
holdout is never consulted while choosing.

Two numbers come out, and the second matters as much as the first:

* how the committed configuration does on the holdout;
* how the *whole grid* does on the holdout, which says whether the winner was
  special or whether everything worked. If a randomly chosen configuration does
  about as well, the result is not evidence about this configuration -- it is
  evidence about the book, which is a different and weaker claim.

**What this cannot do.** The holdout is carved out of a period this project has
already looked at many times. It is not virgin data; nothing here is. What it
tests is whether the *selection procedure* generalises out of sample, which is
strictly weaker than a true pre-registration and strictly stronger than quoting
the maximum of a search.

    uv run python scripts/locked_holdout.py artifacts/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.backtest_reporting import relative_metrics
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

RISK_FREE = 0.045

GRID_DECILE = (0.05, 0.10, 0.20)
GRID_GROSS = (0.5, 1.0)
GRID_REBALANCE = (21, 63, 126)
"""The knobs the historical search ranged over, which is what makes the
multiplicity objection bite. Searching a *different* grid would not answer it."""

SELECTION_CRITERION = "sharpe"
"""Chosen on risk-adjusted return, deliberately *not* on the alpha t-statistic
that gets reported. Selecting on the statistic you then quote invites the same
objection one level down."""


class _Bench:
    """Serves one downloaded benchmark series to every run."""

    def __init__(self, nav: pd.Series):
        self._nav = nav

    def download_benchmark(self, ticker, start, end):
        s = self._nav
        return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def _slice(scored: pd.DataFrame, lo: str | None, hi: str | None) -> pd.DataFrame:
    d = pd.to_datetime(scored["date"])
    m = pd.Series(True, index=scored.index)
    if lo:
        m &= d >= pd.Timestamp(lo)
    if hi:
        m &= d <= pd.Timestamp(hi)
    return scored[m]


def _evaluate(scored, execution, bench, decile, gross, rebalance) -> dict:
    cfg = LongShortConfig(
        decile=decile, long_weight=gross / 2, short_weight=gross / 2,
        rebalance_every=rebalance, slippage_bps=5.0,
        risk_free_rate=RISK_FREE, benchmark_ticker="SPY",
    )
    res = run_long_short_backtest(scored, cfg, provider=bench,
                                  execution_prices=execution)
    m = dict(res.metrics)
    if len(res.bench_daily_nav) > 1:
        m.update(relative_metrics(res.daily_nav, res.bench_daily_nav,
                                  overlap_days=rebalance,
                                  risk_free_rate=RISK_FREE))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--split", default="2023-01-01",
                    help="First session of the holdout")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    scored = pd.read_parquet(args.baseline_dir / "wf_scored.parquet")
    scored["date"] = pd.to_datetime(scored["date"])
    execution = pd.read_parquet(args.baseline_dir / "execution_prices.parquet")

    sel = _slice(scored, None, args.split)
    hold = _slice(scored, args.split, None)
    print(f"selection {sel['date'].min().date()}..{sel['date'].max().date()} "
          f"({sel['date'].nunique()} sessions)")
    print(f"holdout   {hold['date'].min().date()}..{hold['date'].max().date()} "
          f"({hold['date'].nunique()} sessions)")

    import yfinance as yf
    raw = yf.download("SPY", start=str(scored["date"].min().date()),
                      end=str((scored["date"].max() + pd.Timedelta(days=1)).date()),
                      progress=False, auto_adjust=True)["Close"]
    bench = _Bench(raw.squeeze().dropna())

    combos = [(d, g, r) for d in GRID_DECILE for g in GRID_GROSS
              for r in GRID_REBALANCE]
    print(f"\nSearching {len(combos)} configurations on the selection window "
          f"by {SELECTION_CRITERION}…")

    sel_rows = []
    for d, g, r in combos:
        m = _evaluate(sel, execution, bench, d, g, r)
        sel_rows.append({"decile": d, "gross": g, "rebalance": r,
                         "sharpe": m.get("sharpe", float("nan")),
                         "cagr": m.get("cagr", float("nan")),
                         "alpha_t": m.get("alpha_t", float("nan"))})
    sel_df = pd.DataFrame(sel_rows).sort_values(SELECTION_CRITERION, ascending=False)
    print(sel_df.head(5).to_string(index=False))

    win = sel_df.iloc[0]
    print(f"\nCOMMITTED: decile={win.decile:g}, gross={win.gross:g}x, "
          f"rebalance={int(win.rebalance)}d "
          f"(selection Sharpe {win.sharpe:.2f}, t {win.alpha_t:+.2f})")

    print("\nEvaluating the committed configuration on the holdout, once.")
    held = _evaluate(hold, execution, bench, win.decile, win.gross,
                     int(win.rebalance))
    print(f"  CAGR    {held.get('cagr', float('nan')):+.2%}")
    print(f"  Sharpe  {held.get('sharpe', float('nan')):.2f}")
    print(f"  max DD  {held.get('max_drawdown', float('nan')):+.2%}")
    print(f"  beta    {held.get('beta', float('nan')):+.3f}")
    print(f"  alpha   {held.get('alpha_ann', float('nan')):+.2%} "
          f"(HAC t {held.get('alpha_t', float('nan')):+.2f})")

    # The control: what the rest of the grid did on the holdout. If everything
    # works, the winner tells you nothing about the winner.
    hold_rows = []
    for d, g, r in combos:
        m = _evaluate(hold, execution, bench, d, g, r)
        hold_rows.append({"decile": d, "gross": g, "rebalance": r,
                          "alpha_t": m.get("alpha_t", float("nan")),
                          "alpha_ann": m.get("alpha_ann", float("nan")),
                          "sharpe": m.get("sharpe", float("nan"))})
    hold_df = pd.DataFrame(hold_rows)
    t = hold_df["alpha_t"].dropna()
    print(f"\nWhole grid on the holdout: alpha t median {t.median():+.2f}, "
          f"range {t.min():+.2f}..{t.max():+.2f}; "
          f"{int((t > 2).sum())}/{len(t)} above +2")
    rank = int((hold_df["alpha_t"] > held.get("alpha_t", float("nan"))).sum()) + 1
    print(f"The committed configuration ranks {rank}/{len(hold_df)} on the "
          f"holdout by alpha t.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "split": args.split,
            "criterion": SELECTION_CRITERION,
            "committed": {k: float(win[k]) for k in ("decile", "gross", "rebalance")},
            "holdout": {k: (None if not np.isfinite(v) else float(v))
                        for k, v in held.items() if isinstance(v, (int, float))},
            "grid_holdout": hold_df.to_dict("records"),
        }, indent=2))
        print(f"\nReport -> {args.report}")


if __name__ == "__main__":
    main()
