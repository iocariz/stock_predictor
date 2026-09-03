"""Choose the configuration on one window; test it once on the next.

The long-short book measures alpha of +7.98%/yr at HAC t = +2.60 over the full
period. The objection to reading that as an edge is multiplicity: the
configuration it was measured at was picked from a search over exactly those
knobs, on the same data. The best of dozens of correlated variants is not a
pre-registered test, however carefully each variant was measured.

So: split the out-of-sample period. Search the grid on the **selection** window
only, commit to the single winner, and evaluate it **once** on the holdout. The
holdout is never consulted while choosing.

**The holdout must continue the strategy, not restart it.** The first version of
this script re-ran the engine over a truncated panel, which is a different
thing in three ways: the rebalance calendar is anchored to row zero of whatever
frame it is handed, so slicing moved every trade date; the book began flat, so
positions and short liabilities open at the split were discarded; and both
slices included the split session. What that measured was a fresh strategy
launched on the holdout's first session -- not the strategy under test, carried
forward. Here the engine runs once over the whole panel and the holdout window
is *measured out of the continuing NAV*, so the trades either side of the split
are the trades the strategy would actually have made.

Selection still runs on a truncated panel, which is correct: there the
truncation is the real beginning, not a boundary cut through a live book.

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.backtest_reporting import nav_metrics, relative_metrics
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest
from stock_predictor.replay import SnapshotIncomplete, SnapshotProvider

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


def _before(scored: pd.DataFrame, split: str) -> pd.DataFrame:
    """Sessions strictly before the split.

    Strict, so that no session appears in both windows. With the default splits
    this changes nothing -- 1 January is never a trading day -- but a split on a
    session the market was open would have put that day's signal on both sides.
    """
    return scored[pd.to_datetime(scored["date"]) < pd.Timestamp(split)]


def _run(scored, execution, provider, decile, gross, rebalance):
    cfg = LongShortConfig(
        decile=decile, long_weight=gross / 2, short_weight=gross / 2,
        rebalance_every=rebalance, slippage_bps=5.0,
        risk_free_rate=RISK_FREE, benchmark_ticker="SPY",
    )
    return run_long_short_backtest(scored, cfg, provider=provider,
                                   execution_prices=execution)


def _whole(res, rebalance: int) -> dict:
    m = dict(res.metrics)
    if len(res.bench_daily_nav) > 1:
        m.update(relative_metrics(res.daily_nav, res.bench_daily_nav,
                                  overlap_days=rebalance,
                                  risk_free_rate=RISK_FREE))
    return m


def _segment(res, split: str, rebalance: int) -> dict:
    """Measure the holdout window out of a run that spans the whole panel.

    The strategy is not restarted at the split: it arrives holding whatever it
    held, on the rebalance calendar it was already on, and this reads the
    performance of that continuing book from the split onward.
    """
    lo = pd.Timestamp(split)
    nav = res.daily_nav
    seg = nav.loc[nav.index >= lo]
    if len(seg) < 3:
        return {}
    m = nav_metrics(seg, risk_free_rate=RISK_FREE)
    bench = res.bench_daily_nav
    bseg = bench.loc[bench.index >= lo] if len(bench) else bench
    if len(bseg) > 1:
        m.update(relative_metrics(seg, bseg, overlap_days=rebalance,
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

    # The benchmark comes from the snapshot. Downloading it here made the
    # result depend on what Yahoo served that afternoon, which is the opposite
    # of what a pre-registered test is for.
    try:
        provider = SnapshotProvider(args.baseline_dir)
        provider.download_benchmark("SPY", None, None)
    except (SnapshotIncomplete, FileNotFoundError) as exc:
        sys.exit(f"{exc}\n\nRecord it first: uv run python "
                 f"scripts/record_baseline_benchmark.py {args.baseline_dir}")

    sel = _before(scored, args.split)
    hold_dates = scored.loc[scored["date"] >= pd.Timestamp(args.split), "date"]
    print(f"selection {sel['date'].min().date()}..{sel['date'].max().date()} "
          f"({sel['date'].nunique()} sessions)")
    print(f"holdout   {hold_dates.min().date()}..{hold_dates.max().date()} "
          f"({hold_dates.nunique()} sessions), measured as a continuation")

    combos = [(d, g, r) for d in GRID_DECILE for g in GRID_GROSS
              for r in GRID_REBALANCE]
    print(f"\nSearching {len(combos)} configurations on the selection window "
          f"by {SELECTION_CRITERION}…")

    sel_rows = []
    for d, g, r in combos:
        m = _whole(_run(sel, execution, provider, d, g, r), r)
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

    # One full-panel run per configuration. The committed configuration's
    # holdout figures and the whole-grid control both come out of these, so the
    # control is measured the same way as the headline rather than by a second
    # code path that might not agree.
    print("\nRunning the grid over the whole panel and measuring from the "
          "split forward…")
    hold_rows = []
    held: dict = {}
    for d, g, r in combos:
        m = _segment(_run(scored, execution, provider, d, g, r), args.split, r)
        row = {"decile": d, "gross": g, "rebalance": r,
               "alpha_t": m.get("alpha_t", float("nan")),
               "alpha_ann": m.get("alpha_ann", float("nan")),
               "sharpe": m.get("sharpe", float("nan"))}
        hold_rows.append(row)
        if (d, g, int(r)) == (win.decile, win.gross, int(win.rebalance)):
            held = m

    print("\nThe committed configuration on the holdout:")
    print(f"  CAGR    {held.get('cagr', float('nan')):+.2%}")
    print(f"  Sharpe  {held.get('sharpe', float('nan')):.2f}")
    print(f"  max DD  {held.get('max_drawdown', float('nan')):+.2%}")
    print(f"  beta    {held.get('beta', float('nan')):+.3f}")
    print(f"  alpha   {held.get('alpha_ann', float('nan')):+.2%} "
          f"(HAC t {held.get('alpha_t', float('nan')):+.2f})")

    # The control: what the rest of the grid did on the holdout. If everything
    # works, the winner tells you nothing about the winner.
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
            "holdout_measured_as": "continuation of a whole-panel run",
            "benchmark": "SPY, from the baseline snapshot",
            "committed": {k: float(win[k]) for k in ("decile", "gross", "rebalance")},
            "holdout": {k: (None if not np.isfinite(v) else float(v))
                        for k, v in held.items() if isinstance(v, (int, float))},
            "grid_holdout": hold_df.to_dict("records"),
        }, indent=2))
        print(f"\nReport -> {args.report}")


if __name__ == "__main__":
    main()
