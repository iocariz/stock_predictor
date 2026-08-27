"""Gate a rebuilt baseline before any number from it is quoted.

Every performance figure this project has published was measured on artifacts
whose provenance was not recorded, and several turned out to be wrong for
reasons unrelated to the strategy: a look-ahead in cohort construction, a
scored panel pricing its own fills, an alpha that was mostly the risk-free
rate. The common factor is that none of them failed loudly. They produced a
number, and the number looked plausible.

So a baseline is not trusted because it ran. It is trusted because it passes
these four gates, each of which is a hard failure:

1. **Accounting reconciles exactly.** The NAV the engine reports must equal a
   NAV reconstructed independently from the cohort ledger. A mismatch means
   money appeared or vanished somewhere between the trades and the curve.
2. **No stale fills, and no unaccounted refusals.** Every fill priced against a
   real quote on the session it claims to trade. Refusing to fill a delisted
   name is correct and expected -- demanding zero refusals would only be
   satisfiable by a survivorship-biased panel -- but every refusal must end in
   a stated disposal or an open deferral, never in capital quietly vanishing.
3. **Point-in-time integrity.** No scored row outside index membership on its
   own date, no label whose forward window runs past the data, and complete
   execution coverage for everything scored.
4. **Deterministic rerun.** The same inputs produce byte-identical outputs.
   Anything else means an unrecorded input.

Usage:

    uv run python scripts/verify_baseline.py artifacts/baseline

Exit status is 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from stock_predictor.backtest import BacktestConfig, run_backtest, run_rank_hold_backtest
from stock_predictor.bundle import describe_bundle, price_divergence, validate_execution_panel
from stock_predictor.pit import load_sp500_stints

ACCOUNTING_TOLERANCE = 1e-6
"""Relative. This is float arithmetic on the same quantities, not a modelling
approximation, so anything above this is a real discrepancy."""


class Gate:
    """One check, its verdict, and enough detail to act on a failure."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.notes.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def report(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.name}"
        return "\n".join([head, *(f"        {n}" for n in self.notes)])


# ---------------------------------------------------------------------------
# 1. Accounting
# ---------------------------------------------------------------------------


def gate_accounting(result, config: BacktestConfig, label: str) -> Gate:
    """Rebuild the NAV from the trade ledger and compare it to the reported one.

    The engine debits a cohort's capital at entry and credits ``capital *
    (1 + net_return)`` at exit. Reconstructing that independently is a real
    check: it catches a cohort that was double-credited, one whose capital was
    never returned, and any silent adjustment applied to the curve but not to
    the trades.
    """
    g = Gate(f"accounting reconciles ({label})")
    nav = result.daily_nav
    if not len(nav):
        g.fail("no NAV series")
        return g

    cash = float(config.initial_capital)
    settled_pnl = 0.0
    open_cohort_capital = 0.0
    last_date = nav.index[-1]
    for c in result.cohorts:
        if pd.Timestamp(c.exit_date) <= last_date:
            settled_pnl += c.capital * c.net_return
        else:
            open_cohort_capital += c.capital

    m = result.metrics
    reported = float(nav.iloc[-1])

    # The identity being asserted:
    #
    #     final NAV = initial capital
    #               + P&L of everything that closed
    #               + unrealized P&L of whatever is still held
    #
    # Positions still open at the end never appear in the closed-trade list,
    # so without the last term the ledger cannot explain the curve -- which is
    # how the first run of this gate "failed" rank-hold by 17% while the engine
    # was right. The engines report the open leg so the identity is closeable.
    open_unrealized = (
        float(m.get("open_position_value", 0.0))
        - float(m.get("open_position_basis", 0.0))
    )
    n_open = int(m.get("n_open_positions", 0))

    if n_open or "open_position_value" in m:
        expected = cash + settled_pnl + open_unrealized
        rel = abs(reported - expected) / max(abs(expected), 1.0)
        g.note(f"final NAV {reported:,.2f} vs ledger {expected:,.2f} "
               f"({n_open} open, unrealized {open_unrealized:+,.2f}, rel {rel:.2e})")
        if rel > ACCOUNTING_TOLERANCE:
            g.fail(f"NAV does not reconcile with the trade ledger (rel {rel:.2e})")
    elif open_cohort_capital == 0.0:
        expected = cash + settled_pnl
        rel = abs(reported - expected) / max(abs(expected), 1.0)
        g.note(f"final NAV {reported:,.2f} vs ledger {expected:,.2f} (rel {rel:.2e})")
        if rel > ACCOUNTING_TOLERANCE:
            g.fail(f"NAV does not reconcile with the trade ledger (rel {rel:.2e})")
    else:
        # A cohort open past the end of the calendar is marked to market by the
        # NAV builder; the ledger alone cannot reproduce that mark, so this
        # gate cannot make an exact claim and says so rather than passing
        # quietly on a weaker one.
        g.fail(f"{open_cohort_capital:,.2f} still deployed at {last_date.date()} "
               "with no open-position mark reported; ledger not closeable")

    # A NAV that jumps without a trade is the other half of the same question.
    ret = nav.pct_change().dropna()
    if len(ret) and ret.abs().max() > 0.5:
        when = ret.abs().idxmax()
        g.fail(f"implausible one-day NAV move {ret.loc[when]:+.1%} on {when.date()}")

    if (nav <= 0).any():
        g.fail("NAV touches zero or below")
    return g


# ---------------------------------------------------------------------------
# 2. Fills
# ---------------------------------------------------------------------------


def gate_fills(result, label: str) -> Gate:
    g = Gate(f"fills are real and every refusal is accounted for ({label})")
    m = result.metrics
    requested = int(m.get("fills_requested", 0))
    rejected = int(m.get("fills_rejected", 0))
    stale = int(m.get("stale_fills", 0))
    deferred = int(m.get("exits_deferred", 0))
    disposals = int(sum(v for k, v in m.items() if k.startswith("disposals_")))
    g.note(f"requested {requested}, rejected {rejected}, stale {stale}, "
           f"deferred {deferred}, disposed {disposals}")
    if requested == 0:
        g.fail("no fills were even attempted")

    # A *stale* fill is a trade that did not happen: a carried-forward price
    # standing in for a quote. Zero, always.
    if stale:
        g.fail(f"{stale} fill(s) priced from a carried-forward quote")

    # A *rejected* fill is the opposite -- the engine refusing to invent a
    # price for a name that stopped trading. Demanding zero of those would
    # only be satisfiable by a panel with no delistings in it, which is the
    # survivorship bias this whole exercise exists to remove. On the 2019-2026
    # panel the rejections are names like MRO, DFS, JNPR and HES: real
    # acquisitions, correctly refused and then disposed of by policy.
    #
    # What must hold is that every rejection was *accounted for* -- disposed
    # by evidence or by the stated fallback, or still deferred and reported --
    # rather than silently dropped.
    unaccounted = rejected - disposals - deferred
    if unaccounted > 0:
        g.fail(f"{unaccounted} rejected fill(s) neither disposed nor deferred; "
               "capital vanished without a stated policy")
    for k, v in sorted(m.items()):
        if k.startswith("disposals_") and v:
            g.note(f"{k}: {int(v)}")
    return g


# ---------------------------------------------------------------------------
# 3. Point-in-time integrity
# ---------------------------------------------------------------------------


def gate_pit(scored: pd.DataFrame, execution: pd.DataFrame, horizon: int) -> Gate:
    g = Gate("point-in-time integrity")

    try:
        stints = load_sp500_stints()
    except Exception as exc:  # noqa: BLE001 - a check that cannot run is a failure
        g.fail(f"could not load membership stints: {exc}")
        return g

    work = scored[["date", "ticker"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str)

    st = stints.copy()
    st["ticker"] = st["ticker"].astype(str)
    st["start_date"] = pd.to_datetime(st["start_date"])
    st["end_date"] = pd.to_datetime(st["end_date"]).fillna(pd.Timestamp.max)

    merged = work.merge(st, on="ticker", how="left")
    inside = (merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])
    ok_pairs = merged[inside][["date", "ticker"]].drop_duplicates()
    all_pairs = work.drop_duplicates()
    outside = len(all_pairs) - len(ok_pairs)
    g.note(f"{len(all_pairs):,} scored (date, ticker) pairs; {outside:,} outside membership")
    if outside:
        bad = all_pairs.merge(ok_pairs, on=["date", "ticker"], how="left", indicator=True)
        sample = bad[bad["_merge"] == "left_only"].head(5)
        g.fail(f"{outside} scored row(s) fall outside index membership, e.g. "
               + ", ".join(f"{r.ticker}@{pd.Timestamp(r.date).date()}"
                           for r in sample.itertuples()))

    # A label needs a full forward window. One that runs past the data is a
    # label built from prices that do not exist yet.
    if "target_5pct" in scored.columns:
        labelled = scored.dropna(subset=["target_5pct"])
        if len(labelled):
            sessions = pd.DatetimeIndex(sorted(work["date"].unique()))
            last_labelable = sessions[-(horizon + 1)] if len(sessions) > horizon else sessions[0]
            latest = pd.to_datetime(labelled["date"]).max()
            g.note(f"latest labelled session {latest.date()}, "
                   f"last labelable {pd.Timestamp(last_labelable).date()}")
            if latest > last_labelable:
                g.fail(f"labels exist through {latest.date()} but only "
                       f"{pd.Timestamp(last_labelable).date()} has a full "
                       f"{horizon}-session forward window")

    findings = validate_execution_panel(scored, execution)
    if findings:
        g.fail(describe_bundle(findings).replace("\n", "\n        "))
    else:
        g.note("execution panel covers every scored row")

    div = price_divergence(scored, execution)
    if div["compared"]:
        g.note(f"scored vs execution prices: {int(div['disagreeing'])}/"
               f"{int(div['compared'])} differ (max {div['max_abs_pct']:.2%})")
    return g


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def _hash_series(s: pd.Series) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(s.round(10), index=True).values.tobytes()
    ).hexdigest()[:16]


def gate_determinism(scored, execution, config, engine, label: str) -> Gate:
    """Same inputs, same outputs. Twice, in one process, then compared by hash."""
    g = Gate(f"deterministic rerun ({label})")
    a = engine(scored, config, execution_prices=execution)
    b = engine(scored, config, execution_prices=execution)
    ha, hb = _hash_series(a.daily_nav), _hash_series(b.daily_nav)
    g.note(f"NAV hash {ha} / {hb}")
    if ha != hb:
        g.fail("two runs over identical inputs produced different NAV series")
    for key in ("cagr", "sharpe", "max_drawdown", "n_cohorts"):
        va, vb = a.metrics.get(key), b.metrics.get(key)
        if va != vb:
            g.fail(f"metric {key} differs between runs: {va} vs {vb}")
    return g


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("baseline_dir", type=Path)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--max-cohorts", type=int, default=2)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--rebalance-day", default="Friday")
    ap.add_argument("--exit-rank", type=int, default=40)
    args = ap.parse_args()

    d = args.baseline_dir
    scores_path = d / "wf_scored.parquet"
    exec_path = d / "execution_prices.parquet"
    for p in (scores_path, exec_path):
        if not p.exists():
            sys.exit(f"missing {p}; run scripts/rebuild_baseline.sh first")

    scored = pd.read_parquet(scores_path)
    scored["date"] = pd.to_datetime(scored["date"])
    execution = pd.read_parquet(exec_path)

    print(f"scored     {len(scored):,} rows, "
          f"{scored['date'].nunique():,} sessions, "
          f"{scored['ticker'].nunique():,} tickers")
    print(f"execution  {execution.shape[0]:,} sessions x {execution.shape[1]:,} tickers")
    print()

    common = dict(top_n=args.top_n, holding_days=args.horizon,
                  max_overlapping_cohorts=args.max_cohorts,
                  slippage_bps=args.slippage_bps, benchmark_ticker=None,
                  rebalance_day=args.rebalance_day, reject_stale_fills=True)

    gates: list[Gate] = [gate_pit(scored, execution, args.horizon)]

    for label, engine, extra in (
        ("cohort", run_backtest, {}),
        ("rank-hold", run_rank_hold_backtest, {"exit_rank": args.exit_rank}),
    ):
        cfg = BacktestConfig(**common, **extra)
        res = engine(scored, cfg, execution_prices=execution)
        gates.append(gate_accounting(res, cfg, label))
        gates.append(gate_fills(res, label))
        gates.append(gate_determinism(scored, execution, cfg, engine, label))
        m = res.metrics
        print(f"{label:10s} CAGR {m.get('cagr', float('nan')):7.2%}  "
              f"Sharpe {m.get('sharpe', float('nan')):5.2f}  "
              f"maxDD {m.get('max_drawdown', float('nan')):7.2%}  "
              f"trades {int(m.get('n_cohorts', 0))}")
    print()

    for g in gates:
        print(g.report())

    failed = [g for g in gates if not g.passed]
    print()
    if failed:
        print(f"{len(failed)} of {len(gates)} gates FAILED. "
              "No number from this baseline should be quoted.")
        sys.exit(1)
    print(f"All {len(gates)} gates passed.")

    manifest = d / "snapshot" / "manifest.json"
    cfg_path = d / "config.json"
    if manifest.exists():
        man = json.loads(manifest.read_text())
        print(f"Baseline run {man.get('run_id', '?')} "
              f"commit {man.get('git_commit', '?')[:12]}"
              + ("  (DIRTY TREE)" if man.get("git_dirty") else ""))
        for name, meta in sorted(man.get("snapshots", {}).items()):
            print(f"  {name:22s} {meta.get('sha256', '?')[:16]}  "
                  f"{meta.get('rows', '?')} rows")
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("git_dirty"):
            print("WARNING: built from a dirty working tree; not reproducible "
                  "from the recorded commit alone.")


if __name__ == "__main__":
    main()
