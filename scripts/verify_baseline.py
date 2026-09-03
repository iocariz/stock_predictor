"""Gate a rebuilt baseline before any number from it is quoted.

Every performance figure this project has published was measured on artifacts
whose provenance was not recorded, and several turned out to be wrong for
reasons unrelated to the strategy: a look-ahead in cohort construction, a
scored panel pricing its own fills, an alpha that was mostly the risk-free
rate. The common factor is that none of them failed loudly. They produced a
number, and the number looked plausible.

So a baseline is not trusted because it ran. It is trusted because it passes
these five gates, each of which is a hard failure:

0. **The artifacts are the ones the run produced.** Inputs were hashed from the
   start, and since the last round the manifest's snapshot hashes are actually
   recomputed. But the two files this verifier *reads* -- ``wf_scored.parquet``
   and ``execution_prices.parquet`` -- were hashed by nothing, and swapping the
   scores for a forgery nudged 12% toward the realised forward return passed
   every other gate here while printing a 98% cohort CAGR. Both are now hashed
   at write time; on top of that the execution panel is re-derived from the
   hashed snapshot it was pivoted from, which ties it to an input hash rather
   than to a promise and survives a manifest rewritten to match.
1. **Accounting reconciles exactly.** The NAV the engine reports must equal a
   NAV reconstructed independently from the cohort ledger. A mismatch means
   money appeared or vanished somewhere between the trades and the curve.
2. **No stale fills, and no unaccounted refusals.** Every fill priced against a
   real quote on the session it claims to trade. Refusing to fill a delisted
   name is correct and expected -- demanding zero refusals would only be
   satisfiable by a survivorship-biased panel -- but every refusal must end in
   a stated disposal or an open deferral, never in capital quietly vanishing.
3. **Point-in-time integrity, and no survivorship.** No scored row outside index
   membership on its own date, no label whose forward window runs past the
   data, complete execution coverage for everything scored, and every company
   that left the index during the window present *with prices* -- counted by
   rows, not by column, because an empty column passes a presence check.
4. **Deterministic backtest.** The same scored panel produces byte-identical
   NAV. Note the scope: this does not make the *pipeline* reproducible. Two
   rebuilds from one commit and one pinned window differ by several points of
   CAGR, because vendor float noise flips tree splits. See
   :func:`gate_determinism`.

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
from stock_predictor.providers.hybrid_provider import DEFAULT_CACHE
from stock_predictor.renames import rename_coverage
from stock_predictor.replay import SnapshotIncomplete, SnapshotProvider

ACCOUNTING_TOLERANCE = 1e-6
"""Relative. This is float arithmetic on the same quantities, not a modelling
approximation, so anything above this is a real discrepancy."""


def read_manifest_key(baseline_dir: Path, key: str):
    """A recorded build fact, or ``None``. Missing is not the same as empty."""
    path = Path(baseline_dir) / "snapshot" / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get(key)
    except (json.JSONDecodeError, OSError):
        return None


def load_snapshot_stints(baseline_dir: Path) -> pd.DataFrame:
    """Index membership **as recorded with this baseline**.

    Not the live table. The verdict on a fixed set of artifacts must not move
    because membership changed, or because the rename map changed -- resolving
    ticker renames alone rewrote 12 stints, which would have silently
    re-scored every baseline built before it.
    """
    path = Path(baseline_dir) / "snapshot" / "stints.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no snapshot stints at {path}; this baseline cannot be verified "
            "reproducibly. Rebuild it with scripts/rebuild_baseline.sh."
        )
    return pd.read_parquet(path)


def verify_snapshot_hashes(baseline_dir: Path) -> "Gate":
    """Recompute every snapshot's sha256 and compare it to the manifest.

    The manifest recorded hashes and nothing ever checked them, so a corrupted
    or swapped artifact verified clean. A list of hashes nobody recomputes is
    a list of hashes.
    """
    g = Gate("snapshot artifacts match their recorded hashes")
    manifest = Path(baseline_dir) / "snapshot" / "manifest.json"
    if not manifest.exists():
        g.fail(f"no manifest at {manifest}; integrity cannot be established")
        return g
    try:
        man = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        g.fail(f"manifest is unreadable: {exc}")
        return g

    snaps = man.get("snapshots", {})
    if not snaps:
        g.fail("manifest records no snapshots")
        return g

    for name, meta in sorted(snaps.items()):
        recorded = str(meta.get("sha256", ""))
        path = Path(baseline_dir) / "snapshot" / f"{name}.parquet"
        if not path.exists():
            g.fail(f"{name}: recorded in the manifest but missing from disk")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != recorded:
            g.fail(f"{name}: sha256 {actual[:16]} does not match the recorded "
                   f"{recorded[:16]}")
        else:
            g.note(f"{name:22s} {actual[:16]}  {meta.get('rows', '?')} rows")
    return g


OUTPUTS = ("wf_scored", "execution_prices")
"""The artifacts this verifier reads. Hashing the inputs and then evaluating
unhashed outputs checks the wrong files: a ``wf_scored.parquet`` whose scores
were nudged toward the realised forward return passed all twelve gates while
printing a 98% cohort CAGR against the real ~20%."""


def gate_output_hashes(baseline_dir: Path) -> "Gate":
    """Recompute the sha256 of every artifact the gates below evaluate.

    ``specs.md:334`` requires output hashes alongside the input hashes. They
    were the one class missing, and outputs are precisely what a reader of this
    report cares about.
    """
    g = Gate("baseline outputs match their recorded hashes")
    d = Path(baseline_dir)
    outputs = read_manifest_key(d, "outputs")
    if not outputs:
        g.fail("the manifest records no output hashes, so these artifacts "
               "cannot be vouched for. Rebuild, or seal an older baseline "
               "with scripts/seal_baseline_outputs.py.")
        return g

    for name in OUTPUTS:
        meta = outputs.get(name)
        path = d / f"{name}.parquet"
        if not meta:
            g.fail(f"{name}: evaluated by this verifier but not recorded")
            continue
        if not path.exists():
            g.fail(f"{name}: recorded in the manifest but missing from disk")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = str(meta.get("sha256", ""))
        if actual != recorded:
            g.fail(f"{name}: sha256 {actual[:16]} does not match the recorded "
                   f"{recorded[:16]}; this is not the file the run produced")
            continue
        prov = str(meta.get("provenance", "unknown"))
        g.note(f"{name:22s} {actual[:16]}  ({prov})")

    # Sealing records the bytes as they stood when someone ran the sealer, not
    # as they came out of the pipeline. That is worth having and is not the
    # same claim, so it is never reported as though it were.
    sealed = [n for n in OUTPUTS
              if str(outputs.get(n, {}).get("provenance", "")).startswith("sealed")]
    if sealed:
        g.note(f"NOTE: {', '.join(sealed)} sealed after the fact — tamper-evident "
               "from the seal onward, but not proof of provenance")
    return g


def gate_execution_derivation(baseline_dir: Path) -> "Gate":
    """The root execution panel must be the pivot of the hashed snapshot.

    Stronger than a recorded hash, because it ties the output to an input hash
    rather than to a promise: it holds on baselines built before output hashing
    existed, and it survives a manifest that was rewritten alongside the file.

    ``pivot_table`` drops all-NaN columns, so the wide panel legitimately
    carries names the long snapshot does not -- 70 of them in the real baseline,
    every one empty. A column carrying *data* the snapshot never held is
    fabricated, and that is the distinction drawn here.
    """
    g = Gate("execution panel derives from the hashed snapshot")
    d = Path(baseline_dir)
    src = d / "snapshot" / "execution_prices.parquet"
    out = d / "execution_prices.parquet"
    if not src.exists():
        g.fail(f"no {src}; the execution panel's provenance cannot be checked")
        return g
    if not out.exists():
        g.fail(f"no {out}")
        return g

    long = pd.read_parquet(src)
    long["date"] = pd.to_datetime(long["date"])
    value_col = "close" if "close" in long.columns else "adj_close"
    expected = long.pivot_table(index="date", columns="ticker",
                                values=value_col, aggfunc="first").sort_index()
    actual = pd.read_parquet(out)
    actual.index = pd.to_datetime(actual.index)
    actual = actual.sort_index()

    if not actual.index.equals(expected.index):
        g.fail(f"sessions differ: output has {len(actual):,}, snapshot implies "
               f"{len(expected):,}")
        return g

    missing = [c for c in expected.columns if c not in actual.columns]
    if missing:
        g.fail(f"{len(missing)} ticker(s) in the snapshot are absent from the "
               f"output: {', '.join(map(str, missing[:8]))}")
        return g

    invented = [c for c in actual.columns
                if c not in expected.columns and actual[c].notna().any()]
    if invented:
        g.fail(f"{len(invented)} ticker(s) carry prices the snapshot never "
               f"held: {', '.join(map(str, invented[:8]))}")

    aligned = actual.reindex(columns=expected.columns)
    if not aligned.isna().equals(expected.isna()):
        n = int((aligned.isna() != expected.isna()).sum().sum())
        g.fail(f"{n:,} cell(s) present in one and absent in the other")
    diff = (aligned - expected).abs().max().max()
    if pd.notna(diff) and float(diff) > 0.0:
        g.fail(f"prices diverge from the snapshot by up to {float(diff):.6g}")

    empty = len(actual.columns) - len(expected.columns)
    g.note(f"{actual.shape[0]:,} sessions x {expected.shape[1]:,} priced tickers "
           f"reproduced exactly from the hashed snapshot"
           + (f" (+{empty} empty column(s))" if empty > 0 else ""))
    return g


PINNED_METRICS = ("cagr", "sharpe", "max_drawdown", "beta", "alpha_ann", "alpha_t")
"""What a reader of BASELINE.md actually quotes. Anything published has to be
checkable, or publishing it is an assertion rather than a measurement."""

DEFAULT_TOLERANCE = {
    "cagr": 5e-5, "sharpe": 5e-3, "max_drawdown": 5e-5,
    "beta": 5e-3, "alpha_ann": 5e-5, "alpha_t": 5e-3,
}
"""Tight on purpose. Given the same artifacts and the same code these are
deterministic -- the determinism gate proves the NAV is byte-identical -- so
the only slack needed is float summation order across library versions. A drift
worth arguing about is thousands of times larger than this."""

RISK_FREE = 0.045


def measure_engines(scored, execution, *, provider, top_n: int, horizon: int,
                    max_cohorts: int, slippage_bps: float, rebalance_day: str,
                    exit_rank: int) -> dict[str, dict]:
    """Run all three engines and return both the results and their metrics.

    Shared with ``scripts/pin_baseline_metrics.py`` so that what gets pinned and
    what gets checked are the same computation rather than two that agree by
    inspection.
    """
    from stock_predictor.backtest_reporting import relative_metrics
    from stock_predictor.long_short import LongShortConfig, run_long_short_backtest

    bench_ticker = "SPY" if provider is not None else None
    out: dict[str, dict] = {}

    ls_cfg = LongShortConfig(
        rebalance_every=horizon, slippage_bps=slippage_bps,
        benchmark_ticker=bench_ticker, risk_free_rate=RISK_FREE,
        reject_stale_fills=True,
    )
    ls = run_long_short_backtest(scored, ls_cfg, provider=provider,
                                 execution_prices=execution)
    ls_metrics = dict(ls.metrics)
    if len(getattr(ls, "bench_daily_nav", ())) > 1:
        ls_metrics.update(relative_metrics(
            ls.daily_nav, ls.bench_daily_nav,
            overlap_days=horizon, risk_free_rate=RISK_FREE))
    out["long-short"] = {"result": ls, "config": ls_cfg, "metrics": ls_metrics}

    common = dict(top_n=top_n, holding_days=horizon,
                  max_overlapping_cohorts=max_cohorts,
                  slippage_bps=slippage_bps, benchmark_ticker=bench_ticker,
                  rebalance_day=rebalance_day, reject_stale_fills=True)
    for label, engine, extra in (
        ("cohort", run_backtest, {}),
        ("rank-hold", run_rank_hold_backtest, {"exit_rank": exit_rank}),
    ):
        cfg = BacktestConfig(**common, **extra)
        res = engine(scored, cfg, provider=provider, execution_prices=execution)
        metrics = dict(res.metrics)
        bench_nav = getattr(res, "spy_daily_nav", None)
        if bench_nav is not None and len(bench_nav) > 1:
            metrics.update(relative_metrics(
                res.daily_nav, bench_nav,
                overlap_days=horizon, risk_free_rate=RISK_FREE))
        out[label] = {"result": res, "config": cfg, "metrics": metrics}
    return out


def gate_expected_metrics(baseline_dir: Path, measured: dict[str, dict]) -> "Gate":
    """Compare this run's headline figures against the ones pinned to it.

    Every other gate here checks the baseline against *itself*: that it
    reconciles, that it fills honestly, that its bytes are what was recorded.
    None of them checks it against what was **published** about it, and that is
    how the artifact came to be swapped without a word. At ``c656df9`` the
    baseline was rebuilt -- new run id, new commit, every snapshot hash
    different, 37,156 fewer labelled rows -- and BASELINE.md's results tables
    were left describing the artifact that had just been replaced. Every gate
    passed, because none of them was looking.
    """
    g = Gate("headline metrics match the values pinned to this baseline")
    path = Path(baseline_dir) / "expected_metrics.json"
    if not path.exists():
        g.fail(f"no {path.name}; this baseline's published figures are not "
               "checkable. Pin them with scripts/pin_baseline_metrics.py.")
        return g
    try:
        pin = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        g.fail(f"{path.name} is unreadable: {exc}")
        return g

    # A pin taken against a different artifact is worse than no pin: it looks
    # like verification and asserts nothing.
    run_id = read_manifest_key(baseline_dir, "run_id")
    if pin.get("run_id") and run_id and pin["run_id"] != run_id:
        g.fail(f"pinned against run {pin['run_id']}, but these artifacts are "
               f"{run_id}. Re-pin, or restore the artifacts.")
        return g

    tol = {**DEFAULT_TOLERANCE, **(pin.get("tolerance") or {})}
    engines = pin.get("engines") or {}
    if not engines:
        g.fail(f"{path.name} pins no engines")
        return g

    for label in sorted(engines):
        expected = engines[label]
        if label not in measured:
            g.fail(f"{label}: pinned but not measured by this verifier")
            continue
        actual = measured[label]["metrics"]
        for key in PINNED_METRICS:
            want = expected.get(key)
            if want is None:
                continue
            got = actual.get(key)
            if got is None or not pd.notna(got):
                g.fail(f"{label}.{key}: pinned at {want:+.6g} but not measured")
                continue
            if abs(float(got) - float(want)) > tol.get(key, 5e-5):
                g.fail(f"{label}.{key}: {float(got):+.6g} vs pinned "
                       f"{float(want):+.6g} "
                       f"(drift {float(got) - float(want):+.4g})")
        g.note(f"{label:11s} CAGR {actual.get('cagr', float('nan')):7.2%}  "
               f"alpha {actual.get('alpha_ann', float('nan')):+7.2%}  "
               f"t {actual.get('alpha_t', float('nan')):+5.2f}")

    prov = str(pin.get("provenance", "unknown"))
    g.note(f"pinned {pin.get('pinned_at_utc', '?')} at commit "
           f"{str(pin.get('pinned_at_commit', '?'))[:12]} ({prov})")
    return g


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

    # specs.md:414 -- cash and holdings MUST reconcile on *every* session. The
    # terminal identity above is one equation about one day; a curve that ends
    # right and wanders in between satisfied it. This is the same identity
    # asserted 1,924 times instead of once.
    cash = getattr(result, "daily_cash", None)
    held = getattr(result, "daily_positions", None)
    if cash is None or held is None:
        g.fail("engine reports no per-session ledger; only the final identity "
               "could be checked")
        return g

    recomputed = cash + held
    resid = (recomputed - nav).abs()
    scale = float(max(abs(float(nav.iloc[0])), 1.0))
    worst = float(resid.max()) if len(resid) else 0.0
    if worst > ACCOUNTING_TOLERANCE * scale:
        when = resid.idxmax()
        g.fail(f"cash + holdings != NAV on {pd.Timestamp(when).date()}: "
               f"{float(recomputed.loc[when]):,.2f} vs {float(nav.loc[when]):,.2f} "
               f"({int((resid > ACCOUNTING_TOLERANCE * scale).sum())} session(s) "
               "disagree)")
    else:
        g.note(f"cash + holdings = NAV on all {len(nav):,} sessions "
               f"(worst residual {worst:.2e})")

    # specs.md:417 -- fees applied after sizing must not overdraw the account.
    # Only meaningful for a long-only book: short proceeds legitimately carry
    # cash above capital, and a levered book can legitimately run it negative.
    if float(held.min()) >= 0.0 and float(cash.min()) < -ACCOUNTING_TOLERANCE * scale:
        when = cash.idxmin()
        g.fail(f"cash is negative ({float(cash.min()):,.2f}) on "
               f"{pd.Timestamp(when).date()}")
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


def gate_pit(scored: pd.DataFrame, execution: pd.DataFrame, horizon: int,
             stints: pd.DataFrame) -> Gate:
    g = Gate("point-in-time integrity")

    work = scored[["date", "ticker"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ticker"] = work["ticker"].astype(str)

    st = stints.copy()
    st["ticker"] = st["ticker"].astype(str)
    st["start_date"] = pd.to_datetime(st["start_date"])
    st["end_date"] = pd.to_datetime(st["end_date"]).fillna(pd.Timestamp.max)

    merged = work.merge(st, on="ticker", how="left")
    # Half-open, matching production: pit.filter_panel_to_pit keeps
    # [start_date, end_date). Accepting `<= end_date` here made the gate one
    # session looser than the filter it exists to police, so a regression that
    # scored a company on its first *non*-member session would have passed.
    inside = (merged["date"] >= merged["start_date"]) & (merged["date"] < merged["end_date"])
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

    # specs.md:240. The panel is clean today; asserting it keeps it that way
    # through a future merge that fans out or a hand-built input. A duplicated
    # row becomes two holdings of one company at half weight each, and every
    # count downstream reports it as two positions.
    dup = scored.duplicated(subset=[ "date", "ticker"], keep=False)
    n_dup = int(dup.sum())
    g.note(f"{len(scored):,} rows, "
           f"{len(scored[['date', 'ticker']].drop_duplicates()):,} unique (date, ticker)")
    if n_dup:
        g.fail(f"{n_dup} duplicate (date, ticker) row(s); one ticker scored "
               "twice on a date becomes two holdings of the same company")

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
# 3b. Survivorship
# ---------------------------------------------------------------------------


MIN_SCORED_COVERAGE = 0.95
"""Priced departed names must actually be scorable while they were members.

The old check asked only whether *some* departed name reached the scored panel,
which a badly truncated panel also satisfies."""

MIN_ROWS_TO_COUNT = 20
"""A column with a handful of prints is not a recovered ticker."""


def _vendor_absent(cache_dir: Path) -> set[str]:
    """Tickers the vendor was asked for and cannot usefully serve.

    Read from the cache rather than guessed. This is what makes the gate's bar
    the *measured* ceiling instead of a number chosen to pass: a name missing
    because nobody fetched it fails, a name missing because it does not exist
    upstream is tolerated and listed.

    "Cannot serve" includes a stub. Tiingo returns 1-12 rows for BK, CDAY,
    CSRA, DF, PEAK and WRK -- mostly renames and mergers -- and returns the
    same on a refetch. A manifest entry alone would call those recovered,
    because the frame was not literally empty; counting the rows is what
    distinguishes "the vendor has nothing" from "we ran out of quota", and
    only the second is fixable.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / "_manifest.json"
    if not path.exists():
        return set()
    try:
        man = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()

    absent: set[str] = set()
    for t, e in man.items():
        if not isinstance(e, dict):
            continue
        if e.get("empty"):
            absent.add(t)
            continue
        cached = cache_dir / f"{t}.parquet"
        if not cached.exists():
            continue
        try:
            if len(pd.read_parquet(cached)) < MIN_ROWS_TO_COUNT:
                absent.add(t)          # fetched; a stub is all there is
        except Exception:  # noqa: BLE001 - unreadable means refetch, not tolerate
            continue
    return absent


def gate_survivorship(execution: pd.DataFrame, scored: pd.DataFrame,
                      stints: pd.DataFrame, absent: set[str],
                      absent_source: str, recycled: set[str] | None = None) -> Gate:
    """Are the companies that left the index actually in the panel?

    Checking that the *column* exists is not enough, and that mistake has
    already been made once here: a rate-limited rebuild produced an execution
    panel with all 358 departed names present as columns and 148 of them
    entirely empty. Column presence passed; the panel was survivorship-biased
    anyway, and the scored panel silently lost those names. So this counts
    prices, not columns.

    The bar is the measured ceiling, not a percentage. A departed name missing
    because the quota ran out is a failure -- it is recoverable, and shipping
    without it biases every return. A name missing because the vendor has no
    data for it at all is tolerated, named in the output, and written to
    ``survivorship_gap.json`` beside the baseline so the residual is auditable
    rather than folded into a threshold.
    """
    g = Gate("survivorship: departed names are present with data")
    g.note(f"vendor-absent set from {absent_source}")
    st = stints.copy()
    st["ticker"] = st["ticker"].astype(str)
    st["end_date"] = pd.to_datetime(st["end_date"])
    lo, hi = execution.index.min(), execution.index.max()
    departed = st[(st["end_date"].notna())
                  & (st["end_date"] >= lo) & (st["end_date"] <= hi)]
    names = sorted(set(departed["ticker"]))
    if not names:
        g.fail("no departed names found in the window; check the stint data")
        return g

    # Priced *while it was a member*. Counting any prices at all credited
    # reused symbols as survivorship recoveries -- Qwest's Q "recovered" by a
    # different company's 2025 listing -- and put reported coverage 15.6
    # points above the truth, 91.4% against 75.8%.
    priced, empty = [], []
    for t in names:
        if t not in execution.columns:
            empty.append(t)
            continue
        col = execution[t]
        real = col[(col.notna()) & (col > 0)]
        rows = st[st["ticker"] == t]
        inside = pd.Series(False, index=real.index)
        for r in rows.itertuples():
            e0 = pd.Timestamp(r.end_date)
            inside |= ((real.index >= pd.Timestamp(r.start_date))
                       & (real.index <= (e0 if pd.notna(e0) else real.index.max())))
        if int(inside.sum()) >= MIN_ROWS_TO_COUNT:
            priced.append(t)
        else:
            empty.append(t)

    # Three reasons a departed name can be missing, and only one is fixable.
    #   absent    the vendor has nothing, or a stub
    #   reused    the vendor has prices, but for whoever got the symbol next --
    #             refetching returns the same wrong company
    #   missing   nobody fetched it; this is the one that fails
    # Reused symbols are read from the manifest where the build recorded them.
    # Detecting them from the panel only worked before the contaminated prices
    # were removed; afterwards the column is empty and indistinguishable from
    # one nobody fetched.
    recorded = set(recycled or ())
    reused = sorted(
        t for t in empty
        if t not in absent and (
            t in recorded or (
                t in execution.columns
                and int(((execution[t].notna()) & (execution[t] > 0)).sum())
                >= MIN_ROWS_TO_COUNT
            )
        )
    )
    unavailable = sorted(set(t for t in empty if t in absent) | set(reused))
    recoverable = sorted(t for t in empty if t not in unavailable)
    coverage = len(priced) / len(names)
    ceiling = (len(names) - len(unavailable)) / len(names)

    g.note(f"{len(names)} names left the index in-window; {len(priced)} carry "
           f"prices ({coverage:.1%}); vendor ceiling {ceiling:.1%}")
    if unavailable:
        g.note(f"{len(unavailable)} unavailable upstream (tolerated): "
               + ", ".join(unavailable[:12]) + (" …" if len(unavailable) > 12 else ""))
    if reused:
        g.note(f"  of which {len(reused)} are reused symbols now held by another "
               f"issuer: " + ", ".join(reused[:10])
               + (" …" if len(reused) > 10 else ""))
    if recoverable:
        g.fail(f"{len(recoverable)} departed name(s) are recoverable but absent "
               f"-- refetch before quoting anything: {recoverable[:12]}")

    # Per name and date, not "at least one". The old check passed as long as a
    # single departed name reached the scored panel, which a badly truncated
    # panel would also satisfy. What matters is whether each priced departed
    # name is actually *scorable* on the sessions it was a member.
    scored_dates = pd.DatetimeIndex(sorted(pd.to_datetime(scored["date"]).unique()))
    by_ticker = scored.assign(date=pd.to_datetime(scored["date"])).groupby(
        scored["ticker"].astype(str)
    )["date"].apply(lambda x: set(x))
    lo, hi = scored_dates.min(), scored_dates.max()

    eligible = missing_rows = 0
    thin: list[tuple[str, float]] = []
    for t in priced:
        rows = st[st[ "ticker"] == t]
        want: set = set()
        for r in rows.itertuples():
            s0 = max(pd.Timestamp(r.start_date), lo)
            e0 = min(pd.Timestamp(r.end_date), hi)
            if pd.isna(s0) or pd.isna(e0) or s0 > e0:
                continue
            want |= set(scored_dates[(scored_dates >= s0) & (scored_dates <= e0)])
        if not want:
            continue
        have = by_ticker.get(t, set()) & want
        eligible += len(want)
        missing_rows += len(want) - len(have)
        ratio = len(have) / len(want)
        if ratio < 0.5:
            thin.append((t, ratio))

    covered = 1.0 - (missing_rows / eligible) if eligible else 1.0
    g.note(f"departed names scored on {covered:.1%} of the sessions they were "
           f"members ({eligible - missing_rows:,}/{eligible:,} name-sessions)")
    if thin:
        worst = ", ".join(f"{t} {r:.0%}" for t, r in sorted(thin, key=lambda x: x[1])[:8])
        g.note(f"{len(thin)} priced departed name(s) scored on under half their "
               f"member sessions: {worst}")
    if covered < MIN_SCORED_COVERAGE:
        g.fail(f"priced departed names reach the scored panel on only "
               f"{covered:.1%} of their member sessions")

    g.residual = {                                    # written out by main()
        "departed_in_window": len(names),
        "with_prices": len(priced),
        "coverage": round(coverage, 4),
        "vendor_ceiling": round(ceiling, 4),
        "unavailable_upstream": unavailable,
        "reused_symbols": reused,
        "recoverable_but_absent": recoverable,
        "scored_session_coverage": round(covered, 4),
        "eligible_name_sessions": eligible,
    }
    return g


# ---------------------------------------------------------------------------
# 3c. Renames
# ---------------------------------------------------------------------------


MIN_RENAME_COVERAGE = 0.99
"""A rename carries the company's history forward. Anything less is a different
corporate event."""


def gate_renames(stints: pd.DataFrame, execution: pd.DataFrame) -> Gate:
    """Prove the rename map on *this* baseline's data, not on fixtures.

    The map was applied unconditionally at load time while its validation
    function ran only in synthetic tests -- a claim, not a gate. Every entry is
    now checked against the panel that was actually built.

    This needs the predecessor symbols, which canonicalisation replaces. They
    survive in the ``alias`` column; a baseline built before that column
    existed cannot prove its own mapping and is failed rather than passed on
    the assumption that it was fine.

    What this can establish, precisely: the successor prices the predecessor's
    membership (necessary), and the two never trade concurrently after the
    effective date (a real falsifier). What it cannot: that they are the same
    issuer. That needs a permanent identifier this project does not carry, and
    each entry's recorded note remains the actual warrant.
    """
    g = Gate("ticker renames are supported by the baseline's own prices")
    if "alias" not in stints.columns:
        g.fail("snapshot stints carry no alias column, so the renames applied "
               "to this baseline cannot be checked from it -- rebuild")
        return g

    rows = []
    for r in stints.itertuples():
        for a in (str(r.alias).split("|") if r.alias else []):
            rows.append({"ticker": a, "start_date": r.start_date,
                         "end_date": r.end_date})
    if not rows:
        g.note("no renames applied to this baseline")
        return g

    cov = rename_coverage(pd.DataFrame(rows), execution)
    if not cov:
        g.fail("aliases are recorded but none could be evaluated")
        return g

    weak, concurrent = [], []
    for old_sym, v in sorted(cov.items()):
        if v["coverage"] < MIN_RENAME_COVERAGE:
            weak.append(f"{old_sym}->{v['successor']} {v['coverage']:.0%}")
        if v["concurrent_sessions"]:
            concurrent.append(
                f"{old_sym}/{v['successor']} {v['concurrent_sessions']} sessions")
    # Concurrency is the only real falsifier here -- after the symbol changed,
    # both cannot trade -- and it needs the *predecessor* column, which
    # canonicalisation removes before the panel is downloaded. On the real
    # baseline that is 0 of 15, so reporting "15 checked" claimed a test that
    # never ran. Coverage and concurrency are now counted separately.
    testable = [k for k, v in cov.items() if v.get("concurrency_testable")]
    g.note(f"{len(cov)} rename(s) checked for successor coverage")
    g.note(f"{len(testable)} of {len(cov)} testable for concurrent trading "
           f"(needs the predecessor's own prices, which canonicalisation "
           f"removes from this panel)")
    if not testable:
        g.note("NOTE: coverage shows a successor prices the predecessor's "
               "membership; it cannot show they are the same issuer. The "
               "recorded note on each entry remains the warrant.")
    if weak:
        g.fail("successor does not price the predecessor's membership: "
               + ", ".join(weak))
    if concurrent:
        g.fail("both symbols traded after the effective date, so they are not "
               "one issuer: " + ", ".join(concurrent))
    return g


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def _hash_series(s: pd.Series) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(s.round(10), index=True).values.tobytes()
    ).hexdigest()[:16]


def gate_determinism(scored, execution, config, engine, label: str) -> Gate:
    """Same scored panel, same NAV. Twice, in one process, compared by hash.

    Scope, stated because the name overstates it: this covers the *backtest*
    given a fixed scored panel. It does not cover the pipeline that produced
    that panel, and the pipeline is **not** reproducible across runs.

    Two rebuilds from the same commit, the same pinned data window and the
    same seed produced execution panels that agree to 2e-6 relative -- float
    noise in the vendor's adjustment arithmetic, not revised data -- and
    cohort CAGRs of 18.21% and 22.40%. LightGBM splits flip on near-ties, the
    ranking changes, and a different fifteen names get held. Both runs passed
    every gate here.

    So a number from this pipeline carries run-to-run uncertainty of several
    points of CAGR, and quoting one to two decimals implies a precision that
    does not exist. Bit-level reproducibility would need replay from the
    hashed snapshot instead of a fresh download.
    """
    g = Gate(f"deterministic backtest, fixed scores ({label})")
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
    ap.add_argument("--report", type=Path, default=None,
                    help="Write the survivorship residual here. Verification "
                         "never writes into the baseline directory: reading an "
                         "artifact must not modify it.")
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

    integrity = verify_snapshot_hashes(d)
    stints = load_snapshot_stints(d)

    # The vendor-absent set decides the survivorship bar, so where it comes
    # from decides whether the verdict is reproducible. Prefer the copy
    # recorded with the baseline; fall back to the live cache only with a note
    # saying the result is no longer a property of these artifacts alone.
    recorded = d / "vendor_absent.json"
    if recorded.exists():
        absent = set(json.loads(recorded.read_text()))
        absent_source = "vendor_absent.json recorded with the baseline"
    else:
        absent = _vendor_absent(DEFAULT_CACHE)
        absent_source = (f"the LIVE cache at {DEFAULT_CACHE} — not reproducible; "
                         "rebuild to record it with the baseline")

    recycled = set(read_manifest_key(d, "recycled_symbols") or ())
    surv = gate_survivorship(execution, scored, stints, absent, absent_source,
                             recycled)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(getattr(surv, "residual", {}), indent=2))
        print(f"Survivorship residual -> {args.report}")
    gates: list[Gate] = [
        integrity,
        gate_output_hashes(d),
        gate_execution_derivation(d),
        gate_pit(scored, execution, args.horizon, stints),
        gate_renames(stints, execution),
        surv,
    ]

    # The benchmark comes from the snapshot. Turning it off -- which is what
    # this verifier used to do -- meant beta, alpha and the HAC t-statistic
    # were the one part of the published results that nothing checked, which is
    # unfortunate given they are the part the conclusions rest on.
    provider = None
    try:
        provider = SnapshotProvider(d)
        provider.download_benchmark("SPY", None, None)
    except (SnapshotIncomplete, FileNotFoundError, OSError) as exc:
        print(f"NOTE: no recorded benchmark ({exc}); relative metrics cannot "
              f"be verified. Record one with "
              f"scripts/record_baseline_benchmark.py {d}\n")
        provider = None

    measured = measure_engines(
        scored, execution, provider=provider, top_n=args.top_n,
        horizon=args.horizon, max_cohorts=args.max_cohorts,
        slippage_bps=args.slippage_bps, rebalance_day=args.rebalance_day,
        exit_rank=args.exit_rank,
    )

    bench_gate = Gate("relative metrics are measured against a recorded benchmark")
    if provider is None:
        bench_gate.fail("no benchmark in the snapshot, so beta, alpha and the "
                        "HAC t-statistic are unverifiable from these artifacts")
    else:
        n = len(measured["long-short"]["result"].bench_daily_nav)
        if n < 2:
            bench_gate.fail("the recorded benchmark produced no usable series")
        else:
            bench_gate.note(f"SPY, {n:,} sessions, from the snapshot")
    gates.append(bench_gate)
    gates.append(gate_expected_metrics(d, measured))

    for label in ("long-short", "cohort", "rank-hold"):
        m = measured[label]["metrics"]
        print(f"{label:10s} CAGR {m.get('cagr', float('nan')):7.2%}  "
              f"Sharpe {m.get('sharpe', float('nan')):5.2f}  "
              f"maxDD {m.get('max_drawdown', float('nan')):7.2%}  "
              f"beta {m.get('beta', float('nan')):+5.2f}  "
              f"alpha {m.get('alpha_ann', float('nan')):+7.2%} "
              f"(t {m.get('alpha_t', float('nan')):+5.2f})")
    print()

    # The long-short book has no cohort ledger to reconcile a *terminal*
    # identity against -- it is a continuously marked book, not a sequence of
    # closed baskets -- which is why its accounting used to be skipped
    # outright. The per-session identity specs.md:414 actually asks for does
    # apply, and is checked here.
    ls = measured["long-short"]["result"]
    g_ls = Gate("accounting reconciles (long-short)")
    ls_cash = getattr(ls, "daily_cash", None)
    ls_held = getattr(ls, "daily_positions", None)
    if ls_cash is None or ls_held is None:
        g_ls.fail("engine reports no per-session ledger")
    else:
        resid = (ls_cash + ls_held - ls.daily_nav).abs()
        scale = float(max(abs(float(ls.daily_nav.iloc[0])), 1.0))
        worst = float(resid.max()) if len(resid) else 0.0
        if worst > ACCOUNTING_TOLERANCE * scale:
            when = resid.idxmax()
            g_ls.fail(f"cash + holdings != NAV on {pd.Timestamp(when).date()} "
                      f"(worst residual {worst:.2e})")
        else:
            g_ls.note(f"cash + holdings = NAV on all {len(ls.daily_nav):,} "
                      f"sessions (worst residual {worst:.2e})")
        g_ls.note(f"short book carries cash above capital, as it must: "
                  f"peak cash {float(ls_cash.max()):,.0f} vs capital "
                  f"{ls.config.initial_capital:,.0f}")
    gates.append(g_ls)
    gates.append(gate_fills(ls, "long-short"))
    from stock_predictor.long_short import run_long_short_backtest

    ls_b = run_long_short_backtest(scored, measured["long-short"]["config"],
                                   provider=provider, execution_prices=execution)
    g = Gate("deterministic backtest, fixed scores (long-short)")
    ha, hb = _hash_series(ls.daily_nav), _hash_series(ls_b.daily_nav)
    g.note(f"NAV hash {ha} / {hb}")
    if ha != hb:
        g.fail("two runs over identical inputs produced different NAV series")
    if (ls.daily_nav <= 0).any():
        g.fail("long-short NAV touches zero or below")
    gates.append(g)

    for label, engine in (("cohort", run_backtest),
                          ("rank-hold", run_rank_hold_backtest)):
        res = measured[label]["result"]
        cfg = measured[label]["config"]
        gates.append(gate_accounting(res, cfg, label))
        gates.append(gate_fills(res, label))
        gates.append(gate_determinism(scored, execution, cfg, engine, label))
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

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("git_dirty"):
            print("WARNING: built from a dirty working tree; not reproducible "
                  "from the recorded commit alone.")


if __name__ == "__main__":
    main()
