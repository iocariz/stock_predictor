"""CLI entry for training (snapshots + run manifest)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from stock_predictor import repro
from stock_predictor.data_provider import get_provider
from stock_predictor.pit import (
    SP500_STINTS_URL,
    current_members,
    drop_recycled_prices,
    load_sp500_stints,
)
from stock_predictor.training import (
    LABEL_TARGETS,
    OPTUNA_METRICS,
    build_feature_panel,
    build_labeled_panel,
    evaluate_test_set,
    feature_importances,
    monthly_walk_forward,
    purge_train_dates,
    resolve_optuna_metric,
    run_optuna_search,
    save_eval_plots,
    save_model_artifacts,
    select_scoring_rows,
    select_training_rows,
    train_final_model,
    train_final_rank_model,
)
from stock_predictor.universe import (
    DEFAULT_MIN_COVERAGE,
    check_download_coverage,
    split_universes,
    universe_hash,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train S&P 500 LightGBM model (notebook pipeline).")
    p.add_argument("--start", default="2018-01-01", help="Price download start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Price download end (YYYY-MM-DD); default: today")
    p.add_argument("--train-end", default="2022-12-31", dest="train_end")
    p.add_argument("--test-start", default="2023-01-01", dest="test_start")
    p.add_argument(
        "--sample-n",
        type=int,
        default=500,
        dest="sample_n",
        help="Cap the universe at this many tickers, drawn as a seeded RANDOM "
        "sample (not an alphabetical prefix). Use a large value (e.g. 10000) "
        "for the full universe",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help="Symbols per yfinance request (default 100). Lower it if Yahoo "
        "throttles a large universe; no effect with --provider tiingo",
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        dest="min_coverage",
        help="Fail the run if the price download returns fewer than this "
        "fraction of the CURRENT index members. Departed members that the "
        "vendor no longer serves are reported as a survivorship gap, not "
        "gated (0 = never fail, warn only)",
    )
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--no-optuna", action="store_true", help="Skip Optuna hyperparameter search")
    p.add_argument("--optuna-trials", type=int, default=40, dest="optuna_trials")
    p.add_argument(
        "--optuna-metric",
        default="auto",
        choices=list(OPTUNA_METRICS),
        dest="optuna_metric",
        help="What Optuna maximizes. auto (default): topn_excess for the "
        "ranker, pr_auc for the classifier. topn_excess/topn_ir score the "
        "traded rule — mean per-date excess forward return of the top-N "
        "basket (N = --wf-top-k), the IR variant divided by its std. ndcg "
        "rewards ordering the whole list and measurably flattens the traded "
        "end, so it is no longer the ranker default",
    )
    p.add_argument("--ts-cv-splits", type=int, default=5, dest="ts_cv_splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-earnings", action="store_true", help="Omit Yahoo earnings feature (faster)")
    p.add_argument("--skip-walk-forward", action="store_true")
    p.add_argument("--wf-min-train-rows", type=int, default=5000, dest="wf_min_train_rows")
    p.add_argument("--wf-top-k", type=int, default=10, dest="wf_top_k")
    p.add_argument("--earnings-workers", type=int, default=8, dest="earnings_workers")
    p.add_argument("--plots-dir", type=Path, default=None, help="Save PNGs here; omit to skip plots")
    p.add_argument("--output-model", type=Path, default=None, help="Pickle LGBMClassifier + metadata")
    p.add_argument("--run-backtest", action="store_true", help="Run portfolio backtest after walk-forward")
    p.add_argument(
        "--wf-scores-path",
        type=Path,
        default=None,
        dest="wf_scores_path",
        help="Save walk-forward scored panel to this parquet path",
    )
    p.add_argument(
        "--train-through-latest",
        action="store_true",
        dest="train_through_latest",
        help="Ignore --train-end and fit through the newest date a label can "
        "exist for (HORIZON sessions behind the last session). For a "
        "production refit: the walk-forward in such a run measures nothing, "
        "so evaluate with a held-back window separately",
    )
    p.add_argument(
        "--execution-prices-path",
        type=Path,
        default=None,
        dest="execution_prices_path",
        help="Write the unfiltered adj_close panel (dates x tickers) from THIS "
        "run's download, for backtest-sp500 --execution-prices. Nothing used to "
        "produce this file, so the backtest silently paired whatever stale "
        "parquet happened to be on disk, or fell back to forward-filled prices",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Write hashed parquet snapshots + manifest.json here (default: artifacts/runs/<run_id>/)",
    )
    p.add_argument(
        "--replay-snapshot",
        type=Path,
        default=None,
        dest="replay_snapshot",
        help="Rebuild from a previous run's snapshot instead of downloading. "
             "Hashes are verified first. Feature engineering, labelling and "
             "training all re-run; only the inputs are replaced, so a code "
             "change still shows.",
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable reproducibility snapshots (no parquet dumps, minimal manifest)",
    )
    p.add_argument(
        "--provider",
        default="yfinance",
        choices=["yfinance", "tiingo", "hybrid"],
        help="Data provider: yfinance (default) or tiingo (Tiingo equities + FRED macro)",
    )
    p.add_argument(
        "--no-macro-merge",
        action="store_true",
        help="Disable Yahoo↔FRED macro cross-fill (use single macro source only)",
    )
    p.add_argument(
        "--objective",
        default="rank",
        choices=["rank", "binary"],
        help="rank (default): LGBMRanker (lambdarank, grouped by date) on "
        "per-date forward-return quintile grades. binary: LGBMClassifier on "
        "fwd_ret >= --threshold. The binary label is satisfied mechanically by "
        "volatility — a name needs a wide distribution to clear +5%% in ten "
        "sessions — so it trains a volatility ranker (score-vs-vol_21d "
        "cross-sectional IC +0.75 vs +0.43 for rank)",
    )
    p.add_argument(
        "--rank-objective",
        action="store_true",
        dest="rank_objective",
        help="Deprecated: rank is now the default. Accepted for compatibility; "
        "conflicts with --objective binary",
    )
    p.add_argument(
        "--label-target",
        default="raw",
        choices=list(LABEL_TARGETS),
        dest="label_target",
        help="With --rank-objective, what the ranker is asked to rank: raw "
        "forward return, vol_adj (per unit trailing volatility), excess "
        "(minus the date's cross-sectional median), or excess_vol_adj. The "
        "default +5%%-in-10-days label is satisfied mechanically by "
        "volatility, so a model trained on it ranks risk, not return",
    )
    p.add_argument(
        "--fundamentals",
        action="store_true",
        help="Add point-in-time fundamental features from SEC EDGAR XBRL "
        "(free, no key). Joined on each report's filing date, never its "
        "fiscal period end. First run downloads ~3.8 MB per ticker; set "
        "SEC_USER_AGENT to a descriptive string with contact details, or SEC "
        "answers 403",
    )
    p.add_argument(
        "--edgar-cache",
        type=Path,
        default=Path("artifacts/edgar_cache"),
        dest="edgar_cache",
        help="Directory for cached EDGAR company facts (one parquet per ticker)",
    )
    p.add_argument(
        "--strict-dropna",
        action="store_true",
        dest="strict_dropna",
        help="Drop rows with ANY NaN feature (legacy behavior); default keeps "
        "them since LightGBM handles missing values natively",
    )
    return p.parse_args()


BENCHMARK_TICKER = "SPY"
"""What every relative metric in this project is measured against. Recorded
with the run so beta, alpha and the HAC t-statistic can be re-derived from the
artifacts instead of from a fresh download."""


def _download_benchmark_series(provider, sessions) -> pd.DataFrame | None:
    """The benchmark over the panel's window, long-form, or ``None``.

    Never fatal: a run whose benchmark fetch fails still produces a usable
    baseline, it just cannot have its relative metrics verified offline. The
    verifier says so rather than assuming.
    """
    if provider is None or not len(sessions):
        return None
    try:
        series = provider.download_benchmark(
            BENCHMARK_TICKER,
            str(pd.Timestamp(sessions.min()).date()),
            str(pd.Timestamp(sessions.max()).date()),
        )
    except Exception as exc:  # noqa: BLE001 - vendor failures are not fatal here
        print(f"  Note: benchmark not recorded ({type(exc).__name__}: {exc})")
        return None
    if series is None or not len(series):
        print("  Note: benchmark not recorded (vendor returned nothing)")
        return None
    s = pd.Series(series).dropna()
    return pd.DataFrame({
        "date": pd.DatetimeIndex(s.index),
        "ticker": BENCHMARK_TICKER,
        "close": s.to_numpy(dtype=float),
    })


def resolve_objective(args: argparse.Namespace) -> str:
    """Reconcile --objective with the legacy --rank-objective flag."""
    if args.rank_objective and args.objective == "binary":
        sys.exit(
            "Error: --rank-objective conflicts with --objective binary. "
            "--rank-objective is deprecated (rank is the default); drop it."
        )
    return args.objective


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
"""Hand-picked baseline, used as-is when --no-optuna and as the base otherwise."""


def validate_train_test_window(train_end: str, test_start: str) -> None:
    """Training must end before testing begins, or the test period leaks."""
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        raise ValueError(
            f"--train-end ({train_end}) must be before --test-start ({test_start})."
        )


def resolve_model_params(optuna_best: dict) -> dict:
    """Baseline hyperparameters with any tuned values layered on top.

    Returns a fresh dict: handing out the module-level default by reference
    would let one run's tuning leak into the next.
    """
    params = dict(DEFAULT_LGBM_PARAMS)
    params.update(optuna_best or {})
    return params


def build_run_extra(
    args: argparse.Namespace, *, objective: str, tune_metric: str,
) -> dict[str, Any]:
    """The knobs a manifest must record to make a run reproducible.

    Resolved values, not raw flags: ``--objective auto`` recorded as "auto"
    stops identifying the run the moment the default changes.
    """
    return {
        "start": args.start,
        "end": args.end,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "sample_n": args.sample_n,
        "min_coverage": args.min_coverage,
        "horizon": args.horizon,
        "threshold": args.threshold,
        "skip_earnings": args.skip_earnings,
        "no_optuna": args.no_optuna,
        "no_macro_merge": args.no_macro_merge,
        "strict_dropna": args.strict_dropna,
        "objective": objective,
        "label_target": args.label_target,
        "optuna_metric": tune_metric,
        "fundamentals": bool(args.fundamentals),
        "seed": args.seed,
    }


def split_train_test(
    features_clean: pd.DataFrame,
    *,
    train_end: str,
    test_start: str | None,
    horizon: int,
    refit: bool,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split labelled rows for evaluation, or hand all of them to a refit.

    Two modes, because they are two different jobs:

    **Evaluation** holds out everything from *test_start* and purges the
    horizon before it out of the training set. Labels look ``horizon`` sessions
    ahead, so a training row inside that window saw test-period prices.

    **Refit** is a production build. ``train_end`` is already the newest
    session with a complete forward window, so there is no held-out period --
    and purging against a test set that does not exist deleted the horizon of
    rows nearest the present, the most relevant data the model has. With 500
    sessions at horizon 63 that silently cost 63 sessions and left zero test
    rows for evaluation to choke on.

    Returns ``(train, None)`` for a refit: ``None`` rather than an empty frame,
    so a caller that forgets to branch fails loudly instead of passing a
    zero-row frame into something that cannot score it.
    """
    labelled = features_clean[features_clean[date_col] <= pd.Timestamp(train_end)]

    if refit:
        return labelled, None

    if test_start is None:
        raise ValueError("evaluation mode needs a test_start")
    train = purge_train_dates(labelled, test_start, horizon)
    test = features_clean[features_clean[date_col] >= pd.Timestamp(test_start)]
    if not len(test):
        raise ValueError(
            f"evaluation split produced no test rows at or after {test_start} "
            f"(panel ends {features_clean[date_col].max().date() if len(features_clean) else 'n/a'}). "
            "For a production build pass --train-through-latest, which trains "
            "on every labelable row and skips evaluation."
        )
    return train, test


def latest_trainable_end(sessions, horizon: int):
    """Newest session that can still carry a label.

    A label looks *horizon* sessions ahead, so the most recent trainable date
    is that far behind the last session available. A scheduled refit that
    hard-codes a date instead learns nothing new each month.
    """
    idx = pd.DatetimeIndex(sessions).sort_values()
    if len(idx) <= horizon:
        raise ValueError(
            f"not enough history: {len(idx)} sessions for a {horizon}-session "
            "label horizon"
        )
    return pd.Timestamp(idx[-(horizon + 1)])


def build_model_meta(
    args: argparse.Namespace,
    *,
    feature_cols: list[str],
    objective: str,
    tune_metric: str,
    optuna_best: dict,
    manual_params: dict,
    n_trees: int,
    universe: list[str] | None = None,
    fitted_through=None,
    train_end: str,
    test_start: str | None,
    importance: dict,
    pr_auc: float,
    roc_auc: float,
    run_id: str,
    snapshot_root: Path | None,
) -> dict[str, Any]:
    """What a saved model remembers about how it was built.

    ``feature_cols`` and ``horizon`` are contractual —
    :func:`~stock_predictor.predict.load_model` refuses metadata without them.
    ``seed`` and ``sample_n`` are what the live path uses to rebuild the same
    universe the model was fitted on.
    """
    return {
        "feature_cols": feature_cols,
        # What the model actually learned through. Purging removes `horizon`
        # sessions before test_start, so this trails train_end by a quarter at
        # horizon 63 -- and the freshness gate was reading the optimistic date.
        "fitted_through": (
            str(pd.Timestamp(fitted_through).date())
            if fitted_through is not None else None
        ),
        "objective": objective,
        "label_target": args.label_target,
        "optuna_metric": tune_metric,
        "horizon": args.horizon,
        "threshold": args.threshold,
        "start": args.start,
        "end": args.end,
        # The *effective* window, not the flags. A refit overrides train_end
        # to the newest labelable session and has no test period; reading args
        # here recorded the hard-coded defaults instead, so a refit's metadata
        # claimed a train_end months before the one it actually used.
        "train_end": train_end,
        "test_start": test_start,
        "sample_n": args.sample_n,
        "seed": args.seed,
        # The tickers actually drawn, not the recipe for drawing them. Training
        # samples the union over the whole window; a live run resampling a
        # 400-day window with the same seed draws from a different population
        # and gets a different set. Recording the draw removes the dependence.
        "universe": sorted(universe) if universe else None,
        "universe_hash": universe_hash(universe) if universe else None,
        "skip_earnings": args.skip_earnings,
        "optuna_best": optuna_best,
        "manual_params": manual_params,
        "best_iteration": n_trees,
        "feature_importance": importance,
        "metrics": {"pr_auc": pr_auc, "roc_auc": roc_auc},
        "run_id": run_id,
        "snapshot_dir": str(snapshot_root) if snapshot_root else None,
    }


def main() -> None:
    args = parse_args()
    objective = resolve_objective(args)
    tune_metric = resolve_optuna_metric(args.optuna_metric, objective)
    replay = getattr(args, "replay_snapshot", None)
    replay_sector_map = replay_macro = None
    if replay is not None:
        from stock_predictor import replay as _replay

        hashes = _replay.verify(replay)
        print(f"Replaying {replay} ({len(hashes)} artifacts verified)")
        gaps = _replay.missing_for_exact_replay(replay)
        if gaps:
            print(f"  WARNING: snapshot lacks {', '.join(gaps)}; this replay "
                  "cannot be exact. Rebuild to record them.")
        provider = _replay.SnapshotProvider(replay)
        replay_sector_map = _replay.load_sector_map(replay)
        replay_macro = _replay.load_macro(replay)
    else:
        provider = get_provider(args.provider, batch_size=args.batch_size)
    start, end = args.start, args.end
    train_end, test_start = args.train_end, args.test_start
    horizon, threshold = args.horizon, args.threshold
    try:
        validate_train_test_window(train_end, test_start)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    run_id = repro.new_run_id()
    snapshot_root: Path | None = None
    manifest: dict[str, Any] | None = None
    if not args.no_snapshot:
        snapshot_root = args.snapshot_dir or (repro.REPO_ROOT / "artifacts" / "runs" / run_id)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        manifest = repro.build_base_manifest(
            run_id=run_id,
            argv=sys.argv.copy(),
            extra=build_run_extra(args, objective=objective, tune_metric=tune_metric),
        )
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    print("Loading PIT stints & ticker universe…")
    if replay is not None:
        from stock_predictor import replay as _replay

        # Recorded membership, not today's. The rename work rewrote 12 stints;
        # replaying against the live table would reproduce a different run.
        stints = _replay.load_stints(replay)
    else:
        stints = load_sp500_stints(SP500_STINTS_URL)
    # Three universes, three purposes. Sampling the download-window population
    # let names admitted to the index AFTER training compete for slots with the
    # historical ones, so future membership decided which past companies the
    # model saw -- at --sample-n 500, 352 of the historical names differed.
    split = split_universes(
        stints, start=start, train_end=train_end, end=end,
        sample_n=args.sample_n, seed=args.seed,
    )
    tickers = split.download
    sample = split.download
    fitted_universe = split.fitted
    print(f"  Union tickers overlapping window: {len(tickers)}")
    if split.joined_after_training:
        print(f"  {len(split.joined_after_training)} ticker(s) joined after "
              f"{train_end}; downloaded for recent cross-sections and pricing, "
              "but not part of the fitted universe")
    if len(sample) < len(tickers):
        print(
            f"  Downloading {len(sample)} tickers "
            f"(seeded random sample, seed={args.seed})…"
        )
    else:
        print(f"  Downloading all {len(sample)} tickers…")

    if manifest is not None and snapshot_root is not None:
        meta = repro.snapshot_parquet(stints, snapshot_root / "stints.parquet")
        repro.register_snapshot(manifest, "stints", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    adj_close, volume = provider.download_equity_ohlcv(sample, start, end)
    print(f"  adj_close shape: {adj_close.shape}")

    # A ticker is not a company. Once a member is acquired or renamed its
    # symbol is retired and the exchange may reassign it, so the panel picks
    # up a different issuer's prices under a departed member's name -- Qwest's
    # Q priced from 2025, Anadarko's APC from 2026. 54 of 347 departed names
    # were like this. The PIT filter keeps them out of the scored panel, but
    # they stay in the execution panel where a deferred exit walking forward
    # for the next real quote can reach them.
    adj_close, volume, recycled = drop_recycled_prices(
        adj_close, stints, volume=volume,
    )
    if recycled:
        print(f"  Dropped reused-symbol price blocks for {len(recycled)} "
              f"departed ticker(s): {', '.join(recycled[:8])}"
              + (" …" if len(recycled) > 8 else ""))
    # Recorded, not dropped invisibly. Removing the contaminated prices also
    # removes the evidence that the symbol was reused, so without this the
    # survivorship gate cannot tell "another issuer holds this symbol now"
    # from "nobody ever fetched it" -- and would demand a refetch that returns
    # the same wrong company.
    if manifest is not None:
        manifest["recycled_symbols"] = recycled
        if snapshot_root is not None:
            repro.write_manifest(snapshot_root / "manifest.json", manifest)
    coverage = check_download_coverage(
        sample, adj_close,
        min_coverage=args.min_coverage,
        active=current_members(stints),
        label="equity download",
    )
    if manifest is not None:
        manifest["universe"] = {
            "requested": len(sample),
            "union_overlapping_window": len(tickers),
            "coverage": coverage,
        }

    if manifest is not None and snapshot_root is not None:
        long_px = repro.wide_prices_to_long(adj_close, volume)
        meta = repro.snapshot_parquet(long_px, snapshot_root / "equity_prices_long.parquet")
        repro.register_snapshot(manifest, "equity_prices_long", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    # PIT filtering is deferred to build_feature_panel so per-ticker rolling
    # features see each symbol's full contiguous history first.
    fundamentals = None
    if args.fundamentals:
        from stock_predictor.fundamentals import fetch_fundamentals

        print("Fetching SEC EDGAR fundamentals…")
        fundamentals = fetch_fundamentals(sample, cache_dir=args.edgar_cache)
        n_tk = fundamentals["ticker"].nunique() if len(fundamentals) else 0
        print(f"  {len(fundamentals):,} facts for {n_tk} tickers")

    # The execution panel is this run's own download, before any PIT filter:
    # what a holding was actually worth, including after it left the index.
    if args.execution_prices_path is not None:
        args.execution_prices_path.parent.mkdir(parents=True, exist_ok=True)
        adj_close.to_parquet(args.execution_prices_path)
        print(f"  Saved execution prices to {args.execution_prices_path} "
              f"({adj_close.shape[0]} dates x {adj_close.shape[1]} tickers)")
        if manifest is not None and snapshot_root is not None:
            meta = repro.snapshot_parquet(
                repro.wide_prices_to_long(adj_close, volume),
                snapshot_root / "execution_prices.parquet",
            )
            repro.register_snapshot(manifest, "execution_prices", meta)
            # The wide panel at the baseline root is what every backtest and
            # the verifier actually read, so it needs its own hash. The long
            # snapshot above is the input it was pivoted from.
            repro.register_output(manifest, "execution_prices",
                                  args.execution_prices_path)
            repro.write_manifest(snapshot_root / "manifest.json", manifest)

    refit = bool(args.train_through_latest)
    if refit:
        # A scheduled refit that hard-codes train_end learns nothing new.
        # There is no held-out period here: train_end is already the newest
        # session with a complete forward window. test_start stays None so
        # nothing downstream invents one.
        train_end = str(latest_trainable_end(adj_close.index, horizon).date())
        test_start = None
        print(f"  Refit mode: training through {train_end} "
              f"(newest labellable session); evaluation and walk-forward "
              f"are skipped -- there is nothing held out to measure on.")

    labeled = build_labeled_panel(adj_close, None, horizon, threshold)
    print(f"  Positive rate: {labeled['target_5pct'].mean():.4%}")

    if manifest is not None and snapshot_root is not None:
        # Pre-PIT: the filter now runs inside build_feature_panel, so this
        # snapshot is the labeled panel over the full downloaded history.
        meta = repro.snapshot_parquet(labeled, snapshot_root / "labeled.parquet")
        repro.register_snapshot(manifest, "labeled", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    features, feature_cols = build_feature_panel(
        labeled,
        volume,
        start=start,
        end=end,
        skip_earnings=args.skip_earnings,
        earnings_workers=args.earnings_workers,
        provider=provider,
        provider_name=args.provider,
        macro_merge=not args.no_macro_merge,
        stints=stints,
        fundamentals=fundamentals,
        sector_map=replay_sector_map,
        macro_raw=replay_macro,
        sources=(sources := {}),
    )

    # The sector map and the macro series were downloaded mid-build and never
    # recorded, so two of a run's five external inputs were whatever the
    # network returned that day. Recording them is what makes replay exact.
    if manifest is not None and snapshot_root is not None:
        for name in ("sector_map", "macro"):
            frame = sources.get("sector_map" if name == "sector_map" else "macro_raw")
            if frame is None or not len(frame):
                continue
            meta = repro.snapshot_parquet(frame, snapshot_root / f"{name}.parquet")
            repro.register_snapshot(manifest, name, meta)

        # The benchmark is the fifth external input, and was the last one still
        # fetched at report time: every published beta, alpha and HAC t was
        # measured against whatever SPY the network served that day, so none of
        # them could be checked against the artifacts afterwards.
        bench = _download_benchmark_series(provider, adj_close.index)
        if bench is not None:
            meta = repro.snapshot_parquet(bench, snapshot_root / "benchmark.parquet")
            repro.register_snapshot(manifest, "benchmark", meta)
            print(f"  Recorded benchmark {BENCHMARK_TICKER}: {len(bench):,} sessions")

        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    features_clean = select_training_rows(
        features, feature_cols, "target_5pct", strict=args.strict_dropna,
    )
    # The walk-forward trains on labelled rows but must *score* every row a
    # live run would score, including the newest sessions whose forward return
    # is not knowable yet. Handing it features_clean deleted exactly those.
    features_scorable = select_scoring_rows(
        features, feature_cols, "target_5pct", strict=args.strict_dropna,
    )
    train, test = split_train_test(
        features_clean, train_end=train_end, test_start=test_start,
        horizon=horizon, refit=refit,
    )
    print(f"Train {train[feature_cols].shape} | pos {train['target_5pct'].mean():.4%}")
    if test is None:
        print("Test  (none - production refit)")
    else:
        print(f"Test  {test[feature_cols].shape} | pos {test['target_5pct'].mean():.4%}")

    if manifest is not None and snapshot_root is not None:
        # Feature panel can be large; snapshot cleaned training+test panel only
        meta = repro.snapshot_parquet(features_clean, snapshot_root / "features_clean.parquet")
        repro.register_snapshot(manifest, "features_clean", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    optuna_best: dict = {}
    if not args.no_optuna:
        optuna_best = run_optuna_search(
            train,
            feature_cols,
            ts_cv_splits=args.ts_cv_splits,
            n_trials=args.optuna_trials,
            seed=args.seed,
            purge_days=horizon,
            objective=objective,
            label_target=args.label_target,
            optuna_metric=tune_metric,
            rank_eval_k=args.wf_top_k,
        )

    manual_params = resolve_model_params(optuna_best)

    if objective == "rank":
        model, n_trees = train_final_rank_model(
            train, feature_cols, manual_params, args.seed, purge_days=horizon,
            label_target=args.label_target, metric=tune_metric,
            eval_k=args.wf_top_k,
        )
    else:
        model, n_trees = train_final_model(
            train, feature_cols, manual_params, args.seed, purge_days=horizon,
            metric=tune_metric, eval_k=args.wf_top_k,
        )
    # A refit has no held-out period, so there is nothing to evaluate on. The
    # empty frame used to be passed in regardless, and evaluation cannot score
    # zero rows -- which is how every scheduled run would have failed after
    # paying for the fit.
    pr_auc = roc_auc = float("nan")
    if test is not None:
        pr_auc, roc_auc, daily_precision = evaluate_test_set(model, test, feature_cols)

        if args.plots_dir is not None:
            save_eval_plots(
                args.plots_dir,
                daily_precision,
                test["target_5pct"],
                feature_cols,
                model,
            )

    wf_scores: pd.DataFrame | None = None
    # Walk-forward measures out-of-sample months after test_start. In refit
    # mode there are none, and it would silently return an empty frame -- the
    # quieter half of the same defect.
    if refit and not args.skip_walk_forward:
        print("  Refit mode: skipping walk-forward (no out-of-sample period). "
              "Use `evaluate` for measurement and `refit` for the artifact.")
    if not args.skip_walk_forward and not refit:
        need_scores = args.run_backtest or args.wf_scores_path is not None
        wf_out = monthly_walk_forward(
            features_scorable,
            feature_cols,
            "target_5pct",
            "date",
            test_start,
            manual_params,
            inner_val_frac=0.05,
            min_train_rows=args.wf_min_train_rows,
            top_k=args.wf_top_k,
            random_state=args.seed,
            return_scores=need_scores,
            purge_days=horizon,
            objective=objective,
            label_target=args.label_target,
            metric=tune_metric,
        )
        if need_scores:
            wf_results, wf_scores = wf_out
        else:
            wf_results = wf_out
        print(wf_results.to_string(index=False))
        if len(wf_results):
            # Per-date first: it is the unit the strategy trades in. The
            # pooled AUC follows as context, explicitly labelled.
            print(
                f"Walk-forward: per-date P@{args.wf_top_k} "
                f"{wf_results['precision_at_k'].mean():.4f} | "
                f"rank IC {wf_results['rank_ic'].mean():+.4f} | "
                f"top-{args.wf_top_k} excess "
                f"{wf_results['top_k_excess'].mean():+.4%} | "
                f"pooled PR-AUC {wf_results['pr_auc_pooled'].mean():.4f}"
            )
        if args.plots_dir is not None and len(wf_results):
            fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
            wf_results.set_index("month")["rank_ic"].astype(float).plot(
                ax=axes[0], marker="o", title="Walk-forward per-date rank IC",
            )
            wf_results.set_index("month")["precision_at_k"].astype(float).plot(
                ax=axes[1],
                marker="o",
                color="darkgreen",
                title=f"Per-date Precision@{args.wf_top_k}",
            )
            plt.tight_layout()
            fig.savefig(args.plots_dir / "walk_forward.png", dpi=120)
            plt.close(fig)
        if wf_scores is not None and args.wf_scores_path is not None:
            args.wf_scores_path.parent.mkdir(parents=True, exist_ok=True)
            wf_scores.to_parquet(args.wf_scores_path, index=False)
            print(f"Saved walk-forward scores to {args.wf_scores_path}")
            # Nothing else in the baseline can re-derive a model's scores, so
            # this hash is the only thing standing between the verifier and a
            # forged score file.
            if manifest is not None:
                repro.register_output(manifest, "wf_scored", args.wf_scores_path)

    if args.run_backtest:
        if wf_scores is None or len(wf_scores) == 0:
            print(
                "No walk-forward scores available for backtest. "
                + ("A refit has no out-of-sample period to score; use "
                   "`evaluate` (REFIT=0) to measure."
                   if refit else
                   "Run without --skip-walk-forward.")
            )
        else:
            from stock_predictor.backtest import (
                BacktestConfig,
                plot_backtest,
                print_report,
                run_backtest,
            )

            # Hardcoded defaults, deliberately: train-sp500 has no strategy
            # flags, so this is a sanity check on the scores rather than a
            # measurement of whatever you intend to trade. Say so, or the
            # numbers below get quoted as if they were the strategy's.
            bt_config = BacktestConfig()
            print(
                f"  (sanity backtest at defaults: top_n={bt_config.top_n}, "
                f"holding_days={bt_config.holding_days}, "
                f"weighting={bt_config.weighting}, "
                f"rebalance_day={bt_config.rebalance_day}. For the configuration "
                "you trade, run backtest-sp500 via scripts/run_pipeline.sh, "
                "which shares its flags with predict-sp500.)"
            )
            bt_result = run_backtest(
                wf_scores, bt_config, provider=provider,
                execution_prices=adj_close,
            )
            print_report(bt_result)
            if args.plots_dir is not None:
                plot_backtest(bt_result, args.plots_dir)

    if args.output_model is not None:
        meta = build_model_meta(
            args,
            feature_cols=feature_cols,
            objective=objective,
            tune_metric=tune_metric,
            optuna_best=optuna_best,
            manual_params=manual_params,
            n_trees=n_trees,
            universe=fitted_universe,
            fitted_through=train["date"].max() if len(train) else None,
            train_end=train_end,
            test_start=test_start,
            importance=feature_importances(model, feature_cols),
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            run_id=run_id,
            snapshot_root=snapshot_root,
        )
        save_model_artifacts(args.output_model, model, meta, optuna_best)
        if manifest is not None and snapshot_root is not None:
            p = Path(args.output_model)
            manifest["model_artifact"] = {
                "path": str(p.resolve()),
                "sha256": repro.sha256_file(p),
            }
            meta_path = p.with_suffix(".meta.json")
            manifest["meta_json"] = {
                "path": str(meta_path.resolve()),
                "sha256": repro.sha256_file(meta_path) if meta_path.exists() else None,
            }

    if manifest is not None:
        manifest["status"] = "completed"
        manifest["results"] = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "features_clean_rows": int(len(features_clean)),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "feature_columns": feature_cols,
        }
        root = snapshot_root or (repro.REPO_ROOT / "artifacts" / "runs" / run_id)
        root.mkdir(parents=True, exist_ok=True)
        repro.write_manifest(root / "manifest.json", manifest)
        print(f"Run manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
