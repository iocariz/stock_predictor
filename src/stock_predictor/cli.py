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

from stock_predictor.data_provider import get_provider
from stock_predictor.pit import SP500_STINTS_URL, load_sp500_stints, tickers_overlapping_window
from stock_predictor import repro
from stock_predictor.training import (
    build_feature_panel,
    build_labeled_panel,
    evaluate_test_set,
    monthly_walk_forward,
    run_optuna_search,
    save_eval_plots,
    save_model_artifacts,
    train_final_model,
    wide_field,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train S&P 500 LightGBM model (notebook pipeline).")
    p.add_argument("--start", default="2018-01-01", help="Price download start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Price download end (YYYY-MM-DD); default: today")
    p.add_argument("--train-end", default="2022-12-31", dest="train_end")
    p.add_argument("--test-start", default="2023-01-01", dest="test_start")
    p.add_argument("--sample-n", type=int, default=500, dest="sample_n")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--no-optuna", action="store_true", help="Skip Optuna hyperparameter search")
    p.add_argument("--optuna-trials", type=int, default=40, dest="optuna_trials")
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
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Write hashed parquet snapshots + manifest.json here (default: artifacts/runs/<run_id>/)",
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable reproducibility snapshots (no parquet dumps, minimal manifest)",
    )
    p.add_argument(
        "--provider",
        default="yfinance",
        choices=["yfinance", "tiingo"],
        help="Data provider: yfinance (default) or tiingo (Tiingo equities + FRED macro)",
    )
    p.add_argument(
        "--no-macro-merge",
        action="store_true",
        help="Disable Yahoo↔FRED macro cross-fill (use single macro source only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)
    start, end = args.start, args.end
    train_end, test_start = args.train_end, args.test_start
    horizon, threshold = args.horizon, args.threshold
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        sys.exit(f"Error: --train-end ({train_end}) must be before --test-start ({test_start}).")

    run_id = repro.new_run_id()
    snapshot_root: Path | None = None
    manifest: dict[str, Any] | None = None
    if not args.no_snapshot:
        snapshot_root = args.snapshot_dir or (repro.REPO_ROOT / "artifacts" / "runs" / run_id)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        manifest = repro.build_base_manifest(
            run_id=run_id,
            argv=sys.argv.copy(),
            extra={
                "start": start,
                "end": end,
                "train_end": train_end,
                "test_start": test_start,
                "sample_n": args.sample_n,
                "horizon": horizon,
                "threshold": threshold,
                "skip_earnings": args.skip_earnings,
                "no_optuna": args.no_optuna,
                "no_macro_merge": args.no_macro_merge,
                "seed": args.seed,
            },
        )
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    print("Loading PIT stints & ticker universe…")
    stints = load_sp500_stints(SP500_STINTS_URL)
    tickers = tickers_overlapping_window(stints, start, end)
    print(f"  Union tickers overlapping window: {len(tickers)}")
    sample = tickers[: args.sample_n]
    print(f"  Downloading {len(sample)} tickers (sample-n cap)…")

    if manifest is not None and snapshot_root is not None:
        meta = repro.snapshot_parquet(stints, snapshot_root / "stints.parquet")
        repro.register_snapshot(manifest, "stints", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    adj_close, volume = provider.download_equity_ohlcv(sample, start, end)
    print(f"  adj_close shape: {adj_close.shape}")

    if manifest is not None and snapshot_root is not None:
        long_px = repro.wide_prices_to_long(adj_close, volume)
        meta = repro.snapshot_parquet(long_px, snapshot_root / "equity_prices_long.parquet")
        repro.register_snapshot(manifest, "equity_prices_long", meta)
        repro.write_manifest(snapshot_root / "manifest.json", manifest)

    labeled = build_labeled_panel(adj_close, stints, horizon, threshold)
    print(f"  Positive rate: {labeled['target_5pct'].mean():.4%}")

    if manifest is not None and snapshot_root is not None:
        meta = repro.snapshot_parquet(labeled, snapshot_root / "labeled_pit.parquet")
        repro.register_snapshot(manifest, "labeled_pit", meta)
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
    )

    features_clean = features.dropna(subset=feature_cols + ["target_5pct"])
    train = features_clean[features_clean["date"] <= train_end]
    test = features_clean[features_clean["date"] >= test_start]
    print(f"Train {train[feature_cols].shape} | pos {train['target_5pct'].mean():.4%}")
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
        )

    manual_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if optuna_best:
        manual_params.update(optuna_best)

    model, n_trees = train_final_model(train, feature_cols, manual_params, args.seed)
    pr_auc, roc_auc, weekly_precision = evaluate_test_set(model, test, feature_cols)

    if args.plots_dir is not None:
        save_eval_plots(
            args.plots_dir,
            weekly_precision,
            test["target_5pct"],
            feature_cols,
            model,
        )

    wf_scores: pd.DataFrame | None = None
    if not args.skip_walk_forward:
        need_scores = args.run_backtest or args.wf_scores_path is not None
        wf_out = monthly_walk_forward(
            features_clean,
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
        )
        if need_scores:
            wf_results, wf_scores = wf_out
        else:
            wf_results = wf_out
        print(wf_results.to_string(index=False))
        if len(wf_results):
            print(
                f"Walk-forward: mean PR-AUC {wf_results['pr_auc'].mean():.4f} | "
                f"mean weekly P@{args.wf_top_k} "
                f"{wf_results['mean_weekly_precision_at_k'].mean():.4f}"
            )
        if args.plots_dir is not None and len(wf_results):
            fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
            wf_results.set_index("month")["pr_auc"].astype(float).plot(
                ax=axes[0], marker="o", title="Walk-forward PR-AUC",
            )
            wf_results.set_index("month")["mean_weekly_precision_at_k"].astype(float).plot(
                ax=axes[1],
                marker="o",
                color="darkgreen",
                title=f"Weekly Precision@{args.wf_top_k}",
            )
            plt.tight_layout()
            fig.savefig(args.plots_dir / "walk_forward.png", dpi=120)
            plt.close(fig)
        if wf_scores is not None and args.wf_scores_path is not None:
            args.wf_scores_path.parent.mkdir(parents=True, exist_ok=True)
            wf_scores.to_parquet(args.wf_scores_path, index=False)
            print(f"Saved walk-forward scores to {args.wf_scores_path}")

    if args.run_backtest:
        if wf_scores is None or len(wf_scores) == 0:
            print(
                "No walk-forward scores available for backtest. Run without --skip-walk-forward."
            )
        else:
            from stock_predictor.backtest import BacktestConfig, plot_backtest, print_report, run_backtest

            bt_config = BacktestConfig()
            bt_result = run_backtest(wf_scores, bt_config, provider=provider)
            print_report(bt_result)
            if args.plots_dir is not None:
                plot_backtest(bt_result, args.plots_dir)

    if args.output_model is not None:
        meta = {
            "feature_cols": feature_cols,
            "horizon": horizon,
            "threshold": threshold,
            "start": start,
            "end": end,
            "train_end": train_end,
            "test_start": test_start,
            "sample_n": args.sample_n,
            "skip_earnings": args.skip_earnings,
            "optuna_best": optuna_best,
            "manual_params": manual_params,
            "best_iteration": n_trees,
            "metrics": {"pr_auc": pr_auc, "roc_auc": roc_auc},
            "run_id": run_id,
            "snapshot_dir": str(snapshot_root) if snapshot_root else None,
        }
        save_model_artifacts(args.output_model, model, meta, optuna_best)
        if manifest is not None and snapshot_root is not None:
            p = Path(args.output_model)
            manifest["model_artifact"] = {
                "path": str(p.resolve()),
                "sha256": repro.sha256_file(p),
            }
            manifest["meta_json"] = {
                "path": str(p.with_suffix(".meta.json").resolve()),
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
