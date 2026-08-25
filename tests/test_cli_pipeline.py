"""``train-sp500`` end to end, offline.

Every network edge is stubbed and the universe is synthetic, so this runs in
seconds — but it exercises the real ``cli.main``: universe sampling, feature
staging, the PIT filter, training, evaluation, the walk-forward, and model
persistence.

This is the test shape that was missing. A wiring bug — ``main`` handing the
walk-forward the *training* row selection instead of the *scoring* one —
survived a full regeneration cycle and 390 passing unit tests, because every
test called ``monthly_walk_forward`` directly and nothing crossed the seam
where the mistake lived.
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor import cli

N_TICKERS = 40
DATES = pd.bdate_range("2015-01-01", periods=900)
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]


class _FakeProvider:
    """Deterministic prices with a mild cross-sectional drift to learn."""

    def download_equity_ohlcv(self, tickers, start, end):
        rng = np.random.default_rng(0)
        keep = [t for t in tickers if t in TICKERS]
        drift = np.linspace(-2e-4, 2e-4, len(keep))
        steps = rng.normal(0, 0.01, (len(DATES), len(keep))) + drift
        px = 100 * np.exp(np.cumsum(steps, axis=0))
        adj = pd.DataFrame(px, index=DATES, columns=keep)
        vol = pd.DataFrame(1e6, index=DATES, columns=keep)
        return adj, vol

    def download_macro(self, start, end):
        return pd.DataFrame({"date": DATES, "vix": 16.0,
                             "tnx_yield": 3.0, "irx_yield": 4.5})

    def download_benchmark(self, ticker, start, end):
        return pd.Series(1.0, index=DATES, name=ticker)


def _stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": TICKERS,
        "start_date": [pd.Timestamp("2014-01-01")] * N_TICKERS,
        "end_date": [pd.NaT] * N_TICKERS,
    })


def _sectors() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": TICKERS,
        "sector": ["Tech", "Health", "Energy", "Financials"] * (N_TICKERS // 4),
    })


def _run(tmp_path, *extra: str):
    argv = [
        "train-sp500",
        "--start", "2015-01-01", "--train-end", "2017-12-31",
        "--test-start", "2018-01-01",
        "--sample-n", str(N_TICKERS),
        "--no-optuna", "--skip-earnings", "--no-macro-merge", "--no-snapshot",
        "--horizon", "21", "--wf-top-k", "5", "--min-coverage", "0",
        *extra,
    ]
    with (
        patch("sys.argv", argv),
        patch.object(cli, "get_provider", return_value=_FakeProvider()),
        patch.object(cli, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
    ):
        cli.main()


# ---------------------------------------------------------------------------
# The pipeline runs
# ---------------------------------------------------------------------------


def test_the_pipeline_completes_and_reports(tmp_path, capsys) -> None:
    _run(tmp_path, "--skip-walk-forward")
    out = capsys.readouterr().out
    assert "Union tickers overlapping window" in out
    assert "Train (" in out and "Test  (" in out


def test_a_model_is_written_with_metadata_the_live_path_accepts(tmp_path) -> None:
    """The training and live halves must agree on the artifact contract."""
    from stock_predictor.predict import load_model, resolve_universe_seed

    out = tmp_path / "m.pkl"
    _run(tmp_path, "--skip-walk-forward", "--output-model", str(out), "--seed", "7")

    assert out.exists()
    model, meta = load_model(out)          # raises if the contract is broken
    assert model is not None
    assert meta["horizon"] == 21
    assert resolve_universe_seed(None, meta) == 7
    assert json.loads(out.with_suffix(".meta.json").read_text())["seed"] == 7


def test_the_walk_forward_scores_every_tradable_row(tmp_path) -> None:
    """Regression for the wiring bug: main must hand the walk-forward the
    *scoring* selection, so the newest sessions — the ones with no forward
    return yet — still get a score. Feeding it the training selection silently
    deleted the final horizon."""
    scores = tmp_path / "wf.parquet"
    _run(tmp_path, "--wf-scores-path", str(scores))

    assert scores.exists()
    panel = pd.read_parquet(scores)
    assert "has_label" in panel.columns
    unlabelled = panel[~panel["has_label"]]
    assert len(unlabelled) > 0, "the final horizon must survive into the panel"
    assert unlabelled["prob"].notna().all(), "every tradable row must be scored"

    width = panel.groupby("date").size()
    assert width.tail(10).min() > N_TICKERS // 2, (
        "the panel must not taper to a handful of names at the live edge"
    )


def test_the_scored_panel_reaches_the_last_downloaded_session(tmp_path) -> None:
    scores = tmp_path / "wf.parquet"
    _run(tmp_path, "--wf-scores-path", str(scores))
    panel = pd.read_parquet(scores)
    assert panel["date"].max() == DATES[-1]


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_an_overlapping_train_test_window_exits(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "--skip-walk-forward",
             "--train-end", "2019-01-01", "--test-start", "2018-01-01")
    assert "train-end" in str(exc.value)


def test_the_backtest_stage_runs_off_the_walk_forward_scores(tmp_path, capsys) -> None:
    _run(tmp_path, "--run-backtest")
    assert "BACKTEST" in capsys.readouterr().out.upper()


def test_a_saved_model_is_a_real_pickle_not_a_stub(tmp_path) -> None:
    out = tmp_path / "m.pkl"
    _run(tmp_path, "--skip-walk-forward", "--output-model", str(out))
    with open(out, "rb") as fh:
        payload = pickle.load(fh)
    assert hasattr(payload["model"], "predict")
    assert payload["meta"]["feature_cols"], "a model with no features is useless"


# ---------------------------------------------------------------------------
# Reproducibility artifacts
# ---------------------------------------------------------------------------


def _run_with_snapshot(tmp_path, *extra: str):
    """Same rig, but with the snapshot machinery on instead of --no-snapshot."""
    argv = [
        "train-sp500",
        "--start", "2015-01-01", "--train-end", "2017-12-31",
        "--test-start", "2018-01-01", "--sample-n", str(N_TICKERS),
        "--no-optuna", "--skip-earnings", "--no-macro-merge",
        "--horizon", "21", "--wf-top-k", "5", "--min-coverage", "0",
        "--snapshot-dir", str(tmp_path / "run"),
        *extra,
    ]
    with (
        patch("sys.argv", argv),
        patch.object(cli, "get_provider", return_value=_FakeProvider()),
        patch.object(cli, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
    ):
        cli.main()
    return tmp_path / "run"


def test_a_run_writes_a_manifest_and_its_input_snapshots(tmp_path) -> None:
    root = _run_with_snapshot(tmp_path, "--skip-walk-forward")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    for name in ("stints", "equity_prices_long", "labeled", "features_clean"):
        assert name in manifest["snapshots"], f"{name} must be snapshotted"
    assert (root / "stints.parquet").exists()


def test_the_manifest_records_the_resolved_run_configuration(tmp_path) -> None:
    root = _run_with_snapshot(tmp_path, "--skip-walk-forward", "--seed", "3")
    config = json.loads((root / "manifest.json").read_text())["config"]
    assert config["seed"] == 3
    assert config["horizon"] == 21
    assert config["objective"] in ("rank", "binary")


def test_the_manifest_records_universe_coverage(tmp_path) -> None:
    """A run that quietly downloaded half its universe must be identifiable
    after the fact."""
    root = _run_with_snapshot(tmp_path, "--skip-walk-forward")
    universe = json.loads((root / "manifest.json").read_text())["universe"]
    assert universe["requested"] == N_TICKERS
    assert "coverage" in universe


def test_a_saved_model_is_hashed_into_the_manifest(tmp_path) -> None:
    out = tmp_path / "m.pkl"
    root = _run_with_snapshot(tmp_path, "--skip-walk-forward",
                              "--output-model", str(out))
    artifact = json.loads((root / "manifest.json").read_text())["model_artifact"]
    assert len(artifact["sha256"]) == 64


def test_plots_are_written_when_asked_for(tmp_path) -> None:
    plots = tmp_path / "plots"
    _run(tmp_path, "--plots-dir", str(plots))
    assert plots.exists()
    assert any(plots.glob("*.png")), "a --plots-dir run must produce figures"


def test_a_backtest_without_scores_says_so_rather_than_crashing(tmp_path, capsys) -> None:
    _run(tmp_path, "--run-backtest", "--skip-walk-forward")
    assert "No walk-forward scores" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Production refit (REFIT=1 / --train-through-latest)
# ---------------------------------------------------------------------------


def test_a_scheduled_refit_completes(tmp_path, capsys) -> None:
    """The whole defect, end to end.

    Refit set test_start to the day after the newest labelable session, purged
    another full horizon off the training set, and handed the resulting empty
    test frame to evaluation, which cannot score zero rows. The scheduled
    workflow sets REFIT=1, so every monthly run failed after paying for the fit.
    """
    model = tmp_path / "model_candidate.pkl"
    _run(tmp_path, "--train-through-latest", "--output-model", str(model))
    out = capsys.readouterr().out
    assert "Refit mode" in out
    assert model.exists()


def test_a_refit_trains_past_the_evaluation_train_end(tmp_path) -> None:
    """It has to actually learn something newer, which was the point of the flag."""
    refit_model = tmp_path / "refit.pkl"
    _run(tmp_path, "--train-through-latest", "--output-model", str(refit_model))
    with open(refit_model, "rb") as fh:
        refit_meta = pickle.load(fh)["meta"]

    plain_model = tmp_path / "plain.pkl"
    _run(tmp_path, "--output-model", str(plain_model))
    with open(plain_model, "rb") as fh:
        plain_meta = pickle.load(fh)["meta"]

    assert refit_meta["fitted_through"] > plain_meta["fitted_through"]
    # And it records the window it used, not the --train-end flag it ignored.
    assert refit_meta["train_end"] != "2017-12-31"
    assert refit_meta["test_start"] is None


def test_a_refit_does_not_claim_metrics_it_never_measured(tmp_path) -> None:
    refit_model = tmp_path / "refit.pkl"
    _run(tmp_path, "--train-through-latest", "--output-model", str(refit_model))
    with open(refit_model, "rb") as fh:
        meta = pickle.load(fh)["meta"]
    assert np.isnan(meta["metrics"]["pr_auc"]), "no test set means no PR-AUC"
