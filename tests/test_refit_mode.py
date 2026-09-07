"""A production refit is not a small evaluation run.

``--train-through-latest`` exists so the monthly cron does not keep refitting
the same hard-coded ``train_end``. It set ``train_end`` to the newest labelable
session and ``test_start`` to the day after — and then went on doing everything
an evaluation run does, including purging a full horizon of training rows away
from a test period that, by construction, contains nothing:

    panel end:             2025-11-28
    newest labelable:      2025-09-02
    actual fitted through: 2025-06-05      <- 63 sessions thrown away
    test rows:             0

The empty test set then reached ``evaluate_test_set``, which cannot score a
zero-row frame. ``.github/workflows/train-sp500.yml`` sets ``REFIT=1`` on
schedule, so every monthly run would fail after paying for the fit.

Purging protects a test period from training rows that saw its prices. With no
test period there is nothing to protect, and the horizon of rows nearest the
present — the most relevant rows the model has — is exactly what was discarded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.cli import latest_trainable_end, split_train_test

HORIZON = 63
SESSIONS = pd.bdate_range(end="2025-11-28", periods=500)


def _labelled() -> pd.DataFrame:
    """Rows carrying a label — only sessions with a full forward window."""
    end = latest_trainable_end(SESSIONS, HORIZON)
    dates = SESSIONS[SESSIONS <= end]
    return pd.DataFrame({"date": dates, "target_5pct": 1.0, "f0": 0.5})


def _train_end() -> str:
    return str(latest_trainable_end(SESSIONS, HORIZON).date())


# ---------------------------------------------------------------------------
# Refit: train on everything labelable
# ---------------------------------------------------------------------------


def test_refit_fits_through_the_newest_labelable_session() -> None:
    """The defect: another full horizon was purged off the end."""
    train, _ = split_train_test(
        _labelled(), train_end=_train_end(), test_start=None,
        horizon=HORIZON, refit=True,
    )
    assert str(train["date"].max().date()) == _train_end() == "2025-09-02"


def test_refit_discards_no_sessions() -> None:
    labelled = _labelled()
    train, _ = split_train_test(
        labelled, train_end=_train_end(), test_start=None,
        horizon=HORIZON, refit=True,
    )
    assert len(train) == len(labelled), "a refit has no test period to purge against"


def test_refit_produces_no_test_set() -> None:
    _, test = split_train_test(
        _labelled(), train_end=_train_end(), test_start=None,
        horizon=HORIZON, refit=True,
    )
    assert test is None, "an empty frame would be passed on to evaluation; None is refused"


# ---------------------------------------------------------------------------
# Evaluation mode is unchanged
# ---------------------------------------------------------------------------


def test_evaluation_mode_still_purges() -> None:
    """Regression guard: the leak this purge prevents is real."""
    labelled = _labelled()
    test_start = "2025-01-02"
    train, test = split_train_test(
        labelled, train_end="2024-12-31", test_start=test_start,
        horizon=HORIZON, refit=False,
    )
    assert train["date"].max() < pd.Timestamp(test_start)
    gap = len(labelled[(labelled["date"] > train["date"].max())
                       & (labelled["date"] < pd.Timestamp(test_start))])
    assert gap >= HORIZON - 1, "a full horizon must separate train from test"
    assert test is not None and len(test) > 0


def test_evaluation_mode_test_set_starts_at_test_start() -> None:
    _, test = split_train_test(
        _labelled(), train_end="2024-12-31", test_start="2025-01-02",
        horizon=HORIZON, refit=False,
    )
    assert test["date"].min() >= pd.Timestamp("2025-01-02")


def test_an_evaluation_split_that_yields_no_test_rows_is_refused() -> None:
    """Silently evaluating nothing is how this went unnoticed for a release."""
    with pytest.raises(ValueError, match="no test rows"):
        split_train_test(
            _labelled(), train_end=_train_end(), test_start="2099-01-01",
            horizon=HORIZON, refit=False,
        )


# ---------------------------------------------------------------------------
# The arithmetic the report pinned down
# ---------------------------------------------------------------------------


def test_the_reported_arithmetic() -> None:
    assert str(SESSIONS[-1].date()) == "2025-11-28"
    assert _train_end() == "2025-09-02"

    old_train, _ = split_train_test(
        _labelled(), train_end=_train_end(),
        test_start=str(pd.Timestamp(_train_end()) + pd.Timedelta(days=1))[:10],
        horizon=HORIZON, refit=True,
    )
    # Under the fix the refit keeps everything; the old path stopped at
    # 2025-06-05, 63 sessions earlier.
    assert str(old_train["date"].max().date()) == "2025-09-02"


# ---------------------------------------------------------------------------
# Metadata records the window that was used, not the flag that was ignored
# ---------------------------------------------------------------------------


def test_a_refit_records_its_own_window_not_the_default_flag() -> None:
    """``build_model_meta`` read ``args.train_end``, which a refit overrides.
    A monthly build would have claimed the hard-coded default months after it
    had stopped using it."""
    import argparse

    from stock_predictor.cli import build_model_meta

    args = argparse.Namespace(
        start="2010-01-01", end=None, train_end="2024-12-31",
        test_start="2025-01-01", sample_n=500, horizon=HORIZON,
        threshold=0.05, skip_earnings=True, label_target="raw", seed=42,
    )
    meta = build_model_meta(
        args, feature_cols=["ret_1d"], objective="rank", tune_metric="auto",
        optuna_best={}, manual_params={}, n_trees=1, importance={},
        pr_auc=float("nan"), roc_auc=float("nan"), run_id="r1",
        snapshot_root=None,
        train_end="2025-09-02", test_start=None,       # what the refit used
        fitted_through=pd.Timestamp("2025-09-02"),
    )
    assert meta["train_end"] == "2025-09-02"
    assert meta["test_start"] is None
    assert meta["fitted_through"] == "2025-09-02"


# ---------------------------------------------------------------------------
# The manifest a refit writes
# ---------------------------------------------------------------------------


def test_the_manifest_survives_a_refits_absent_test_set() -> None:
    """A refit returns ``(train, None)`` by design, and the run manifest asked
    it for ``len(test)``.

    The model and its metadata were written first, so a refit produced a usable
    artifact and then died with ``TypeError: object of type 'NoneType' has no
    len()`` before recording the manifest — no run id, no input hashes, no
    provenance for the thing about to be deployed, and a non-zero exit. The
    monthly retrain runs with ``REFIT=1``, so it hit this every time.
    """
    import inspect

    from stock_predictor import cli

    src = inspect.getsource(cli.main)
    block = src[src.index('manifest["results"]'):]
    line = next(ln for ln in block.splitlines() if '"test_rows"' in ln)
    assert "len(test)" not in line or "if test is not None" in line, (
        f"the manifest still calls len() on a refit's absent test set: {line.strip()}")


def test_a_refit_records_that_it_had_no_test_set() -> None:
    """``None`` rather than ``0``: a refit has no test period, which is not the
    same claim as a test period that happened to be empty. The codebase draws
    that distinction everywhere else it matters."""
    import inspect

    from stock_predictor import cli

    src = inspect.getsource(cli.main)
    block = src[src.index('manifest["results"]'):]
    line = next(ln for ln in block.splitlines() if '"test_rows"' in ln)
    assert "None" in line, f"a refit should record test_rows as None: {line.strip()}"


# ---------------------------------------------------------------------------
# Deploying what a refit actually produced
# ---------------------------------------------------------------------------


def test_deploy_checks_freshness_against_a_panel_the_run_wrote() -> None:
    """A refit skips the walk-forward, so it writes no scored panel.

    ``deploy_model()`` passed ``--panel "$WF_SCORES"`` unconditionally, so the
    freshness check read a scores file left behind by some *other* run. In
    practice that meant refusing a candidate trained through 2026-06-05 because
    a panel from eighteen days earlier was 11 sessions behind — and it could
    equally have *passed* a stale candidate whose leftover panel happened to be
    recent. A validation that reads a file the run did not produce is not
    validating the run.

    The execution panel is written by every mode, refit included, so it is the
    one the check can rely on.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "run_pipeline.sh").read_text()
    block = src[src.index("deploy_model()"):]
    block = block[:block.index("\n}")]
    assert "EXECUTION_PRICES" in block, (
        "deploy still validates freshness against a panel a refit never writes")


def test_the_refit_path_names_the_panel_it_uses() -> None:
    """Whichever file the check reads, the operator has to be able to see which
    one it was — a silent fallback is how this went unnoticed."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "run_pipeline.sh").read_text()
    block = src[src.index("deploy_model()"):]
    block = block[:block.index("\n}")]
    assert "echo" in block or "--panel" in block
