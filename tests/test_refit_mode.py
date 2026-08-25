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
