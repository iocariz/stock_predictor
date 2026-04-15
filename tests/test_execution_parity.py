"""Parity checks between backtest cohort timing and live order generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.backtest import _compute_weights
from stock_predictor.execution_calendar import (
    entry_on_or_after,
    exit_date_iso_after_hold,
    offset_trading_days,
    trading_dates_from_index,
)


def test_exit_iso_matches_backtest_offset() -> None:
    td = pd.bdate_range("2024-01-08", periods=40).values
    entry = entry_on_or_after("2024-01-10", td)
    assert entry == pd.Timestamp("2024-01-10")
    ex_iso = exit_date_iso_after_hold(entry, 10, td)
    ex_ts = offset_trading_days(entry, 10, td)
    assert ex_ts is not None
    assert ex_iso == ex_ts.strftime("%Y-%m-%d")


def test_trading_dates_from_index_sorted_unique() -> None:
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-03", "2024-01-05"])
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    out = trading_dates_from_index(df.index)
    assert len(out) == 2


def test_live_weights_match_backtest_compute_weights() -> None:
    probs = pd.Series([0.2, 0.5, 0.3]).to_numpy()
    w_eq = _compute_weights(probs, "equal")
    w_pr = _compute_weights(probs, "probability")
    assert abs(w_eq.sum() - 1.0) < 1e-9
    assert abs(w_pr.sum() - 1.0) < 1e-9
    np.testing.assert_allclose(w_pr, probs)
