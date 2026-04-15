from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.calendar_features import CALENDAR_FEATURE_COLS, add_calendar_features


def test_add_calendar_features_columns() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-03-02", "2020-03-03"]), "ticker": ["A", "A"]})
    out = add_calendar_features(df)
    for c in CALENDAR_FEATURE_COLS:
        assert c in out.columns


def test_fomc_day_zero() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-03-03"]), "x": [1]})
    out = add_calendar_features(df)
    assert out["cal_days_to_next_fomc"].iloc[0] == 0
