from __future__ import annotations

import pandas as pd

from stock_predictor.pit import filter_panel_to_pit, tickers_overlapping_window


def test_tickers_overlapping_window() -> None:
    stints = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-06-01", "2019-01-01"]),
            "end_date": pd.to_datetime(["2020-12-31", pd.NaT, "2020-06-01"]),
        }
    )
    out = tickers_overlapping_window(stints, "2020-06-01", "2021-01-01")
    assert "AAA" in out and "BBB" in out
    assert "CCC" not in out


def test_filter_panel_to_pit() -> None:
    stints = pd.DataFrame(
        {
            "ticker": ["X"],
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date": pd.to_datetime(["2020-06-30"]),
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-01", "2020-07-01"]),
            "ticker": ["X", "X"],
            "v": [1.0, 2.0],
        }
    )
    got = filter_panel_to_pit(panel, stints)
    assert len(got) == 1
    assert got["date"].iloc[0] == pd.Timestamp("2020-03-01")
