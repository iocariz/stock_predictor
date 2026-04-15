"""Tests for Yahoo ↔ FRED macro panel merge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.macro_merge import merge_macro_panels
from stock_predictor.training import _macro_fallback_panel


def test_merge_macro_panels_fills_primary_nan() -> None:
    d = pd.bdate_range("2024-01-02", periods=5)
    primary = pd.DataFrame({
        "date": d,
        "vix": [np.nan, 20.0, np.nan, 22.0, np.nan],
        "tnx_yield": [4.0, np.nan, np.nan, 4.1, 4.2],
        "irx_yield": [np.nan] * 5,
    })
    fallback = pd.DataFrame({
        "date": d,
        "vix": [15.0, 19.0, 21.0, np.nan, 23.0],
        "tnx_yield": [3.9, 4.05, 4.08, 4.12, np.nan],
        "irx_yield": [5.0, 5.1, 5.2, 5.3, 5.4],
    })
    out = merge_macro_panels(primary, fallback)
    assert out["vix"].iloc[0] == pytest.approx(15.0)  # filled from fallback
    assert out["vix"].iloc[1] == pytest.approx(20.0)  # kept primary
    assert out["vix"].iloc[2] == pytest.approx(21.0)  # filled
    assert out["tnx_yield"].iloc[1] == pytest.approx(4.05)  # primary NaN → fallback


def test_merge_macro_panels_empty_primary_uses_fallback() -> None:
    d = pd.bdate_range("2024-01-02", periods=3)
    primary = pd.DataFrame(columns=["date", "vix", "tnx_yield", "irx_yield"])
    fallback = pd.DataFrame({
        "date": d,
        "vix": [1.0, 2.0, 3.0],
        "tnx_yield": [4.0, 4.0, 4.0],
        "irx_yield": [5.0, 5.0, 5.0],
    })
    out = merge_macro_panels(primary, fallback)
    assert len(out) == 3
    assert out["vix"].iloc[-1] == pytest.approx(3.0)


def test_merge_macro_panels_both_empty() -> None:
    out = merge_macro_panels(pd.DataFrame(), pd.DataFrame())
    assert len(out) == 0
    assert list(out.columns) == ["date", "vix", "tnx_yield", "irx_yield"]


def test_macro_fallback_panel_yfinance_no_key_returns_none() -> None:
    assert _macro_fallback_panel("yfinance", "2024-01-01", "2024-02-01", "") is None
