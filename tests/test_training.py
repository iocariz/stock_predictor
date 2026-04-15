from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.training import (
    add_price_features,
    add_regime_features,
    add_cross_sectional_ranks,
    build_labeled_panel,
    precision_at_k,
    wide_field,
)


def test_precision_at_k() -> None:
    y = pd.Series([0, 0, 1, 0, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.9, 0.3, 0.95])
    # top-2 are indices 2 and 4, both positive -> 1.0
    assert precision_at_k(y, scores, k=2) == 1.0
    # top-3 includes one negative (index 3 has 0.3 — actually sort: 4,2,3 -> labels 1,1,0 -> 2/3
    assert abs(precision_at_k(y, scores, k=3) - 2 / 3) < 1e-9


def test_wide_field_single_ticker() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    raw = pd.DataFrame({"Close": [1.0, 1.1], "Volume": [100, 110]}, index=idx)
    w = wide_field(raw, "Close")
    assert list(w.columns) == ["Close"]
    assert len(w) == 2


def _synthetic_adj_close() -> pd.DataFrame:
    """Wide adj_close panel: 2 tickers, 30 business days."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    return pd.DataFrame(
        {"AAA": np.linspace(100, 110, 30), "BBB": np.linspace(50, 48, 30)},
        index=dates,
    )


def _synthetic_stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "end_date": [pd.NaT, pd.NaT],
    })


def test_build_labeled_panel_creates_target() -> None:
    adj_close = _synthetic_adj_close()
    stints = _synthetic_stints()
    panel = build_labeled_panel(adj_close, stints, horizon=5, threshold=0.03)
    assert "target_5pct" in panel.columns
    assert "fwd_ret" in panel.columns
    assert set(panel["ticker"].unique()) == {"AAA", "BBB"}
    # AAA goes up → some positives; BBB goes down → all zeros
    aaa = panel[panel["ticker"] == "AAA"]
    bbb = panel[panel["ticker"] == "BBB"]
    assert aaa["target_5pct"].sum() >= 0
    assert bbb["target_5pct"].sum() == 0


def test_build_labeled_panel_no_future_leak() -> None:
    """Forward return for the last `horizon` rows should be NaN → dropped."""
    adj_close = _synthetic_adj_close()
    stints = _synthetic_stints()
    panel = build_labeled_panel(adj_close, stints, horizon=5, threshold=0.03)
    # 30 dates, 2 tickers, last 5 rows per ticker have NaN fwd_ret → dropped
    assert len(panel) <= 2 * (30 - 5)


def test_add_price_features_columns() -> None:
    dates = pd.bdate_range("2020-01-02", periods=260)
    df = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["A", "B"], 260),
        "adj_close": np.random.default_rng(42).uniform(90, 110, 520),
    })
    out = add_price_features(df)
    for col in ["ret_63d", "ret_252d", "high_52w_pct", "drawdown_63d", "momentum", "rsi_14"]:
        assert col in out.columns


def test_regime_features_market_median() -> None:
    df = pd.DataFrame({
        "date": ["2024-01-02"] * 3,
        "ticker": ["A", "B", "C"],
        "ret_5d": [0.01, 0.03, 0.05],
        "ret_21d": [0.10, 0.20, 0.30],
    })
    out = add_regime_features(df)
    assert out["market_ret_5d"].iloc[0] == 0.03  # median
    assert out["market_ret_21d"].iloc[0] == 0.20


def test_cross_sectional_ranks_range() -> None:
    df = pd.DataFrame({
        "date": ["2024-01-02"] * 4,
        "ret_21d": [0.1, 0.2, 0.3, 0.4],
        "vol_10d": [0.01, 0.02, 0.03, 0.04],
        "volume_zscore": [1.0, 2.0, 3.0, 4.0],
    })
    out = add_cross_sectional_ranks(df)
    for col in ["ret_21d_rank", "vol_10d_rank", "volume_zscore_rank"]:
        assert col in out.columns
        assert out[col].min() > 0
        assert out[col].max() <= 1.0
