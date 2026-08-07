"""Unit tests for inference pipeline (no network calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.predict import build_inference_panel, score_universe


def _make_stints() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "end_date": [pd.NaT, pd.NaT],
    })


def _make_prices(n_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    adj_close = pd.DataFrame({
        "AAA": np.linspace(100, 110, n_days),
        "BBB": np.linspace(50, 55, n_days),
    }, index=dates)
    volume = pd.DataFrame({
        "AAA": np.random.default_rng(42).integers(1_000_000, 5_000_000, n_days),
        "BBB": np.random.default_rng(43).integers(500_000, 2_000_000, n_days),
    }, index=dates)
    return adj_close, volume


class _FakeModel:
    """Stub that mimics LGBMClassifier.predict_proba."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        # Always predict AAA higher than BBB (based on row order)
        probs = np.linspace(0.6, 0.3, n)
        return np.column_stack([1 - probs, probs])


def test_score_universe_ranking() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["AAA", "BBB"], 5),
        "adj_close": [100.0] * 10,
        "feat1": np.random.default_rng(1).random(10),
        "feat2": np.random.default_rng(2).random(10),
    })
    model = _FakeModel()
    scored = score_universe(model, panel, ["feat1", "feat2"])
    assert len(scored) == 2  # one row per ticker on latest date
    assert scored.iloc[0]["prob"] >= scored.iloc[1]["prob"]  # sorted descending


def test_score_universe_specific_date() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["AAA", "BBB"], 5),
        "adj_close": [100.0] * 10,
        "feat1": np.random.default_rng(1).random(10),
    })
    model = _FakeModel()
    scored = score_universe(model, panel, ["feat1"], score_date=pd.Timestamp("2024-01-03"))
    assert len(scored) == 2
    assert all(scored["prob"] > 0)


def test_score_universe_forward_fills_macro_on_all_nan() -> None:
    """When macro features are all-NaN on the score date, forward-fill from prior date."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["AAA", "BBB"], 3),
        "adj_close": [100.0] * 6,
        "feat1": np.random.default_rng(1).random(6),
        "vix": [15.0, 16.0, np.nan, 15.5, 16.5, np.nan],
    })
    model = _FakeModel()
    scored = score_universe(model, panel, ["feat1", "vix"])
    # Should succeed (forward-filled vix from day 2 -> day 3)
    assert len(scored) == 2
    assert all(scored["prob"] > 0)


def test_score_universe_missing_features() -> None:
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02")] * 2,
        "ticker": ["AAA", "BBB"],
        "adj_close": [100.0, 50.0],
    })
    model = _FakeModel()
    with pytest.raises(ValueError, match="missing"):
        score_universe(model, panel, ["nonexistent_feature"])


def test_forward_fill_never_touches_ticker_level_features() -> None:
    """Regression: only macro columns may be forward-filled from a prior date.

    A ticker-level feature that is all-NaN on the score date must NOT be
    smeared with one ticker's previous value — scoring should fail instead.
    """
    dates = pd.bdate_range("2024-01-02", periods=3)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["AAA", "BBB"], 3),
        "adj_close": [100.0] * 6,
        # ticker-level feature: valid earlier, all-NaN on the score date
        "feat1": [0.5, 0.6, np.nan, 0.7, 0.8, np.nan],
    })
    model = _FakeModel()
    with pytest.raises(ValueError, match="NaN features"):
        score_universe(model, panel, ["feat1"])


class _FakeRanker:
    """Mimics LGBMRanker: predict() only, arbitrary-scale scores."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.linspace(1.5, -2.0, len(X))


def test_score_universe_supports_ranker_models() -> None:
    """Live scoring must work with models that lack predict_proba (LGBMRanker)."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "ticker": np.repeat(["AAA", "BBB"], 3),
        "adj_close": [100.0] * 6,
        "feat1": np.random.default_rng(5).random(6),
    })
    scored = score_universe(_FakeRanker(), panel, ["feat1"])
    assert len(scored) == 2
    # Sorted descending by score; scores may be negative for rankers
    assert scored["prob"].iloc[0] >= scored["prob"].iloc[1]
