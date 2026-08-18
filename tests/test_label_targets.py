"""Label targets: what the ranker is actually asked to rank."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import (
    LABEL_TARGETS,
    add_rank_labels,
    label_target_series,
)

DATES = pd.bdate_range("2024-01-01", periods=6)


def _panel() -> pd.DataFrame:
    """Two archetypes: a calm compounder and a volatile lottery ticket.

    Both average the same forward return, but VOLA's is far noisier — the
    pattern the +5%-in-10-days label rewards regardless of expected return.
    """
    rows = []
    for di, d in enumerate(DATES):
        rows.append({"date": d, "ticker": "CALM", "fwd_ret": 0.010, "vol_21d": 0.004})
        rows.append({"date": d, "ticker": "VOLA",
                     "fwd_ret": 0.10 if di % 2 else -0.08, "vol_21d": 0.050})
        rows.append({"date": d, "ticker": "MID", "fwd_ret": 0.005, "vol_21d": 0.010})
    return pd.DataFrame(rows)


def test_known_targets_are_exposed() -> None:
    """'excess' was removed: ranking within a date makes it identical to 'raw'.
    See tests/test_label_excess_removed.py."""
    assert set(LABEL_TARGETS) == {"raw", "vol_adj", "excess_vol_adj"}


def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="label_target"):
        label_target_series(_panel(), "sharpe_ratio")


def test_raw_target_is_the_forward_return() -> None:
    panel = _panel()
    out = label_target_series(panel, "raw")
    pd.testing.assert_series_equal(out, panel["fwd_ret"], check_names=False)


def test_vol_adjusted_target_rewards_return_per_unit_risk() -> None:
    """CALM earns 1% on 0.4% vol (ratio 2.5); VOLA earns 10% on 5% vol (ratio
    2.0). Raw return picks VOLA, vol-adjusted picks CALM — no ties, so the
    ordering is the assertion rather than row order."""
    panel = _panel()
    day = panel[panel["date"] == DATES[1]].reset_index(drop=True)

    raw = label_target_series(day, "raw")
    assert day.loc[raw.idxmax(), "ticker"] == "VOLA"

    adj = label_target_series(day, "vol_adj")
    assert day.loc[adj.idxmax(), "ticker"] == "CALM"
    assert adj.nunique() == len(day), "fixture must not tie"


def test_excess_vol_adjusted_combines_both() -> None:
    panel = _panel()
    out = label_target_series(panel, "excess_vol_adj")
    med = panel.groupby("date")["fwd_ret"].transform("median")
    expected = (panel["fwd_ret"] - med) / panel["vol_21d"]
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy())


def test_zero_volatility_does_not_produce_infinities() -> None:
    panel = _panel()
    panel.loc[panel["ticker"] == "CALM", "vol_21d"] = 0.0
    out = label_target_series(panel, "vol_adj")
    assert np.isfinite(out.dropna().to_numpy()).all()


def test_missing_vol_column_raises_for_vol_targets() -> None:
    panel = _panel().drop(columns=["vol_21d"])
    with pytest.raises(ValueError, match="vol_21d"):
        label_target_series(panel, "vol_adj")


# ---------------------------------------------------------------------------
# Grades built from the chosen target
# ---------------------------------------------------------------------------


def test_rank_grades_follow_the_selected_target() -> None:
    """The whole point: switching the target changes which name is graded best."""
    panel = _panel()
    day_mask = panel["date"] == DATES[1]

    raw = add_rank_labels(panel, n_grades=3, label_target="raw")
    adj = add_rank_labels(panel, n_grades=3, label_target="vol_adj")

    best_raw = raw[day_mask].sort_values("rank_grade").iloc[-1]["ticker"]
    best_adj = adj[day_mask].sort_values("rank_grade").iloc[-1]["ticker"]
    assert best_raw == "VOLA"
    assert best_adj == "CALM"


def test_rank_grades_default_to_raw_for_backward_compatibility() -> None:
    panel = _panel()
    default = add_rank_labels(panel, n_grades=3)
    explicit = add_rank_labels(panel, n_grades=3, label_target="raw")
    pd.testing.assert_series_equal(default["rank_grade"], explicit["rank_grade"])


def test_grades_stay_in_range_for_every_target() -> None:
    for target in LABEL_TARGETS:
        graded = add_rank_labels(_panel(), n_grades=5, label_target=target)
        assert graded["rank_grade"].between(0, 4).all(), target
