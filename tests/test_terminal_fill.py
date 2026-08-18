"""Terminal-history censoring (P2-1).

A forward return needs `horizon` future sessions, so dropping rows without
one silently removes each delisted name's *final* horizon sessions — exactly
the terminal decline those names are in the panel to represent. The hybrid
provider recovers 148 departed members; this decides whether recovering them
means anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import build_labeled_panel

DATES = pd.bdate_range("2024-01-01", periods=40)


def _prices(*, dies_at: int | None = 30, crash: bool = True) -> pd.DataFrame:
    alive = np.linspace(100, 110, len(DATES))
    dying = np.linspace(100, 40, len(DATES)) if crash else np.full(len(DATES), 100.0)
    df = pd.DataFrame({"ALIVE": alive, "DYING": dying}, index=DATES)
    if dies_at is not None:
        df.loc[DATES[dies_at:], "DYING"] = np.nan
    return df


def test_delisted_tail_is_kept_and_priced_to_the_last_quote() -> None:
    """The final sessions before a delisting must survive, with the forward
    return measured to the terminal price a holder would actually realize."""
    px = _prices(dies_at=30)
    panel = build_labeled_panel(px, None, horizon=10, threshold=0.05)
    dy = panel[panel.ticker == "DYING"].sort_values("date")

    # The final session itself has no holding period, so the last usable
    # observation is the session before the delisting.
    assert dy["date"].max() == DATES[28]
    last_px = px["DYING"].iloc[29]
    row = dy[dy["date"] == DATES[25]].iloc[0]
    expected = last_px / px["DYING"].iloc[25] - 1
    assert row["fwd_ret"] == pytest.approx(expected)
    assert row["fwd_ret"] < 0, "a terminal decline must show as a loss"


def test_the_terminal_loss_is_actually_negative_and_large() -> None:
    """Regression for the bias itself: censoring these rows removed losses."""
    panel = build_labeled_panel(_prices(dies_at=30), None, horizon=10, threshold=0.05)
    tail = panel[(panel.ticker == "DYING") & (panel.date >= DATES[20])]
    assert len(tail) == 9, "sessions 20-28; the terminal session carries no return"
    assert (tail["fwd_ret"] < 0).all()
    assert tail["fwd_ret"].min() < -0.10


def test_a_still_trading_name_keeps_its_unknown_future_censored() -> None:
    """The panel ending is not a delisting: those rows must still be dropped,
    because the future genuinely is not known yet."""
    panel = build_labeled_panel(_prices(dies_at=None), None, horizon=10, threshold=0.05)
    alive = panel[panel.ticker == "ALIVE"]
    assert alive["date"].max() == DATES[-11], "no forward return, no row"


def test_full_horizon_returns_are_unchanged() -> None:
    """Rows that already had a complete horizon must not move."""
    px = _prices(dies_at=30)
    panel = build_labeled_panel(px, None, horizon=10, threshold=0.05)
    row = panel[(panel.ticker == "DYING") & (panel.date == DATES[10])].iloc[0]
    assert row["fwd_ret"] == pytest.approx(
        px["DYING"].iloc[20] / px["DYING"].iloc[10] - 1
    )


def test_terminal_fill_can_be_disabled() -> None:
    panel = build_labeled_panel(
        _prices(dies_at=30), None, horizon=10, threshold=0.05, terminal_fill=False,
    )
    dy = panel[panel.ticker == "DYING"]
    assert dy["date"].max() == DATES[19], "legacy behaviour drops the tail"


def test_labels_follow_the_filled_returns() -> None:
    px = _prices(dies_at=30, crash=False)
    px.loc[DATES[25:30], "DYING"] = 200.0          # bought out at a premium
    panel = build_labeled_panel(px, None, horizon=10, threshold=0.05)
    row = panel[(panel.ticker == "DYING") & (panel.date == DATES[24])].iloc[0]
    assert row["fwd_ret"] == pytest.approx(1.0)
    assert row["target_5pct"] == 1, "an acquisition premium is a real gain"
