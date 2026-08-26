"""When an execution panel is supplied, it prices the fills. All of them.

``specs.md:233`` — *"``adj_close`` MAY be included for diagnostics, but it is
not authoritative execution data."*

``_prepare_scored`` merged the two the wrong way round: it kept the scored
panel's price wherever it had one and consulted the execution panel only to
patch holes.

    scored price:             100
    execution price:          200
    fill price actually used: 100

So the execution panel was a gap-filler for a point-in-time-filtered signal
table, which is the one thing the spec says it must not be. The scored panel is
a decision signal; its ``adj_close`` rides along for inspection.

The strict reading has a consequence worth stating: a ticker in the scored
panel with no execution row now has *no* price, so its fills are rejected
rather than filled from the signal table. That is the intended behaviour —
``validate_execution_panel`` already reports the coverage gap — and it is why
the disagreement between the two sources is now measured rather than silently
resolved in the wrong direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.backtest import _prepare_scored
from stock_predictor.bundle import price_divergence

DATES = pd.bdate_range("2024-01-01", periods=6)


def _scored(price: float = 100.0, tickers=("A",)) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "ticker": t, "prob": 1.0, "adj_close": price}
        for d in DATES for t in tickers
    ])


def _execution(price: float = 200.0, tickers=("A",)) -> pd.DataFrame:
    return pd.DataFrame({t: [price] * len(DATES) for t in tickers}, index=DATES)


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


def test_the_execution_panel_wins_where_both_have_a_price() -> None:
    """The reported probe, stated as a requirement."""
    _, _, panel, _ = _prepare_scored(_scored(100.0), _execution(200.0))
    assert panel.at[DATES[2], "A"] == pytest.approx(200.0)


def test_the_scored_price_never_reaches_a_fill() -> None:
    _, _, panel, _ = _prepare_scored(_scored(100.0), _execution(200.0))
    assert not (panel["A"] == 100.0).any()


def test_a_ticker_absent_from_the_execution_panel_has_no_price() -> None:
    """Strict: the signal table does not get to price a fill by default."""
    scored = _scored(100.0, tickers=("A", "B"))
    _, _, panel, actual = _prepare_scored(scored, _execution(200.0, tickers=("A",)))
    assert "B" not in panel.columns or panel["B"].isna().all()
    assert "B" not in actual.columns or not actual["B"].any()


def test_extra_execution_tickers_are_kept() -> None:
    """The execution panel is the unfiltered download; it is meant to be wider."""
    _, _, panel, _ = _prepare_scored(_scored(), _execution(200.0, tickers=("A", "Z")))
    assert "Z" in panel.columns


def test_a_gap_in_the_execution_panel_is_a_gap() -> None:
    """Not an invitation to fall back to the scored price."""
    ex = _execution(200.0)
    ex.loc[DATES[3], "A"] = np.nan
    _, _, panel, actual = _prepare_scored(_scored(100.0), ex)
    assert not bool(actual.at[DATES[3], "A"]), "carried forward, not real"
    assert panel.at[DATES[3], "A"] == pytest.approx(200.0), "ffilled from execution"


def test_actual_reflects_the_execution_panel(baseline_free=None) -> None:
    ex = _execution(200.0)
    ex.loc[DATES[4], "A"] = np.nan
    _, _, _, actual = _prepare_scored(_scored(100.0), ex)
    assert bool(actual.at[DATES[0], "A"])
    assert not bool(actual.at[DATES[4], "A"])


# ---------------------------------------------------------------------------
# Without an execution panel, nothing changes
# ---------------------------------------------------------------------------


def test_without_an_execution_panel_the_scored_prices_are_used() -> None:
    """There is nothing else to price with; the caller is warned elsewhere."""
    _, _, panel, actual = _prepare_scored(_scored(100.0), None)
    assert panel.at[DATES[2], "A"] == pytest.approx(100.0)
    assert bool(actual.at[DATES[2], "A"])


def test_an_empty_execution_panel_is_the_same_as_none() -> None:
    _, _, panel, _ = _prepare_scored(_scored(100.0), pd.DataFrame())
    assert panel.at[DATES[2], "A"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# The scored price becomes a diagnostic, since that is all it is allowed to be
# ---------------------------------------------------------------------------


def test_divergence_is_measured() -> None:
    out = price_divergence(_scored(100.0), _execution(200.0))
    assert out["compared"] == len(DATES)
    assert out["disagreeing"] == len(DATES)
    # Relative to the *execution* price, which is the authoritative one:
    # |100 - 200| / 200.
    assert out["max_abs_pct"] == pytest.approx(0.5)


def test_agreement_reports_no_divergence() -> None:
    out = price_divergence(_scored(100.0), _execution(100.0))
    assert out["disagreeing"] == 0
    assert out["max_abs_pct"] == pytest.approx(0.0)


def test_divergence_ignores_cells_only_one_side_has() -> None:
    scored = _scored(100.0, tickers=("A", "B"))
    out = price_divergence(scored, _execution(100.0, tickers=("A",)))
    assert out["compared"] == len(DATES), "B has nothing to compare against"
    assert out["disagreeing"] == 0


def test_divergence_without_an_execution_panel_is_empty() -> None:
    assert price_divergence(_scored(), None)["compared"] == 0
