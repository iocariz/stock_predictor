"""A run's scores and its execution prices must belong to each other.

`run_pipeline.sh` defaults `EXECUTION_PRICES` to
`artifacts/hybrid_adj_close.parquet` and uses it *if the file exists* — but
nothing in `src/`, `scripts/` or CI ever writes that file. In this workspace it
existed only because an ad-hoc script produced it days earlier:

    scores          409 sessions, 2025-01-02 -> 2026-08-20
    execution prices 4181 sessions, 2010-01-04 -> 2026-08-18

Two scored sessions had no execution row and the pipeline paired them without
comment. A missing file degraded just as quietly: the backtest simply dropped
back to forward-filled prices, which on rank-hold is the difference between
+17.28% and +22.95%.

Coverage is therefore checked rather than assumed, and a run records the panel
it was measured against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.bundle import (
    BundleMismatch,
    describe_bundle,
    validate_execution_panel,
)

DATES = pd.bdate_range("2024-01-01", periods=50)
TICKERS = ["AAA", "BBB", "CCC"]


def _scored(dates=None, tickers=None) -> pd.DataFrame:
    dates = DATES if dates is None else dates
    tickers = TICKERS if tickers is None else tickers
    return pd.DataFrame([
        {"date": d, "ticker": t, "prob": 0.5, "adj_close": 100.0}
        for d in dates for t in tickers
    ])


def _panel(dates=None, tickers=None) -> pd.DataFrame:
    dates = DATES if dates is None else dates
    tickers = TICKERS if tickers is None else tickers
    return pd.DataFrame(100.0, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# The coherent case
# ---------------------------------------------------------------------------


def test_a_matching_bundle_passes() -> None:
    assert validate_execution_panel(_scored(), _panel()) == []


def test_extra_history_and_extra_tickers_are_fine() -> None:
    """The execution panel is the full download; it is expected to be wider
    and longer than the point-in-time scored panel."""
    wide = _panel(dates=pd.bdate_range("2020-01-01", periods=1200),
                  tickers=[*TICKERS, "DDD", "EEE"])
    assert validate_execution_panel(_scored(), wide) == []


# ---------------------------------------------------------------------------
# The observed failure
# ---------------------------------------------------------------------------


def test_sessions_missing_from_the_execution_panel_are_reported() -> None:
    """The real case: scores to 2026-08-20, prices to 2026-08-18."""
    short = _panel(dates=DATES[:-2])
    findings = validate_execution_panel(_scored(), short)
    assert [f.kind for f in findings] == ["date_coverage"]
    assert "2" in findings[0].detail


def test_tickers_missing_from_the_execution_panel_are_reported() -> None:
    findings = validate_execution_panel(_scored(), _panel(tickers=["AAA"]))
    assert any(f.kind == "ticker_coverage" for f in findings)
    assert any("BBB" in f.detail for f in findings)


def test_both_gaps_are_reported_together() -> None:
    bad = _panel(dates=DATES[:-2], tickers=["AAA"])
    assert {f.kind for f in validate_execution_panel(_scored(), bad)} == {
        "date_coverage", "ticker_coverage",
    }


def test_an_empty_panel_is_a_finding_not_a_pass() -> None:
    assert validate_execution_panel(_scored(), pd.DataFrame())


def test_a_missing_panel_is_a_finding() -> None:
    """Absent degrades as quietly as stale: the backtest just falls back to
    forward-filled prices."""
    findings = validate_execution_panel(_scored(), None)
    assert [f.kind for f in findings] == ["missing"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_a_non_datetime_index_is_rejected() -> None:
    bad = _panel()
    bad.index = range(len(bad))
    assert any(f.kind == "schema" for f in validate_execution_panel(_scored(), bad))


def test_a_non_numeric_column_is_rejected() -> None:
    bad = _panel()
    bad["AAA"] = "not a price"
    assert any(f.kind == "schema" for f in validate_execution_panel(_scored(), bad))


def test_duplicate_sessions_are_rejected() -> None:
    dup = pd.concat([_panel(), _panel().iloc[[0]]])
    assert any(f.kind == "schema" for f in validate_execution_panel(_scored(), dup))


# ---------------------------------------------------------------------------
# Reporting and raising
# ---------------------------------------------------------------------------


def test_findings_describe_themselves() -> None:
    text = describe_bundle(validate_execution_panel(_scored(), _panel(dates=DATES[:-2])))
    assert "date_coverage" in text
    assert describe_bundle([]) == ""


def test_strict_mode_raises_with_the_reason() -> None:
    with pytest.raises(BundleMismatch, match="date_coverage"):
        validate_execution_panel(_scored(), _panel(dates=DATES[:-2]), strict=True)


def test_strict_mode_passes_a_good_bundle() -> None:
    assert validate_execution_panel(_scored(), _panel(), strict=True) == []


def test_a_tolerance_allows_a_known_lag() -> None:
    """Some vendors settle a session late; that is a decision, not an accident."""
    findings = validate_execution_panel(
        _scored(), _panel(dates=DATES[:-2]), max_missing_sessions=2,
    )
    assert findings == []
