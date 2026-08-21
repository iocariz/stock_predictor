"""A decision should name itself the same way twice.

Cohort IDs were `uuid.uuid4().hex[:8]`, so re-running the same signal produced
different identifiers for the same decision. Idempotency rested entirely on
`last_signal_date`: correct, but unverifiable — nothing downstream could tell
whether two runs had reached the same conclusion.

A deterministic ID is also the thing a broker's client-order-ID has to be, so
this is the prerequisite for submitting orders at all: the guarantee that a
retry after a timeout is recognised as the same order rather than a second one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.execution import cohort_id
from stock_predictor.portfolio import PortfolioState, generate_orders

DATES = pd.bdate_range(end="2024-06-14", periods=200)
PICKS = [
    {"ticker": f"T{i:02d}", "prob": 0.9 - i * 0.01, "adj_close": 100.0}
    for i in range(30)
]
PRICES = {p["ticker"]: p["adj_close"] for p in PICKS}


# ---------------------------------------------------------------------------
# The identifier
# ---------------------------------------------------------------------------


def test_the_same_decision_yields_the_same_id() -> None:
    a = cohort_id("2024-06-14", ["AAA", "BBB"])
    b = cohort_id("2024-06-14", ["AAA", "BBB"])
    assert a == b


def test_order_of_the_names_does_not_matter() -> None:
    """The basket is a set; the id must not depend on how it was enumerated."""
    assert cohort_id("2024-06-14", ["BBB", "AAA"]) == cohort_id("2024-06-14", ["AAA", "BBB"])


def test_a_different_day_is_a_different_cohort() -> None:
    assert cohort_id("2024-06-14", ["AAA"]) != cohort_id("2024-06-17", ["AAA"])


def test_a_different_basket_is_a_different_cohort() -> None:
    assert cohort_id("2024-06-14", ["AAA"]) != cohort_id("2024-06-14", ["BBB"])


def test_the_salt_separates_otherwise_identical_decisions() -> None:
    """Two configurations can pick the same names on the same day and still be
    different positions — a 10-day hold is not a 63-day hold."""
    assert cohort_id("2024-06-14", ["AAA"], salt="10") != cohort_id(
        "2024-06-14", ["AAA"], salt="63"
    )


def test_the_id_is_short_stable_hex() -> None:
    out = cohort_id("2024-06-14", ["AAA"])
    assert len(out) == 8
    assert all(c in "0123456789abcdef" for c in out)


def test_an_empty_basket_still_yields_an_id() -> None:
    assert len(cohort_id("2024-06-14", [])) == 8


def test_ids_are_stable_across_processes() -> None:
    """PYTHONHASHSEED randomizes str.__hash__, so a hash-based id would drift
    between runs. This must not."""
    import subprocess
    import sys

    code = (
        "from stock_predictor.execution import cohort_id;"
        "print(cohort_id('2024-06-14', ['AAA','BBB'], salt='63'))"
    )
    out = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            check=True, env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(out) == 1, f"id drifted across hash seeds: {out}"


# ---------------------------------------------------------------------------
# Reached through order generation
# ---------------------------------------------------------------------------


def _orders(as_of: str = "2024-06-14", holding_days: int = 10):
    return generate_orders(
        PortfolioState(cash=100_000.0),
        PICKS, PRICES,
        top_n=5, max_cohorts=2, holding_days=holding_days, slippage_bps=0.0,
        as_of=as_of, trading_dates=DATES.to_numpy(), force=True,
    )


def test_two_identical_runs_produce_identical_cohort_ids() -> None:
    """The property that makes a re-run auditable rather than merely harmless."""
    a, _ = _orders()
    b, _ = _orders()
    assert {o.cohort_id for o in a} == {o.cohort_id for o in b}
    assert len({o.cohort_id for o in a}) == 1, "one cohort, one id"


def test_identical_runs_agree_on_every_order() -> None:
    a, sa = _orders()
    b, sb = _orders()
    assert [(o.action, o.ticker, o.shares, o.cohort_id) for o in a] == [
        (o.action, o.ticker, o.shares, o.cohort_id) for o in b
    ]
    assert [p.cohort_id for p in sa.positions] == [p.cohort_id for p in sb.positions]


def test_a_different_holding_period_is_a_different_cohort() -> None:
    a, _ = _orders(holding_days=10)
    b, _ = _orders(holding_days=63)
    assert {o.cohort_id for o in a} != {o.cohort_id for o in b}


def test_a_different_signal_day_is_a_different_cohort() -> None:
    a, _ = _orders(as_of="2024-06-14")
    b, _ = _orders(as_of="2024-06-13")
    assert {o.cohort_id for o in a} != {o.cohort_id for o in b}


def test_no_uuid_remains_in_order_generation() -> None:
    """Regression: a stray uuid4 anywhere here reintroduces the drift."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "stock_predictor" / "portfolio.py").read_text()
    assert "uuid4" not in src


def test_rank_hold_ids_are_deterministic_too() -> None:
    from stock_predictor.portfolio import generate_orders_rank_hold

    def run():
        return generate_orders_rank_hold(
            PortfolioState(cash=100_000.0), PICKS, PRICES,
            top_n=5, exit_rank=30, slippage_bps=0.0, as_of="2024-06-14",
            trading_dates=DATES.to_numpy(), force=True,
        )[0]

    assert {o.cohort_id for o in run()} == {o.cohort_id for o in run()}


def test_numpy_ticker_types_do_not_change_the_id() -> None:
    """Tickers can arrive as numpy strings from a pivoted panel."""
    assert cohort_id("2024-06-14", [np.str_("AAA")]) == cohort_id("2024-06-14", ["AAA"])
