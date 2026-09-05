"""Switching hold-mode on a live book mixes incompatible position semantics.

The three live engines disagree about what a position *is*. A fixed-hold
position is one leg of a basket that expires on a date. A rank-hold position is
open-ended and closed by rank decay. A long-short position may carry negative
shares and is closed by the next rebalance's target. Point one engine at
another's state file and it will read those positions under its own rules:
fixed-hold sweeps the long-short book on a date that never arrives, rank-hold
sees a short as a holding ranked worse than everything, long-short treats a
basket's cohort as its own to resize.

``predict.py`` has warned "Do not switch modes on an existing state file" in its
``--hold-mode`` help since rank-hold was added, and nothing enforced it. That
was tolerable while ``fixed`` was the only default anyone used. Making
``long-short`` the default makes an accidental switch likely rather than
hypothetical, so the warning becomes a check.

An empty book is always safe to switch, which is the case that matters here:
the paper state file has never traded.
"""

from __future__ import annotations

import pytest

from stock_predictor.portfolio import (
    LONG_SHORT_COHORT,
    OPEN_ENDED_EXPIRY,
    ModeMismatch,
    PortfolioState,
    Position,
    assert_mode_matches_state,
    init_state,
)


def _pos(*, cohort_id: str, expiry: str, shares: int = 10) -> Position:
    return Position(ticker="AAA", shares=shares, entry_price=100.0,
                    entry_date="2024-01-02", expiry_date=expiry,
                    cohort_id=cohort_id, last_price=100.0)


def _state(*positions: Position) -> PortfolioState:
    return PortfolioState(initial_capital=100_000.0, cash=100_000.0,
                          high_watermark=100_000.0, positions=positions)


FIXED = _pos(cohort_id="c-2024-01-02", expiry="2024-03-01")
RANK = _pos(cohort_id="c-2024-01-02", expiry=OPEN_ENDED_EXPIRY)
LS_LONG = _pos(cohort_id=LONG_SHORT_COHORT, expiry=OPEN_ENDED_EXPIRY)
LS_SHORT = _pos(cohort_id=LONG_SHORT_COHORT, expiry=OPEN_ENDED_EXPIRY, shares=-10)


# ---------------------------------------------------------------------------
# The case that matters today
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fixed", "rank", "long-short"])
def test_an_empty_book_can_become_any_mode(mode: str) -> None:
    """The paper state file has never traded, so this switch is free."""
    assert_mode_matches_state(init_state(), mode)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fixed", "rank"])
def test_a_long_short_book_is_not_handed_to_a_long_only_engine(mode: str) -> None:
    with pytest.raises(ModeMismatch, match="long-short"):
        assert_mode_matches_state(_state(LS_LONG, LS_SHORT), mode)


def test_a_long_only_book_is_not_handed_to_the_long_short_engine() -> None:
    with pytest.raises(ModeMismatch):
        assert_mode_matches_state(_state(FIXED), "long-short")


def test_a_fixed_book_is_not_handed_to_rank_hold() -> None:
    """Rank-hold would never close a position carrying a real expiry."""
    with pytest.raises(ModeMismatch):
        assert_mode_matches_state(_state(FIXED), "rank")


def test_a_rank_book_is_not_handed_to_fixed_hold() -> None:
    """The fixed-expiry sweep would wait for 9999-12-31."""
    with pytest.raises(ModeMismatch):
        assert_mode_matches_state(_state(RANK), "fixed")


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


def test_a_long_short_book_matches_its_own_mode() -> None:
    assert_mode_matches_state(_state(LS_LONG, LS_SHORT), "long-short")


def test_a_fixed_book_matches_its_own_mode() -> None:
    assert_mode_matches_state(_state(FIXED), "fixed")


def test_a_rank_book_matches_its_own_mode() -> None:
    assert_mode_matches_state(_state(RANK), "rank")


# ---------------------------------------------------------------------------
# The message has to be actionable
# ---------------------------------------------------------------------------


def test_the_error_names_both_modes_and_a_way_out() -> None:
    with pytest.raises(ModeMismatch) as exc:
        assert_mode_matches_state(_state(LS_LONG), "fixed")
    msg = str(exc.value)
    assert "long-short" in msg and "fixed" in msg
    assert "--allow-mode-switch" in msg, "no override named"
    assert "--state" in msg, "does not suggest a separate state file"


def test_the_override_lets_it_through() -> None:
    assert_mode_matches_state(_state(LS_LONG), "fixed", allow_switch=True)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_predict_checks_before_it_trades() -> None:
    import inspect

    from stock_predictor import predict

    src = inspect.getsource(predict.main)
    assert "assert_mode_matches_state" in src
    guard = src.index("assert_mode_matches_state")
    for generator in ("generate_orders_long_short(", "generate_orders_rank_hold("):
        assert src.index(generator) > guard, f"{generator} runs before the guard"


# ---------------------------------------------------------------------------
# One flag set, two CLIs
# ---------------------------------------------------------------------------


def _pipeline_flags(hold_mode: str) -> list[str]:
    """The long option names run_pipeline.sh emits for a given HOLD_MODE."""
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        ["bash", "-c",
         f'set -a; HOLD_MODE={hold_mode} DRY_RUN=1; '
         f'cd "{root}" && ./scripts/run_pipeline.sh predict'],
        capture_output=True, text=True, timeout=120,
    ).stdout
    return sorted(set(re.findall(r"--[a-z][a-z0-9-]+", out)))


def _accepted(cli: str) -> set[str]:
    import re
    import subprocess

    out = subprocess.run(["uv", "run", cli, "--help"],
                         capture_output=True, text=True, timeout=180).stdout
    return set(re.findall(r"--[a-z][a-z0-9-]+", out))


@pytest.mark.parametrize("mode", ["fixed", "rank", "long-short"])
def test_every_pipeline_flag_is_accepted_by_predict(mode: str) -> None:
    emitted = _pipeline_flags(mode)
    assert emitted, "the pipeline emitted no flags"
    unknown = [f for f in emitted if f not in _accepted("predict-sp500")]
    assert not unknown, f"predict-sp500 rejects {unknown}"


def test_the_long_short_flags_have_one_name_across_both_clis() -> None:
    """The bug this catches: ``--short-borrow-annual`` existed on
    predict-sp500 and ``--short-borrow`` on backtest-sp500, and
    ``strategy_flags`` feeds one list to both. ``run_pipeline.sh backtest``
    with HOLD_MODE=long-short exited on 'unrecognized arguments' -- the mode
    was unrunnable through the pipeline it had just been wired into.
    """
    shared = ("--decile", "--long-weight", "--short-weight",
              "--rebalance-every", "--min-names-per-side", "--short-borrow")
    predict, backtest = _accepted("predict-sp500"), _accepted("backtest-sp500")
    missing = [f for f in shared
               if f not in predict or f not in backtest]
    assert not missing, f"not on both CLIs: {missing}"
