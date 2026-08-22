"""Units carved out of ``predict.main``.

The live path was 40% covered because almost all of it lived inside one
150-line ``main``. Everything below is logic that decides what gets traded or
what the operator is told, so it is worth testing directly rather than
through a mocked end-to-end run.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta

import pandas as pd
import pytest

from stock_predictor.portfolio import Order, PortfolioState, Position
from stock_predictor.predict import (
    format_signal_report,
    load_model,
    missing_feature_columns,
    resolve_universe_seed,
    sample_mismatch_warning,
)

TODAY = date.today().isoformat()
LATER = (date.today() + timedelta(days=30)).isoformat()
EARLIER = (date.today() - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _write_model(path, meta) -> None:
    with open(path, "wb") as fh:
        pickle.dump({"model": "MODEL", "meta": meta}, fh)


def test_a_model_round_trips_with_its_metadata(tmp_path) -> None:
    p = tmp_path / "m.pkl"
    _write_model(p, {"feature_cols": ["a"], "horizon": 63})
    model, meta = load_model(p)
    assert model == "MODEL"
    assert meta["horizon"] == 63


@pytest.mark.parametrize("meta", [{"horizon": 63}, {"feature_cols": ["a"]}, {}])
def test_metadata_without_the_essentials_is_rejected(tmp_path, meta) -> None:
    """Scoring with the wrong feature list fails deep inside LightGBM with an
    unreadable error; failing here names the missing key."""
    p = tmp_path / "m.pkl"
    _write_model(p, meta)
    with pytest.raises(ValueError, match="feature_cols|horizon"):
        load_model(p)


# ---------------------------------------------------------------------------
# Universe reproducibility
# ---------------------------------------------------------------------------


def test_the_training_seed_is_reused_when_none_is_given() -> None:
    """The live universe must be the same draw the model was fitted on — an
    unrelated sample changes every cross-sectional feature."""
    assert resolve_universe_seed(None, {"seed": 7}) == 7


def test_an_explicit_seed_overrides_the_model() -> None:
    assert resolve_universe_seed(11, {"seed": 7}) == 11


def test_a_model_without_a_seed_falls_back_to_the_default() -> None:
    assert resolve_universe_seed(None, {}) == 42


def test_a_non_numeric_stored_seed_does_not_crash_the_run() -> None:
    assert resolve_universe_seed(None, {"seed": "not-a-number"}) == 42


def test_a_sample_size_mismatch_is_warned_about() -> None:
    msg = sample_mismatch_warning(100, {"sample_n": 500})
    assert msg is not None
    assert "100" in msg and "500" in msg


def test_a_matching_sample_size_is_silent() -> None:
    assert sample_mismatch_warning(500, {"sample_n": 500}) is None


def test_a_model_that_never_recorded_its_sample_size_is_silent() -> None:
    assert sample_mismatch_warning(500, {}) is None


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------


def test_features_the_model_wants_but_the_panel_lacks_are_named() -> None:
    panel = pd.DataFrame(columns=["ticker", "ret_1d"])
    assert missing_feature_columns(panel, ["ret_1d", "fund_roe"]) == {"fund_roe"}


def test_an_aligned_panel_reports_nothing_missing() -> None:
    panel = pd.DataFrame(columns=["ret_1d", "mom_21d"])
    assert missing_feature_columns(panel, ["ret_1d"]) == set()


# ---------------------------------------------------------------------------
# The operator-facing report
# ---------------------------------------------------------------------------


def _state(**kw) -> PortfolioState:
    base = dict(cash=50_000.0, positions=(
        Position("AAA", 10, 100.0, EARLIER, LATER, "c1", 105.0),
    ))
    base.update(kw)
    return PortfolioState(**base)


def _scored() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["BBB", "AAA"], "prob": [0.9, 0.4],
                         "adj_close": [50.0, 105.0]})


def _report(orders, state=None, *, halted=False) -> str:
    return "\n".join(format_signal_report(
        _scored(), orders, state or _state(), nav=100_000.0, drawdown=-0.02,
        halted=halted, top_n=15, max_drawdown=0.20,
    ))


def test_a_buy_is_reported_with_its_score_and_cost() -> None:
    out = _report((Order("BUY", "BBB", 20, 50.0, "c2", "new_pick"),))
    assert "BBB" in out
    assert "0.900" in out, "the score that justified the buy must be shown"
    assert "1,000" in out, "20 shares at 50 is $1,000"


def test_a_sell_reports_realized_pnl_against_the_entry() -> None:
    out = _report((Order("SELL", "AAA", 10, 120.0, "c1", "expiration"),))
    assert "SELL" in out
    assert "+200" in out, "(120 - 100) * 10"


def test_a_sell_with_no_matching_position_does_not_crash() -> None:
    """State and orders can disagree if state was edited between runs."""
    out = _report((Order("SELL", "ZZZ", 5, 10.0, "c9", "expiration"),))
    assert "ZZZ" in out


def test_a_halted_book_that_somehow_has_buys_says_so_loudly() -> None:
    """This test used to assert the opposite — that a halted book hides its
    buys — which is exactly how `--force-rebalance` trading through the kill
    switch stayed invisible: the banner said "no new positions" beside a
    summary line counting five of them.

    Order generation now refuses to produce buys while halted. If the two ever
    disagree again, the report must expose it rather than paper over it."""
    out = _report((Order("BUY", "BBB", 20, 50.0, "c2", "new_pick"),), halted=True)
    assert "KILL-SWITCH" in out
    assert "HALTED" in out
    assert "BBB" in out, "an order that exists must be visible"
    assert "this is a bug" in out, "and must be flagged as one"


def test_a_quiet_day_says_so_rather_than_printing_nothing() -> None:
    assert "No orders today" in _report(())


def test_active_holds_are_listed_and_expiring_ones_are_not() -> None:
    state = _state(positions=(
        Position("AAA", 10, 100.0, EARLIER, LATER, "c1", 105.0),
        Position("CCC", 5, 20.0, EARLIER, EARLIER, "c2", 21.0),
    ))
    out = _report((), state)
    assert "AAA" in out.split("HOLD")[1]
    assert "CCC" not in out.split("HOLD")[1], "an expiring lot is not a hold"


def test_the_summary_counts_every_leg() -> None:
    out = _report((
        Order("SELL", "AAA", 10, 120.0, "c1", "expiration"),
        Order("BUY", "BBB", 20, 50.0, "c2", "new_pick"),
    ))
    assert "1 sells, 1 buys, 1 holds" in out


def test_turnover_counts_both_sides() -> None:
    out = _report((
        Order("SELL", "AAA", 10, 100.0, "c1", "expiration"),
        Order("BUY", "BBB", 20, 50.0, "c2", "new_pick"),
    ))
    assert "$2,000" in out, "1,000 sold + 1,000 bought"


def test_the_drawdown_and_its_limit_are_both_shown() -> None:
    out = _report(())
    assert "-2.0%" in out
    assert "-20%" in out, "the operator needs the limit, not just the level"


def test_print_signal_report_emits_the_same_lines(capsys) -> None:
    """The printing wrapper must not diverge from the formatter under test."""
    from stock_predictor.predict import print_signal_report

    orders = (Order("BUY", "BBB", 20, 50.0, "c2", "new_pick"),)
    print_signal_report(_scored(), orders, _state(), 100_000.0, -0.02, False,
                        top_n=15, max_drawdown=0.20)
    printed = capsys.readouterr().out
    for line in format_signal_report(_scored(), orders, _state(), nav=100_000.0,
                                     drawdown=-0.02, halted=False, top_n=15,
                                     max_drawdown=0.20):
        assert line in printed
