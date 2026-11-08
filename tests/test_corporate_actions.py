"""A held name's history can be rescaled after it was bought; its entry price was not.

The live book stores a raw fill price and marks against the latest print. When
the vendor back-adjusts a ticker's history -- a dividend, a split, a spin-off
-- the latest print moves onto a new scale and the stored entry price stays on
the old one, so the difference is booked as P&L that never happened.

Corteva spun off Vylor (VYLR) on 2026-10-01, one share per share. The live
book was short 12 CTVA from $87.82. After the spin CTVA printed $11.90 and its
history was rescaled by ~0.15, and the book reported the whole $87.82 ->
$11.90 move as a +$911 gain. The short also owed 12 VYLR (~$842) it never
recorded; on a consistent basis the position was up ~$15, not $911.

The backtest is immune because it reads one adjusted panel end to end. These
tests pin the live path to the same scale -- and pin what it must not do on
its own: tell a split from a spin-off, which the factor alone cannot.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stock_predictor.corporate_actions import (
    CORPORATE_ACTION_THRESHOLD,
    describe_adjustments,
    rebase_to_adjusted_history,
)
from stock_predictor.portfolio import (
    OPEN_ENDED_EXPIRY,
    PortfolioState,
    Position,
    init_state,
    load_state,
    portfolio_value,
    save_state,
)

SESSIONS = pd.bdate_range("2026-08-24", "2026-11-06")
CTVA_IS_A_DISTRIBUTION = {"CTVA": "distribution"}


def _panel(**series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(series, index=SESSIONS)


def _flat(px: float) -> pd.Series:
    return pd.Series(px, index=SESSIONS, dtype=float)


def _ctva_adjusted() -> pd.Series:
    """CTVA as the vendor serves it after the spin: one continuous scale."""
    return _flat(11.90)


def _dividend(factor: float, px: float = 100.0) -> pd.Series:
    hist = _flat(px)
    hist[hist.index < pd.Timestamp("2026-10-15")] *= factor
    return hist


def _position(
    ticker: str = "CTVA",
    shares: int = -12,
    entry: float = 87.82,
    ref: float = 87.82,
    entry_date: str = "2026-09-04",
    last: float = 11.90,
    ref_date: str = "",
) -> Position:
    return Position(
        ticker=ticker,
        shares=shares,
        entry_price=entry,
        entry_date=entry_date,
        expiry_date=OPEN_ENDED_EXPIRY,
        cohort_id="long_short",
        last_price=last,
        entry_ref_price=ref,
        entry_ref_date=ref_date,
    )


def _state(*positions: Position) -> PortfolioState:
    return replace(init_state(100_000.0), positions=tuple(positions))


def _pnl(state: PortfolioState, prices: dict[str, float]) -> float:
    return sum(p.shares * (prices[p.ticker] - p.entry_price) for p in state.positions)


def _settle_ctva(state: PortfolioState) -> tuple[PortfolioState, tuple]:
    return rebase_to_adjusted_history(
        state, _panel(CTVA=_ctva_adjusted()), confirmed=CTVA_IS_A_DISTRIBUTION
    )


# --- a confirmed distribution ------------------------------------------------


def test_a_spin_off_does_not_book_its_distribution_as_profit() -> None:
    state = _state(_position())
    prices = {"CTVA": 11.90}
    assert _pnl(state, prices) == pytest.approx(910.6, abs=1)

    new, notes = _settle_ctva(state)

    assert abs(_pnl(new, prices)) < 1, f"spin-off still booked: {_pnl(new, prices):+.2f}"
    assert new.positions[0].entry_price == pytest.approx(11.90, rel=1e-6)
    assert len(notes) == 1 and notes[0].ticker == "CTVA" and notes[0].settled
    assert notes[0].factor == pytest.approx(11.90 / 87.82)
    assert notes[0].is_corporate_action


def test_the_distribution_is_settled_in_cash_so_nav_sees_it() -> None:
    """Rescaling the entry price alone fixes reported P&L and nothing else:
    NAV is cash + shares * mark and never reads entry_price, so the kill
    switch would still see the phantom gain. The short owes the distribution,
    so cash pays it -- here the stand-in for the 12 VYLR it never recorded."""
    state = _state(_position())
    prices = {"CTVA": 11.90}
    new, notes = _settle_ctva(state)

    owed = -12 * 87.82 * (1 - 11.90 / 87.82)
    assert notes[0].cash_adjustment == pytest.approx(owed)
    assert new.cash == pytest.approx(state.cash + owed)
    assert portfolio_value(new, prices) == pytest.approx(
        portfolio_value(state, prices) - 910.6, abs=1
    )
    assert new.high_watermark == state.high_watermark


def test_nav_and_reported_pnl_agree_after_the_rescale() -> None:
    """Cash from the open was -shares * entry. After the adjustment it must be
    -shares * (rescaled entry), or NAV and P&L describe different books."""
    state = _state(_position())
    new, _ = _settle_ctva(state)
    prices = {"CTVA": 11.90}
    open_cash = state.cash + (-12 * 87.82)  # the short's proceeds, taken back out
    assert portfolio_value(new, prices) - open_cash == pytest.approx(_pnl(new, prices))


def test_the_rescale_keeps_slippage_on_the_entry_price() -> None:
    """entry_price carries the fill's slippage; the reference does not. Both
    move by the same factor, so the slippage survives the rescale."""
    new, _ = _settle_ctva(_state(_position(entry=87.86, ref=87.82)))
    p = new.positions[0]
    assert p.entry_price / p.entry_ref_price == pytest.approx(87.86 / 87.82)


def test_an_unquoted_leg_has_its_last_price_rescaled_too() -> None:
    """A stale name is marked from last_price. Rescaling the entry but not the
    mark would move the error rather than remove it."""
    new, _ = _settle_ctva(_state(_position(last=87.50)))
    assert new.positions[0].last_price == pytest.approx(87.50 * 11.90 / 87.82)


def test_settling_twice_is_the_same_as_once() -> None:
    once, _ = _settle_ctva(_state(_position()))
    twice, notes = _settle_ctva(once)
    assert twice.positions == once.positions and twice.cash == once.cash
    assert notes == ()
    # And the default pass, with nothing confirmed, leaves it alone too.
    plain, more = rebase_to_adjusted_history(once, _panel(CTVA=_ctva_adjusted()))
    assert plain.positions == once.positions and more == ()


def test_the_input_state_is_not_mutated() -> None:
    state = _state(_position())
    snapshot = state.positions
    _settle_ctva(state)
    assert state.positions is snapshot
    assert state.positions[0].entry_price == 87.82


# --- what the code settles on its own: dividends -----------------------------


def test_a_long_is_credited_its_dividend() -> None:
    """The same rule for the other sign: a long receives the distribution.
    Dividends were never credited live, so longs ran poorer than the
    backtest's total-return panel by every dividend paid."""
    pos = _position(ticker="AAA", shares=10, entry=100.0, ref=100.0, last=100.0)
    state = _state(pos)
    new, notes = rebase_to_adjusted_history(state, _panel(AAA=_dividend(0.99)))
    assert notes[0].cash_adjustment == pytest.approx(10 * 100.0 * 0.01)
    assert new.cash == pytest.approx(state.cash + 10.0)
    assert "dividend" in describe_adjustments(notes)


def test_a_short_pays_its_dividend() -> None:
    pos = _position(ticker="AAA", shares=-10, entry=100.0, ref=100.0, last=100.0)
    state = _state(pos)
    new, _ = rebase_to_adjusted_history(state, _panel(AAA=_dividend(0.99)))
    assert new.cash == pytest.approx(state.cash - 10.0)


def test_a_dividend_sized_adjustment_is_settled_and_not_flagged() -> None:
    pos = _position(ticker="AAA", shares=10, entry=100.0, ref=100.0, last=100.0)
    new, notes = rebase_to_adjusted_history(_state(pos), _panel(AAA=_dividend(0.995)))
    assert new.positions[0].entry_price == pytest.approx(99.5)
    assert len(notes) == 1 and notes[0].settled and not notes[0].is_corporate_action
    assert 0.005 < CORPORATE_ACTION_THRESHOLD


def test_vendor_rounding_noise_is_not_settled() -> None:
    """Re-served history wobbles in the last digit. Settling it would churn
    cash daily and cry "corporate action" at nothing."""
    pos = _position(ticker="AAA", shares=10, entry=10.0, ref=10.0, last=10.0)
    new, notes = rebase_to_adjusted_history(_state(pos), _panel(AAA=_flat(10.005)))
    assert new.positions == (pos,)
    assert notes == ()


# --- what the code will not decide on its own --------------------------------


def test_a_large_adjustment_waits_for_an_operator() -> None:
    """A 2:1 split and a spin-off worth half the parent both read 0.5, and
    settling one as the other leaves the wrong exposure on the book."""
    pos = _position(ticker="AAA", shares=100, entry=100.0, ref=100.0, last=50.0)
    state = _state(pos)
    new, notes = rebase_to_adjusted_history(state, _panel(AAA=_flat(50.0)))
    assert new.positions == (pos,)
    assert new.cash == state.cash
    assert len(notes) == 1 and not notes[0].settled
    assert "PENDING CORPORATE ACTION" in describe_adjustments(notes)


def test_a_confirmed_split_scales_shares_and_moves_no_cash() -> None:
    pos = _position(ticker="AAA", shares=100, entry=100.0, ref=100.0, last=50.0)
    state = _state(pos)
    new, notes = rebase_to_adjusted_history(
        state, _panel(AAA=_flat(50.0)), confirmed={"AAA": "split"}
    )
    p = new.positions[0]
    assert p.shares == 200
    assert p.entry_price == pytest.approx(50.0)
    assert new.cash == state.cash
    assert portfolio_value(new, {"AAA": 50.0}) == pytest.approx(state.cash + 200 * 50.0)
    assert notes[0].kind == "split" and (notes[0].shares_before, notes[0].shares_after) == (
        100,
        200,
    )


def test_a_confirmed_reverse_split_on_a_short() -> None:
    pos = _position(ticker="AAA", shares=-100, entry=2.0, ref=2.0, last=20.0)
    new, _ = rebase_to_adjusted_history(
        _state(pos), _panel(AAA=_flat(20.0)), confirmed={"AAA": "split"}
    )
    assert new.positions[0].shares == -10
    assert new.positions[0].entry_price == pytest.approx(20.0)


def test_a_split_that_leaves_fractional_shares_is_refused() -> None:
    pos = _position(ticker="AAA", shares=7, entry=100.0, ref=100.0, last=66.67)
    with pytest.raises(ValueError, match="not a whole number"):
        rebase_to_adjusted_history(
            _state(pos), _panel(AAA=_flat(100.0 * 2 / 3)), confirmed={"AAA": "split"}
        )


def test_an_unknown_confirmation_is_refused() -> None:
    with pytest.raises(ValueError, match="confirmed kinds"):
        rebase_to_adjusted_history(
            _state(_position()), _panel(CTVA=_ctva_adjusted()), confirmed={"CTVA": "spinoff"}
        )


# --- legs written before the reference existed ---------------------------------


def test_a_legacy_position_adopts_its_reference_without_moving() -> None:
    """Positions written before the reference existed carry 0. Their entry
    price includes slippage, so within tolerance it is adopted, not settled --
    otherwise every legacy leg would be nudged by its own slippage."""
    pos = _position(ticker="AAA", shares=10, entry=100.05, ref=0.0, last=101.0)
    new, notes = rebase_to_adjusted_history(_state(pos), _panel(AAA=_flat(100.0)))
    p = new.positions[0]
    assert p.entry_price == 100.05
    assert p.entry_ref_price == pytest.approx(100.0)
    assert p.entry_ref_date == "2026-09-04"
    assert notes == ()


def test_a_legacy_gap_is_reported_but_never_settled_on_its_own() -> None:
    """With no recorded reference even a dividend-sized gap could be an entry
    price from another feed. The live CTVA leg is a spin-off; a fill priced off
    a different vendor would look the same. Neither moves cash unconfirmed."""
    for hist in (_ctva_adjusted(), _flat(87.82 * 0.98)):
        pos = _position(ref=0.0)
        state = _state(pos)
        new, notes = rebase_to_adjusted_history(state, _panel(CTVA=hist))
        assert new.positions == (pos,)
        assert new.cash == state.cash
        assert len(notes) == 1 and not notes[0].settled


def test_an_operator_can_settle_a_legacy_gap() -> None:
    """The live CTVA leg: no reference recorded, settled by name."""
    state = _state(_position(ref=0.0))
    new, notes = _settle_ctva(state)
    assert new.positions[0].entry_price == pytest.approx(11.90, rel=1e-6)
    assert new.positions[0].entry_ref_price == pytest.approx(11.90, rel=1e-6)
    assert notes[0].settled and notes[0].is_corporate_action
    assert new.cash == pytest.approx(state.cash - 12 * (87.82 - 11.90))


# --- which session the reference is read on ------------------------------------


def test_the_reference_is_read_on_the_fill_session_not_the_entry_date() -> None:
    """The cohort and rank engines stamp entry_date as the next session but
    price the fill from the last one. Read at entry_date, a leg bought at 100
    that closed at 102 the next day reads as a 2% adjustment and the day's
    return is settled as a distribution -- on every new leg."""
    hist = _flat(102.0)
    hist[hist.index <= pd.Timestamp("2026-09-04")] = 100.0
    pos = _position(
        ticker="AAA",
        shares=10,
        entry=100.0,
        ref=100.0,
        entry_date="2026-09-08",
        ref_date="2026-09-04",
        last=102.0,
    )
    new, notes = rebase_to_adjusted_history(_state(pos), _panel(AAA=hist))
    assert new.positions == (pos,)
    assert notes == ()


def test_an_off_session_entry_date_reads_the_prior_session() -> None:
    """Legs opened before the entry_date fix were stamped 2026-09-07, Labor
    Day. The fill was the 2026-09-04 close, which is the session to read."""
    hist = _flat(50.0)
    hist[pd.Timestamp("2026-09-04")] = 40.0
    panel = pd.DataFrame({"AAA": hist.drop(pd.Timestamp("2026-09-07"))})
    pos = _position(
        ticker="AAA", shares=1, entry=40.0, ref=40.0, entry_date="2026-09-07", last=50.0
    )
    new, notes = rebase_to_adjusted_history(_state(pos), panel)
    assert new.positions == (pos,)
    assert notes == ()


def test_no_history_for_the_reference_session_leaves_the_leg_alone() -> None:
    hist = _flat(12.0)
    hist[hist.index <= pd.Timestamp("2026-09-10")] = np.nan
    pos = _position()
    new, notes = rebase_to_adjusted_history(
        _state(pos), _panel(CTVA=hist), confirmed=CTVA_IS_A_DISTRIBUTION
    )
    assert new.positions == (pos,)
    assert notes == ()


def test_an_entry_before_the_download_window_is_left_alone() -> None:
    pos = _position(entry_date="2025-01-03")
    new, notes = _settle_ctva(_state(pos))
    assert new.positions == (pos,)
    assert notes == ()


def test_a_ticker_missing_from_the_download_is_left_alone() -> None:
    pos = _position(ticker="WBD", shares=-38, entry=12.0, ref=12.0, last=30.95)
    new, notes = _settle_ctva(_state(pos))
    assert new.positions == (pos,)
    assert notes == ()


def test_an_unadjusted_history_leaves_the_book_untouched() -> None:
    pos = _position(ticker="AAA", shares=10, entry=100.05, ref=100.0, last=101.0)
    state = _state(pos)
    new, notes = rebase_to_adjusted_history(state, _panel(AAA=_flat(100.0)))
    assert new.positions == state.positions
    assert notes == ()


# --- persistence and the engines that create legs ------------------------------


def test_the_reference_survives_a_save_and_a_legacy_file_still_loads(tmp_path) -> None:
    path = tmp_path / "state.json"
    save_state(_state(_position(ref=87.82, ref_date="2026-09-04")), path)
    loaded = load_state(path).positions[0]
    assert (loaded.entry_ref_price, loaded.entry_ref_date) == (87.82, "2026-09-04")

    raw = json.loads(path.read_text())
    for p in raw["positions"]:
        del p["entry_ref_price"], p["entry_ref_date"]
    path.write_text(json.dumps(raw))
    loaded = load_state(path).positions[0]
    assert (loaded.entry_ref_price, loaded.entry_ref_date) == (0.0, "")


def test_a_new_long_short_leg_records_its_unslipped_reference() -> None:
    from stock_predictor.portfolio import generate_orders_long_short

    tickers = [f"T{i:02d}" for i in range(20)]
    picks = [{"ticker": t, "prob": float(20 - i)} for i, t in enumerate(tickers)]
    prices = {t: 100.0 for t in tickers}
    dates = pd.bdate_range("2024-01-01", periods=200)
    _, st = generate_orders_long_short(
        init_state(),
        picks,
        prices,
        decile=0.1,
        long_weight=0.5,
        short_weight=0.5,
        rebalance_every=63,
        slippage_bps=10,
        as_of=str(dates[-1].date()),
        trading_dates=dates,
        min_names_per_side=1,
    )
    assert st.positions, "no book opened"
    for p in st.positions:
        assert p.entry_ref_price == 100.0
        assert p.entry_ref_date == str(dates[-1].date())
        assert p.entry_price != 100.0, "slippage should sit on entry_price"


def test_a_new_cohort_leg_records_the_session_it_was_priced_on() -> None:
    """entry_date is the next session; the reference is the last one."""
    from stock_predictor.portfolio import generate_orders

    tickers = [f"T{i:02d}" for i in range(20)]
    picks = [{"ticker": t, "prob": 0.9 - i * 0.01} for i, t in enumerate(tickers)]
    prices = {t: 100.0 for t in tickers}
    dates = pd.bdate_range("2024-01-01", periods=200)
    as_of = str(dates[-1].date())
    _, st = generate_orders(
        init_state(),
        picks,
        prices,
        top_n=5,
        max_cohorts=2,
        holding_days=10,
        slippage_bps=10,
        as_of=as_of,
        trading_dates=dates,
        force=True,
    )
    assert st.positions, "no cohort opened"
    for p in st.positions:
        assert p.entry_ref_date == as_of
        assert p.entry_date > as_of, "entry_date is the next session"
        assert p.entry_ref_price == 100.0
