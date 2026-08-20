"""The shared execution core.

Backtest, paper and live used to each carry their own copy of the same
decisions — which names, what weights, what fill price, how many shares. The
copies were kept in sync by comments saying "mirrors the backtest", and they
drifted anyway: `--min-prob`, `--rank-offset` and `--min-cross-section` reached
the simulation but never the live path, so a configuration could be measured
and then not traded.

Everything here is pure: a scored cross-section in, intended trades out. The
loop that calls it differs between a simulation and a live run, but the
decision must not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.execution import (
    CostModel,
    SelectionRules,
    eligible_candidates,
    portfolio_weights,
    rank_exits,
    select_targets,
    size_targets,
)


def _day(n: int = 40, *, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)],
        "prob": np.linspace(1.0, 0.0, n),
        "adj_close": [price] * n,
    })


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_rules_reject_nonsense() -> None:
    for bad in (dict(top_n=0), dict(rank_offset=-1), dict(weighting="magic"),
                dict(min_cross_section=0), dict(exit_rank=1, top_n=5)):
        with pytest.raises(ValueError):
            SelectionRules(**bad)


def test_the_cross_section_floor_defaults_to_the_basket_size() -> None:
    assert SelectionRules(top_n=10, rank_offset=5).effective_min_cross_section == 15


def test_an_explicit_cross_section_floor_wins() -> None:
    assert SelectionRules(min_cross_section=99).effective_min_cross_section == 99


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_equal_weights_sum_to_one() -> None:
    w = portfolio_weights(np.array([0.9, 0.5, 0.1]), "equal")
    assert w.tolist() == pytest.approx([1 / 3] * 3)


def test_probability_weights_track_the_scores() -> None:
    w = portfolio_weights(np.array([0.6, 0.4]), "probability")
    assert w.tolist() == pytest.approx([0.6, 0.4])


def test_signed_scores_are_rejected_not_traded() -> None:
    """Normalizing lambdarank output by its sum yields negative weights — an
    implicit short in a long-only book — and unbounded leverage when the
    scores nearly cancel."""
    with pytest.raises(ValueError, match="non-negative"):
        portfolio_weights(np.array([1.0, -0.99, 0.01]), "probability")


def test_degenerate_totals_fall_back_to_equal() -> None:
    assert portfolio_weights(np.array([0.0, 0.0]), "probability").tolist() == [0.5, 0.5]


def test_no_candidates_gives_no_weights() -> None:
    assert len(portfolio_weights(np.array([]), "equal")) == 0


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def test_candidates_come_back_best_first() -> None:
    out = eligible_candidates(_day(), SelectionRules(top_n=5))
    assert [c.ticker for c in out[:3]] == ["T00", "T01", "T02"]


def test_a_thin_cross_section_yields_nothing() -> None:
    """Two names is not a ranking, and 'the top 15' of it is not a selection."""
    assert eligible_candidates(_day(2), SelectionRules(top_n=15)) == []


def test_the_score_floor_removes_weak_names() -> None:
    out = eligible_candidates(_day(40), SelectionRules(top_n=40, exit_rank=40, min_prob=0.5))
    assert all(c.prob >= 0.5 for c in out)
    assert len(out) < 40


def test_the_rank_offset_skips_the_head_of_the_list() -> None:
    out = eligible_candidates(_day(), SelectionRules(top_n=5, rank_offset=3))
    assert [c.ticker for c in out[:2]] == ["T03", "T04"]


def test_the_offset_is_applied_after_the_score_floor() -> None:
    """Otherwise the offset would skip names the floor already removed, and
    the traded band would shift with the threshold."""
    rules = SelectionRules(top_n=5, rank_offset=2, min_prob=0.5)
    out = eligible_candidates(_day(40), rules)
    kept = [t for t, p in zip(_day(40).ticker, _day(40).prob, strict=True) if p >= 0.5]
    assert out[0].ticker == kept[2]


def test_rows_without_a_usable_price_are_dropped() -> None:
    day = _day(20)
    day.loc[0, "adj_close"] = np.nan
    day.loc[1, "adj_close"] = 0.0
    out = eligible_candidates(day, SelectionRules(top_n=20, min_cross_section=1))
    assert {"T00", "T01"}.isdisjoint({c.ticker for c in out})


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_targets_are_capped_at_top_n_and_weights_sum_to_one() -> None:
    out = select_targets(_day(), SelectionRules(top_n=5))
    assert len(out) == 5
    assert sum(t.weight for t in out) == pytest.approx(1.0)


def test_a_shrunken_basket_still_deploys_the_whole_book() -> None:
    """A score floor that leaves three names buys three names at a third
    each, rather than leaving two thirds of the capital idle."""
    out = select_targets(_day(40), SelectionRules(top_n=15, min_prob=0.95))
    assert 0 < len(out) < 15
    assert sum(t.weight for t in out) == pytest.approx(1.0)


def test_a_thin_day_produces_no_targets() -> None:
    assert select_targets(_day(2), SelectionRules(top_n=15)) == []


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_buys_lift_the_offer_and_sells_hit_the_bid() -> None:
    costs = CostModel(slippage_bps=10.0)
    assert costs.fill_price(100.0, 1) == pytest.approx(100.1)
    assert costs.fill_price(100.0, -1) == pytest.approx(99.9)


def test_commission_has_a_per_share_and_a_per_order_leg() -> None:
    costs = CostModel(commission_per_share=0.01, commission_per_order=1.0)
    assert costs.commission(100) == pytest.approx(2.0)


def test_negative_costs_are_rejected() -> None:
    for bad in (dict(commission_per_share=-1.0), dict(commission_per_order=-1.0),
                dict(slippage_bps=-1.0)):
        with pytest.raises(ValueError):
            CostModel(**bad)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


FREE = CostModel(slippage_bps=0.0)


def test_fractional_sizing_spends_the_whole_book() -> None:
    """What a simulation does: no lot rounding, so weights are exact."""
    lots = size_targets(select_targets(_day(), SelectionRules(top_n=4)),
                        10_000.0, FREE, whole_shares=False)
    assert sum(lot.cost for lot in lots) == pytest.approx(10_000.0)
    assert lots[0].shares == pytest.approx(25.0)


def test_whole_share_sizing_never_exceeds_the_budget() -> None:
    """What a live account does: integer lots, and the rounding is downward."""
    lots = size_targets(select_targets(_day(price=333.0), SelectionRules(top_n=4)),
                        10_000.0, FREE, whole_shares=True)
    assert all(float(lot.shares).is_integer() for lot in lots)
    assert sum(lot.cost for lot in lots) <= 10_000.0


def test_commissions_cannot_overdraw_the_account() -> None:
    """Sizing that ignores fees produced a negative balance: $1,000 of cash
    with $100 per order came back at -$200."""
    costs = CostModel(slippage_bps=0.0, commission_per_order=100.0)
    lots = size_targets(select_targets(_day(price=90.0), SelectionRules(top_n=10)),
                        1_000.0, costs, whole_shares=True)
    assert sum(lot.cost for lot in lots) <= 1_000.0


def test_a_lot_smaller_than_one_share_is_skipped() -> None:
    lots = size_targets(select_targets(_day(price=1e6), SelectionRules(top_n=5)),
                        1_000.0, FREE, whole_shares=True)
    assert lots == []


def test_no_capital_buys_nothing() -> None:
    assert size_targets(select_targets(_day(), SelectionRules(top_n=5)),
                        0.0, FREE, whole_shares=True) == []


def test_slippage_raises_the_price_actually_paid() -> None:
    targets = select_targets(_day(), SelectionRules(top_n=1))
    free = size_targets(targets, 10_000.0, FREE, whole_shares=False)[0]
    slipped = size_targets(targets, 10_000.0, CostModel(slippage_bps=50.0),
                           whole_shares=False)[0]
    assert slipped.fill_price > free.fill_price
    assert slipped.shares < free.shares, "the same dollars buy fewer shares"


# ---------------------------------------------------------------------------
# Rank exits
# ---------------------------------------------------------------------------


def test_a_name_that_decays_past_the_exit_rank_is_sold() -> None:
    ranked = [f"T{i:02d}" for i in range(40)]
    assert rank_exits({"T35"}, ranked, exit_rank=30) == {"T35"}


def test_a_name_still_inside_the_exit_rank_is_held() -> None:
    ranked = [f"T{i:02d}" for i in range(40)]
    assert rank_exits({"T05"}, ranked, exit_rank=30) == set()


def test_a_name_that_left_the_universe_is_sold() -> None:
    """Delisted or simply unscored today — either way it cannot be ranked, and
    holding it on an absent signal is how a dead position gets stranded."""
    assert rank_exits({"GONE"}, ["T00", "T01"], exit_rank=30) == {"GONE"}


def test_the_boundary_rank_is_inclusive() -> None:
    ranked = [f"T{i:02d}" for i in range(40)]
    assert rank_exits({"T29"}, ranked, exit_rank=30) == set(), "rank 30 is kept"
    assert rank_exits({"T30"}, ranked, exit_rank=30) == {"T30"}, "rank 31 is not"
