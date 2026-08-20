"""One execution core, shared by the backtest, paper trading and live orders.

Each path used to carry its own copy of the same four decisions — which names
to hold, at what weights, at what fill price, in what share count — kept in
step by comments reading "mirrors the backtest". They drifted anyway:
``--min-prob``, ``--rank-offset`` and ``--min-cross-section`` reached the
simulation and never the live path, so a configuration could be measured and
then not traded. ``_compute_weights`` and ``_long_only_weights`` were the same
algorithm in two files with two different error messages.

Everything here is pure — a scored cross-section in, intended trades out. What
legitimately differs between a simulation and a live account is *not* the
decision:

* **Share granularity.** A backtest may hold 25.4 shares; an account may not.
  That is the ``whole_shares`` argument to :func:`size_targets`, and nothing
  else about sizing changes.
* **The loop.** A simulation walks a price panel; a live run fires once
  against the latest quotes.
* **Where state lives.** A simulation keeps an internal ledger; a live run
  keeps :class:`~stock_predictor.portfolio.PortfolioState` on disk.

Those belong to the callers. The selection does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

WEIGHTINGS = ("equal", "probability")


@dataclass(frozen=True)
class SelectionRules:
    """Which names a strategy holds, independent of how it is executed."""

    top_n: int = 15
    """Names in the book."""
    rank_offset: int = 0
    """Skip this many from the head, trading the band
    ``rank_offset+1 .. rank_offset+top_n`` instead of the top of the list."""
    min_prob: float | None = None
    """Score floor. Baskets shrink and weights renormalize over the survivors;
    a date with no eligible name simply does not trade. Compared against the
    raw score, which is uncalibrated for a classifier and unbounded for a
    lambdarank model, so it is only comparable across runs of one family."""
    min_cross_section: int | None = None
    """Fewest scored names a date must carry before it may open anything.
    ``None`` derives ``rank_offset + top_n`` — a basket cannot be filled from
    fewer names than it holds, and "the top 15" of a two-name date is not a
    selection."""
    weighting: str = "equal"
    exit_rank: int = 30
    """Rank-hold only: a holding is sold once it decays past this."""

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        if self.rank_offset < 0:
            raise ValueError(f"rank_offset must be >= 0, got {self.rank_offset}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}"
            )
        if self.min_cross_section is not None and self.min_cross_section < 1:
            raise ValueError("min_cross_section must be >= 1")
        if self.min_prob is not None and not np.isfinite(self.min_prob):
            raise ValueError(f"min_prob must be finite or None, got {self.min_prob!r}")
        if self.exit_rank < self.top_n:
            raise ValueError(
                f"exit_rank ({self.exit_rank}) must be >= top_n ({self.top_n})"
            )

    @property
    def effective_min_cross_section(self) -> int:
        if self.min_cross_section is not None:
            return self.min_cross_section
        return self.rank_offset + self.top_n


@dataclass(frozen=True)
class CostModel:
    """What a trade costs. Identical in simulation and in production —
    a backtest that charges less than the account does is the whole problem."""

    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    commission_per_order: float = 0.0

    def __post_init__(self) -> None:
        for name in ("slippage_bps", "commission_per_share", "commission_per_order"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    def fill_price(self, price: float, side: int) -> float:
        """*side*: +1 buys and lifts the offer, -1 sells and hits the bid."""
        return float(price) * (1 + side * self.slippage_bps / 10_000)

    def commission(self, shares: float) -> float:
        return abs(shares) * self.commission_per_share + self.commission_per_order


@dataclass(frozen=True)
class Candidate:
    ticker: str
    prob: float
    price: float


@dataclass(frozen=True)
class Target:
    """A name to hold, and the share of capital it should take."""

    ticker: str
    weight: float
    price: float


@dataclass(frozen=True)
class SizedLot:
    """A target turned into a tradable quantity at a fill price."""

    ticker: str
    shares: float
    price: float
    """Raw price, before slippage — what the position is marked at."""
    fill_price: float
    commission: float

    @property
    def cost(self) -> float:
        """Cash out of the door, fees included."""
        return self.shares * self.fill_price + self.commission


def portfolio_weights(probs: np.ndarray, weighting: str) -> np.ndarray:
    """Long-only weights summing to 1.

    ``probability`` normalizes scores by their sum, which is only meaningful
    for non-negative scores. Raw lambdarank output straddles zero:
    ``p / sum(p)`` then yields negative weights — an implicit short in a
    long-only book — and unbounded leverage when the scores nearly cancel, so
    ``[1.0, -0.99, 0.01]`` became ``[50.0, -49.5, 0.5]``. Reject those inputs
    rather than trade them.
    """
    n = len(probs)
    if n == 0:
        return np.zeros(0, dtype=float)
    if weighting != "probability":
        return np.ones(n, dtype=float) / n
    if np.any(probs < 0):
        raise ValueError(
            "weighting='probability' requires non-negative scores, got "
            f"min={float(np.min(probs)):.6g}. Raw lambdarank scores are not "
            "probabilities — use weighting='equal' with --rank-objective "
            "models, or a classifier whose scores are in [0, 1]."
        )
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        return np.ones(n, dtype=float) / n
    return probs / total


def eligible_candidates(
    scored_day: pd.DataFrame,
    rules: SelectionRules,
    *,
    score_col: str = "prob",
    ticker_col: str = "ticker",
    price_col: str = "adj_close",
) -> list[Candidate]:
    """Buyable names for one date, best first, after every selection rule.

    Order matters. The cross-section floor is measured on the raw panel: a
    score floor deliberately shrinking the basket is intended, whereas a date
    with nothing to rank is not. The rank offset is applied *after* the score
    floor, so the traded band does not shift as the threshold moves.
    """
    if len(scored_day) < rules.effective_min_cross_section:
        return []

    day = scored_day
    if rules.min_prob is not None:
        day = day[day[score_col] >= rules.min_prob]
    if price_col in day.columns:
        px = pd.to_numeric(day[price_col], errors="coerce")
        day = day[px.notna() & (px > 0)]

    day = day.sort_values(score_col, ascending=False, kind="stable")
    if rules.rank_offset:
        day = day.iloc[rules.rank_offset:]
    return [
        Candidate(
            ticker=str(row[ticker_col]),
            prob=float(row[score_col]),
            price=float(row[price_col]) if price_col in day.columns else float("nan"),
        )
        for _, row in day.iterrows()
    ]


def select_targets(
    scored_day: pd.DataFrame,
    rules: SelectionRules,
    **kwargs,
) -> list[Target]:
    """The basket for one date: up to ``top_n`` names with weights summing to 1.

    A basket shortened by the score floor renormalizes over its survivors, so
    a three-name day is fully invested in three names rather than leaving the
    rest of the book in cash.
    """
    picks = eligible_candidates(scored_day, rules, **kwargs)[: rules.top_n]
    if not picks:
        return []
    weights = portfolio_weights(
        np.array([c.prob for c in picks], dtype=float), rules.weighting,
    )
    return [
        Target(ticker=c.ticker, weight=float(w), price=c.price)
        for c, w in zip(picks, weights, strict=True)
    ]


def size_targets(
    targets: list[Target],
    capital: float,
    costs: CostModel,
    *,
    whole_shares: bool,
) -> list[SizedLot]:
    """Turn weights into quantities against a cash budget.

    ``whole_shares`` is the only thing that separates a simulated fill from a
    real one: an account buys integer lots and rounds down, a simulation does
    not. Both respect the same budget, and both charge the same fees — sizing
    that ignored commissions overdrew the account, turning $1,000 of cash with
    $100 per order into a balance of -$200.
    """
    if capital <= 0:
        return []
    lots: list[SizedLot] = []
    spent = 0.0
    for t in targets:
        if not np.isfinite(t.price) or t.price <= 0:
            continue
        fill = costs.fill_price(t.price, 1)
        budget_left = capital - spent
        shares = t.weight * capital / fill
        if whole_shares:
            shares = floor(shares)
            if shares < 1:
                continue
        elif shares <= 0:
            continue

        commission = costs.commission(shares)
        need = shares * fill + commission
        if need > budget_left:
            affordable = (budget_left - costs.commission_per_order) / (
                fill + costs.commission_per_share
            )
            shares = floor(affordable) if whole_shares else affordable
            if shares < (1 if whole_shares else 0) or shares <= 0:
                continue
            commission = costs.commission(shares)
            need = shares * fill + commission
            if need > budget_left:
                continue

        lots.append(SizedLot(ticker=t.ticker, shares=float(shares), price=t.price,
                             fill_price=fill, commission=commission))
        spent += need
    return lots


def rank_exits(
    held: set[str] | frozenset[str],
    ranked_tickers: list[str],
    exit_rank: int,
) -> set[str]:
    """Holdings to close because their rank decayed, or they left the universe.

    A name absent from today's ranking cannot be ranked at all — delisted, or
    simply unscored — and treating that as "still fine to hold" is how a dead
    position gets stranded in the book.
    """
    rank_of = {t: i + 1 for i, t in enumerate(ranked_tickers)}
    return {t for t in held if rank_of.get(t, exit_rank + 1) > exit_rank}
