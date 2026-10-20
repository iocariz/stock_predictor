"""Dollar-neutral is not sector-neutral either.

``target_book`` takes the top and bottom decile of one global ranking. The
total nets to zero and nothing underneath does. Measured on the live book of
2026-10-02, 94 positions at $96,749 gross:

    Information Technology   +$30,548   +31.6% of gross
    Consumer Discretionary    +$7,003    +7.2%
    Health Care               +$3,841    +4.0%
    Industrials               -$2,234    -2.3%
    Communication Services    -$3,376    -3.5%
    Materials                 -$6,188    -6.4%
    Real Estate               -$8,124    -8.4%
    Financials               -$22,875   -23.6%

Sum of absolute net sector exposure: **92% of gross**. That is a long-Tech
short-Financials trade wearing a stock-selection label, and it follows from
the model ranking volatility positively -- high-vol tech sorts long, while
rate-sensitive financials and REITs sort short.

It matters because the alpha t-stat divides by *residual* risk. Beta is
already controlled for in CAPM alpha, which is why hedging it moved total vol
(11.2% -> 5.5%) and left residual risk slightly worse (7.1% -> 7.6%) -- all
cost, no significance. Sector tilt is uncompensated residual risk, so it sits
in the denominator the t-stat actually uses.

Per-sector balancing: take an equal number of longs and shorts inside each
sector, so every sector nets to zero by construction and the book does too.
A sector too thin to field both sides is skipped rather than half-traded,
because a one-sided sector is the exposure this exists to remove.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.long_short import LongShortConfig, target_book

CAPITAL = 100_000.0


def _scored(spec: dict[str, int], *, with_sector: bool = True) -> pd.DataFrame:
    """One session's ranking. *spec* maps sector -> how many names it holds.

    Scores descend globally in the order sectors are listed, so without
    balancing the first sector sweeps the long side and the last sweeps the
    short side -- the real failure, in miniature.
    """
    rows, score = [], 1000.0
    for sector, n in spec.items():
        for i in range(n):
            rows.append({"ticker": f"{sector[:3].upper()}{i:03d}",
                         "prob": score, "sector": sector})
            score -= 1.0
    df = pd.DataFrame(rows)
    return df if with_sector else df.drop(columns=["sector"])


def _by_sector(book: dict[str, float], scored: pd.DataFrame) -> dict[str, float]:
    sect = dict(zip(scored["ticker"], scored["sector"]))
    out: dict[str, float] = {}
    for t, dollars in book.items():
        out[sect[t]] = out.get(sect[t], 0.0) + dollars
    return out


def _cfg(**kw) -> LongShortConfig:
    base = dict(decile=0.2, long_weight=0.5, short_weight=0.5,
                min_names_per_side=2, sector_neutral=True)
    base.update(kw)
    return LongShortConfig(**base)


# ---------------------------------------------------------------------------
# The exposure being removed
# ---------------------------------------------------------------------------


def test_the_unbalanced_book_concentrates_in_sectors() -> None:
    """The defect, reproduced. Without balancing, one sector takes the whole
    long side and another the whole short side."""
    scored = _scored({"Tech": 10, "Health": 10, "Financials": 10})
    book = target_book(scored, _cfg(sector_neutral=False), CAPITAL)
    nets = _by_sector(book, scored)
    gross = sum(abs(v) for v in book.values())
    tilt = sum(abs(v) for v in nets.values()) / gross
    assert tilt > 0.9, f"expected a concentrated book, got {nets}"


def test_every_sector_nets_to_zero() -> None:
    scored = _scored({"Tech": 10, "Health": 10, "Financials": 10})
    book = target_book(scored, _cfg(), CAPITAL)
    assert book, "no book built"
    for sector, net in _by_sector(book, scored).items():
        assert net == pytest.approx(0.0, abs=1e-6), f"{sector} nets {net}"


def test_the_book_is_still_dollar_neutral() -> None:
    """Sector neutrality must not come at the cost of the property the engine
    already had."""
    scored = _scored({"Tech": 10, "Health": 10, "Financials": 10})
    book = target_book(scored, _cfg(), CAPITAL)
    assert sum(book.values()) == pytest.approx(0.0, abs=1e-6)


def test_gross_exposure_still_matches_the_configured_weights() -> None:
    scored = _scored({"Tech": 10, "Health": 10, "Financials": 10})
    book = target_book(scored, _cfg(), CAPITAL)
    gross = sum(abs(v) for v in book.values())
    assert gross == pytest.approx(CAPITAL * 1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# It still has to pick on the signal
# ---------------------------------------------------------------------------


def test_within_a_sector_the_best_go_long_and_the_worst_short() -> None:
    """Neutrality is not an excuse to stop reading the ranking."""
    scored = _scored({"Tech": 10, "Health": 10})
    book = target_book(scored, _cfg(), CAPITAL)
    tech = {t: v for t, v in book.items() if t.startswith("TEC")}
    longs = sorted(t for t, v in tech.items() if v > 0)
    shorts = sorted(t for t, v in tech.items() if v < 0)
    assert longs and shorts
    assert max(longs) < min(shorts), (
        f"not ranked within sector: long {longs}, short {shorts}")


# ---------------------------------------------------------------------------
# Thin sectors
# ---------------------------------------------------------------------------


def test_a_sector_too_thin_for_both_sides_is_skipped() -> None:
    """Utilities and Energy hold two names each in the live book. A sector
    that cannot field a long and a short is left out entirely -- trading one
    side of it would reintroduce exactly the tilt this removes."""
    scored = _scored({"Tech": 10, "Health": 10, "Utilities": 1})
    book = target_book(scored, _cfg(), CAPITAL)
    assert not [t for t in book if t.startswith("UTI")], (
        "a one-name sector was traded on one side only")


def test_skipping_a_thin_sector_does_not_shrink_gross() -> None:
    """Capital skipped in a thin sector has to go somewhere, or the book
    quietly runs under its configured exposure."""
    scored = _scored({"Tech": 10, "Health": 10, "Utilities": 1})
    book = target_book(scored, _cfg(), CAPITAL)
    gross = sum(abs(v) for v in book.values())
    assert gross == pytest.approx(CAPITAL * 1.0, rel=1e-6)


def test_no_tradable_sector_builds_nothing() -> None:
    """Same contract the unbalanced path has: refuse rather than improvise."""
    scored = _scored({"Tech": 1, "Health": 1})
    assert target_book(scored, _cfg(), CAPITAL) == {}


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def test_a_panel_without_sectors_refuses_rather_than_silently_tilting() -> None:
    """The scored panel carries no sector column today. Asking for neutrality
    and silently getting a concentrated book is the failure mode worth
    refusing -- it is how a backtest and its live path drift apart."""
    scored = _scored({"Tech": 10, "Health": 10}, with_sector=False)
    with pytest.raises(ValueError, match="sector"):
        target_book(scored, _cfg(), CAPITAL)


def test_the_default_is_unchanged_behaviour() -> None:
    """sector_neutral defaults off, so every existing caller and every
    recorded result stays reproducible."""
    assert LongShortConfig().sector_neutral is False
    scored = _scored({"Tech": 10, "Health": 10}, with_sector=False)
    book = target_book(scored, LongShortConfig(decile=0.2,
                                               min_names_per_side=2), CAPITAL)
    assert book, "the old path stopped working"
