"""A print no market produced must not become a label or a feature.

The vendor serves some delisted symbols with two unrelated series interleaved:
Titanium Metals alternates between about $2.70 and about $12,000 from one
session to the next, Cooper Industries moves 34,000-fold in a day. On the
2010-2026 baseline eleven symbols carry such prints. The point-in-time filter
keeps most of them out, because most fall after the company left the index --
but about 780 rows of BMC, CBE, CPWR, MEE, MIL and TIE fall inside their
membership and reached the training panel, with forward returns up to 9,499x
and a 21-day volatility of 1,577.

Once a symbol has printed something impossible there is no telling which of
its later prints are the real ones, so it is not trusted from that session on.
The rule looks only backwards: a print is judged against the one before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.pit import MAX_SESSION_GAIN, quarantine_corrupt_prices

DATES = pd.bdate_range("2012-01-02", periods=60)


def _panel(**overrides: dict[int, float]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for name in ("GOOD", "BAD"):
        steps = rng.normal(0.0, 0.01, len(DATES))
        cols[name] = 50.0 * np.exp(np.cumsum(steps))
    frame = pd.DataFrame(cols, index=DATES)
    for name, edits in overrides.items():
        for row, value in edits.items():
            frame.iloc[row, frame.columns.get_loc(name)] = value
    return frame


def test_a_clean_panel_is_returned_untouched() -> None:
    prices = _panel()
    out = quarantine_corrupt_prices(prices)
    pd.testing.assert_frame_equal(out.prices, prices)
    assert out.quarantined == {}


def test_a_symbol_is_not_trusted_after_an_impossible_print() -> None:
    """TIE's shape: an ordinary price, then thousands of times that, then back."""
    prices = _panel(BAD={30: 12_000.0, 32: 12_400.0})
    out = quarantine_corrupt_prices(prices)
    assert out.quarantined == {"BAD": DATES[30]}
    assert out.prices["BAD"].iloc[:30].equals(prices["BAD"].iloc[:30])
    assert out.prices["BAD"].iloc[30:].isna().all()
    assert out.prices["GOOD"].equals(prices["GOOD"])


def test_the_rule_uses_only_the_past() -> None:
    """Whatever the symbol prints afterwards, the sessions before the first
    impossible print are exactly what they were."""
    a = quarantine_corrupt_prices(_panel(BAD={30: 12_000.0}))
    b = quarantine_corrupt_prices(_panel(BAD={30: 12_000.0, 31: 1.0, 40: 9e6}))
    pd.testing.assert_series_equal(a.prices["BAD"], b.prices["BAD"])


def test_a_violent_but_real_move_is_kept() -> None:
    """GME's best session in January 2021 was +135%. A squeeze is a market."""
    prices = _panel(BAD={30: 50.0 * 2.35})
    for row in range(31, len(DATES)):
        prices.iloc[row, prices.columns.get_loc("BAD")] = 50.0 * 2.35
    assert quarantine_corrupt_prices(prices).quarantined == {}


def test_a_collapse_is_not_a_corrupt_print() -> None:
    """SIVB resumed trading 99% lower. Blanking that would hide the one move a
    short book most needs to see."""
    prices = _panel()
    prices.iloc[30:, prices.columns.get_loc("BAD")] = 0.4
    out = quarantine_corrupt_prices(prices)
    assert out.quarantined == {}
    assert out.prices["BAD"].iloc[-1] == 0.4


def test_the_gain_is_measured_between_prints_not_between_sessions() -> None:
    """A gap in the middle must not hide the jump across it."""
    prices = _panel(BAD={29: np.nan, 30: 12_000.0})
    assert quarantine_corrupt_prices(prices).quarantined == {"BAD": DATES[30]}


def test_volume_is_blanked_with_the_prices() -> None:
    prices = _panel(BAD={30: 12_000.0})
    volume = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    out = quarantine_corrupt_prices(prices, volume=volume)
    assert out.volume["BAD"].iloc[30:].isna().all()
    assert out.volume["BAD"].iloc[:30].notna().all()
    assert out.volume["GOOD"].notna().all()


def test_the_threshold_sits_between_real_sessions_and_corrupt_ones() -> None:
    """Above GameStop's +135%; below Compuware's threefold alternation."""
    assert 1.35 < MAX_SESSION_GAIN <= 2.0


def test_a_threefold_alternation_is_caught() -> None:
    """CPWR's shape: the series flips between two levels a few multiples apart."""
    prices = _panel(BAD={30: 160.0, 31: 50.0, 32: 170.0})
    assert quarantine_corrupt_prices(prices).quarantined == {"BAD": DATES[30]}
