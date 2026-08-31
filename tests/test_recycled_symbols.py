"""A ticker is not a company.

When a company is acquired or renamed its symbol is retired, and the exchange
is free to hand it to somebody else. The panel then holds prices under a
departed member's symbol that belong to a different issuer entirely:

    APC   Anadarko, acquired 2019-08     prices 2026-02-12 .. 2026-08-27
    FB    Facebook, renamed META 2022-06 prices 2025-06-26 .. 2026-08-27
    Q     Qwest, left the index 2011-04  prices 2025-10-27 ..   (14.6 years later)
    SNDK  SanDisk, acquired 2016         prices 2025-02-13 ..   (re-IPO, new entity)

54 of 347 departed names in the baseline are like this: priced *only* outside
the window they were ever members. They were being counted as survivorship
recoveries, which put the reported coverage 15.6 points above the truth --
91.4% against 75.8%.

The point-in-time filter keeps them out of the *scored* panel, which is why
none reached a trade. But they sit in the execution panel, and
``_resolve_leg_exit`` walks forward looking for the next real quote when an
exit cannot fill. Under the default write-off policy the grace period ends the
walk first; under ``fallback="hold"`` nothing does, and the walk would sell a
2019 holding at an unrelated 2026 company's price.

Distinguishing the two cases is not subtle. A company that leaves the index and
keeps trading has *continuous* prices across the boundary. A recycled symbol has
a dead gap first — years of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.pit import drop_recycled_prices

SESSIONS = pd.bdate_range("2010-01-01", "2026-08-28")


def _stints(rows) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": t, "start_date": pd.Timestamp(s),
         "end_date": pd.NaT if e is None else pd.Timestamp(e)} for t, s, e in rows
    ])


def _prices(blocks: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    out = pd.DataFrame(index=SESSIONS)
    for t, spans in blocks.items():
        col = pd.Series(np.nan, index=SESSIONS)
        for s, e in spans:
            m = (SESSIONS >= pd.Timestamp(s)) & (SESSIONS <= pd.Timestamp(e))
            col.loc[m] = 100.0
        out[t] = col
    return out


def _n(frame: pd.DataFrame, t: str) -> int:
    return int(((frame[t].notna()) & (frame[t] > 0)).sum()) if t in frame else 0


# ---------------------------------------------------------------------------
# The recycled case
# ---------------------------------------------------------------------------


def test_a_symbol_reused_years_later_is_dropped() -> None:
    """APC: Anadarko until 2019, somebody else from 2026."""
    st = _stints([("APC", "2010-01-01", "2019-08-08")])
    px = _prices({"APC": [("2010-01-04", "2019-08-08"), ("2026-02-12", "2026-08-27")]})
    before = _n(px, "APC")
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert "APC" in dropped
    assert _n(out, "APC") < before
    assert out.loc[pd.Timestamp("2026-03-02"), "APC"] != out.loc[pd.Timestamp("2026-03-02"), "APC"]


def test_the_original_companys_prices_are_kept() -> None:
    st = _stints([("APC", "2010-01-01", "2019-08-08")])
    px = _prices({"APC": [("2010-01-04", "2019-08-08"), ("2026-02-12", "2026-08-27")]})
    out, _vol, _ = drop_recycled_prices(px, st)
    assert out.loc[pd.Timestamp("2015-06-01"), "APC"] == pytest.approx(100.0)


def test_a_symbol_with_only_recycled_prices_is_emptied() -> None:
    """Q left in 2011 and its only prices start in 2025."""
    st = _stints([("Q", "2010-01-01", "2011-04-01")])
    px = _prices({"Q": [("2025-10-27", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert "Q" in dropped
    assert _n(out, "Q") == 0


# ---------------------------------------------------------------------------
# What must survive
# ---------------------------------------------------------------------------


def test_a_demoted_company_keeps_trading_and_is_untouched() -> None:
    """Leaving the index is not leaving the exchange. Continuous prices across
    the boundary are the same company and must be kept -- they are what prices
    an exit after the name drops out."""
    st = _stints([("XYZ", "2010-01-01", "2020-06-30")])
    px = _prices({"XYZ": [("2010-01-04", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert dropped == []
    assert _n(out, "XYZ") == _n(px, "XYZ")


def test_a_current_member_is_untouched() -> None:
    st = _stints([("AAPL", "2010-01-01", None)])
    px = _prices({"AAPL": [("2010-01-04", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert dropped == []
    assert _n(out, "AAPL") == _n(px, "AAPL")


def test_a_short_gap_is_a_data_hole_not_a_recycle() -> None:
    """Vendors drop weeks. Only a long dead period means the symbol was reissued."""
    st = _stints([("XYZ", "2010-01-01", "2020-06-30")])
    px = _prices({"XYZ": [("2010-01-04", "2020-06-30"), ("2020-08-15", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert dropped == []
    assert _n(out, "XYZ") == _n(px, "XYZ")


def test_the_gap_threshold_is_configurable() -> None:
    st = _stints([("XYZ", "2010-01-01", "2020-06-30")])
    px = _prices({"XYZ": [("2010-01-04", "2020-06-30"), ("2021-06-01", "2026-08-27")]})
    kept, _kv, _ = drop_recycled_prices(px, st, max_gap_days=1000)
    cut, _cv, dropped = drop_recycled_prices(px, st, max_gap_days=180)
    assert _n(kept, "XYZ") > _n(cut, "XYZ")
    assert dropped == ["XYZ"]


def test_a_ticker_with_no_stint_is_left_alone() -> None:
    """Not every column is an index member; benchmarks live here too."""
    st = _stints([("AAPL", "2010-01-01", None)])
    px = _prices({"SPY": [("2010-01-04", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert dropped == []
    assert _n(out, "SPY") == _n(px, "SPY")


def test_rejoining_the_index_does_not_look_like_recycling() -> None:
    """Two stints with continuous prices is one company that came back."""
    st = _stints([("XYZ", "2010-01-01", "2014-01-01"),
                  ("XYZ", "2018-01-01", None)])
    px = _prices({"XYZ": [("2010-01-04", "2026-08-27")]})
    out, _vol, dropped = drop_recycled_prices(px, st)
    assert dropped == []
    assert _n(out, "XYZ") == _n(px, "XYZ")


def test_an_empty_panel_is_handled() -> None:
    out, _vol, dropped = drop_recycled_prices(pd.DataFrame(), _stints([]))
    assert out.empty and dropped == []
