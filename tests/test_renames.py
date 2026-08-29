"""A renamed company is the same company.

The membership stints name tickers, and prices are served under whatever ticker
a company trades as *today*. When a company renames, the old symbol is retired:
Yahoo and Tiingo serve nothing for ANTM, and the whole 2002-2022 history of
Anthem sits under ELV instead. The stint for ANTM therefore pointed at an empty
column, its rows dropped out of the panel, and the company vanished from the
cross-section for twenty years of its membership.

That looked like survivorship — a departed name with no data — but it is not.
The company never departed and the data is not missing; the two are just filed
under different symbols. Twelve of the forty-five names the baseline gate
tolerated as "unavailable upstream" are this, and their successors were already
in the panel with complete history.

Every entry here is validated against prices before it is trusted, because the
plausible-sounding ones are not all real: CBS -> PARA and RX -> IQV look like
renames and are not (a merger and a re-IPO), and their successors carry **zero**
sessions over the predecessor's stint. Guessing would have silently attached
Paramount's history to CBS's membership.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.renames import (
    EFFECTIVE,
    RENAMES,
    TICKER_RENAMES,
    canonical,
    canonicalize_stints,
    rename_coverage,
)


def _stints(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": t, "start_date": pd.Timestamp(s),
          "end_date": pd.NaT if e is None else pd.Timestamp(e)} for t, s, e in rows]
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_a_known_rename_resolves_to_the_current_symbol() -> None:
    assert canonical("ANTM") == "ELV"
    assert canonical("COG") == "CTRA"


def test_an_unknown_ticker_is_left_alone() -> None:
    assert canonical("AAPL") == "AAPL"


def test_a_chain_resolves_to_the_end() -> None:
    """A company can rename twice; the map must not stop halfway."""
    chain = {"AAA": "BBB", "BBB": "CCC"}
    assert canonical("AAA", mapping=chain) == "CCC"


def test_a_cycle_does_not_hang() -> None:
    with pytest.raises(ValueError, match="cycle"):
        canonical("AAA", mapping={"AAA": "BBB", "BBB": "AAA"})


def test_the_map_has_no_self_references() -> None:
    assert not [k for k, v in TICKER_RENAMES.items() if k == v]


def test_every_entry_resolves_to_a_terminal_symbol() -> None:
    for old in TICKER_RENAMES:
        assert canonical(old) not in TICKER_RENAMES


# ---------------------------------------------------------------------------
# Stints
# ---------------------------------------------------------------------------


def test_a_renamed_stint_is_filed_under_the_successor() -> None:
    out = canonicalize_stints(_stints([("ANTM", "2002-07-25", "2022-06-28")]))
    assert list(out["ticker"]) == ["ELV"]


def test_the_two_halves_of_one_membership_become_one_stint() -> None:
    """Anthem's membership did not lapse when it became Elevance."""
    out = canonicalize_stints(_stints([
        ("ANTM", "2002-07-25", "2022-06-28"),
        ("ELV", "2022-06-28", None),
    ]))
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "ELV"
    assert out.iloc[0]["start_date"] == pd.Timestamp("2002-07-25")
    assert pd.isna(out.iloc[0]["end_date"])


def test_a_genuine_rejoin_stays_two_stints() -> None:
    """Leaving the index and rejoining years later is not a rename, and
    merging them would invent membership the company never had."""
    out = canonicalize_stints(_stints([
        ("AAPL", "2000-01-01", "2003-01-01"),
        ("AAPL", "2010-01-01", None),
    ]))
    assert len(out) == 2


def test_unrelated_tickers_are_untouched() -> None:
    rows = [("AAPL", "2010-01-01", None), ("MSFT", "2010-01-01", None)]
    out = canonicalize_stints(_stints(rows))
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    assert len(out) == 2


def test_an_empty_frame_survives() -> None:
    out = canonicalize_stints(_stints([]))
    assert out.empty


# ---------------------------------------------------------------------------
# Validation against prices
# ---------------------------------------------------------------------------


def _prices(cols: dict[str, tuple[str, str]]) -> pd.DataFrame:
    idx = pd.bdate_range("2010-01-01", "2026-08-28")
    out = pd.DataFrame(index=idx)
    for t, (s, e) in cols.items():
        col = pd.Series(float("nan"), index=idx)
        col.loc[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))] = 100.0
        out[t] = col
    return out


def test_coverage_is_reported_for_a_real_rename() -> None:
    stints = _stints([("ANTM", "2012-01-01", "2022-06-28")])
    prices = _prices({"ELV": ("2010-01-01", "2026-08-28")})
    cov = rename_coverage(stints, prices)
    assert cov["ANTM"]["successor"] == "ELV"
    assert cov["ANTM"]["coverage"] == pytest.approx(1.0)


def test_a_successor_without_the_history_scores_zero() -> None:
    """The check that rejected CBS -> PARA and RX -> IQV."""
    stints = _stints([("ANTM", "2012-01-01", "2018-01-01")])
    prices = _prices({"ELV": ("2022-06-28", "2026-08-28")})
    cov = rename_coverage(stints, prices)
    assert cov["ANTM"]["coverage"] == pytest.approx(0.0)


def test_a_missing_successor_is_reported_not_crashed() -> None:
    stints = _stints([("ANTM", "2012-01-01", "2018-01-01")])
    cov = rename_coverage(stints, _prices({"AAPL": ("2010-01-01", "2026-08-28")}))
    assert cov["ANTM"]["successor"] == "ELV"
    assert cov["ANTM"]["coverage"] == 0.0


# ---------------------------------------------------------------------------
# The alias is recorded, not applied invisibly (specs.md:157)
# ---------------------------------------------------------------------------


def test_the_original_symbol_survives_canonicalisation() -> None:
    """Rewriting the ticker and dropping the original is the invisible
    application the spec forbids: downstream sees ELV with no trace it came
    from ANTM, so the substitution cannot be audited from its own output."""
    out = canonicalize_stints(_stints([("ANTM", "2002-07-25", "2022-06-28")]))
    assert out.iloc[0]["alias"] == "ANTM"


def test_an_untouched_ticker_records_no_alias() -> None:
    out = canonicalize_stints(_stints([("AAPL", "2010-01-01", None)]))
    assert out.iloc[0]["alias"] == ""


def test_a_merged_membership_keeps_every_symbol_it_traded_under() -> None:
    out = canonicalize_stints(_stints([
        ("ANTM", "2002-07-25", "2022-06-28"),
        ("ELV", "2022-06-28", None),
    ]))
    assert len(out) == 1
    assert out.iloc[0]["alias"] == "ANTM"


# ---------------------------------------------------------------------------
# Effective dates, and the one falsifier prices can supply
# ---------------------------------------------------------------------------


def test_every_rename_carries_an_effective_date_and_a_note() -> None:
    for r in RENAMES:
        assert pd.notna(pd.Timestamp(r.effective)), r.old
        assert r.note.strip(), r.old


def test_effective_dates_are_exposed_for_every_entry() -> None:
    assert set(EFFECTIVE) == set(TICKER_RENAMES)


def test_concurrent_trading_falsifies_a_rename() -> None:
    """One issuer cannot trade under two symbols at once. This is the only
    real falsifier available from prices, and coverage alone cannot supply it:
    a successor with long history satisfies coverage regardless."""
    stints = _stints([("ANTM", "2012-01-01", "2022-06-28")])
    prices = _prices({"ELV": ("2010-01-01", "2026-08-28"),
                      "ANTM": ("2010-01-01", "2026-08-28")})
    cov = rename_coverage(stints, prices)
    assert cov["ANTM"]["coverage"] == pytest.approx(1.0), "coverage is fooled"
    assert cov["ANTM"]["concurrent_sessions"] > 0, "concurrency is not"


def test_a_clean_rename_shows_no_concurrency() -> None:
    stints = _stints([("ANTM", "2012-01-01", "2022-06-28")])
    prices = _prices({"ELV": ("2010-01-01", "2026-08-28"),
                      "ANTM": ("2010-01-01", "2022-06-28")})
    assert rename_coverage(stints, prices)["ANTM"]["concurrent_sessions"] == 0


def test_coverage_reports_the_effective_date() -> None:
    cov = rename_coverage(_stints([("ANTM", "2012-01-01", "2022-06-28")]),
                          _prices({"ELV": ("2010-01-01", "2026-08-28")}))
    assert cov["ANTM"]["effective"] == "2022-06-28"
