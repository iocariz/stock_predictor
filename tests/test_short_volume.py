"""Short-sale volume as a feature family, not another OHLCV derivative.

Every feature the model currently sees is a price, volume or macro derivative
of one panel. Six research directions have come back negative or neutral, and
validation peaks at 2-10 trees under every configuration tried -- which says
the signal is not in the current inputs rather than that the tuner is wrong.

FINRA publishes consolidated daily short-sale volume on a public CDN, one
dated file per session:

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
    20261029|A|511423.702645|1090|1000128.599576|B,Q,N

It is genuinely new information: it decomposes a session's volume into short
and non-short, which no amount of OHLCV arithmetic recovers. And it is
point-in-time by construction -- a file dated D contains only what traded on
D, and a name that stops trading simply stops appearing, so there is no
survivorship repair to get wrong.

The archive starts 2019, which bounds what any experiment on it can claim.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.short_volume import (
    DEFAULT_PUBLICATION_LAG,
    add_short_volume_features,
    parse_short_volume_file,
    short_volume_panel,
)

RAW = (
    "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
    "20260105|AAA|400.0|0|1000.0|B,Q,N\n"
    "20260105|BBB|750.0|50|1000.0|B,Q,N\n"
    "20260105|CCC|0.0|0|0.0|B,Q,N\n"          # no volume: ratio undefined
    "20260105|DDD|100.0|0|200.0|B,Q,N\n"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_short_ratio_is_the_short_share_of_volume() -> None:
    df = parse_short_volume_file(RAW)
    r = dict(zip(df["ticker"], df["short_ratio"]))
    assert r["AAA"] == pytest.approx(0.40)
    assert r["BBB"] == pytest.approx(0.75)
    assert r["DDD"] == pytest.approx(0.50)


def test_a_zero_volume_session_has_no_ratio() -> None:
    """0/0 is not 0. A name that did not trade has no short share, and
    recording one as zero would read as 'nobody shorted it'."""
    df = parse_short_volume_file(RAW)
    ccc = df[df["ticker"] == "CCC"]
    assert len(ccc) == 1
    assert pd.isna(ccc["short_ratio"].iloc[0])


def test_the_date_is_parsed_from_the_row_not_the_filename() -> None:
    """The row carries the session it describes; trusting a filename is how a
    mislabelled file becomes silent lookahead."""
    df = parse_short_volume_file(RAW)
    assert df["date"].unique().tolist() == [pd.Timestamp("2026-01-05")]


def test_a_header_only_file_is_empty_not_an_error() -> None:
    """Holidays return a file with no rows. A run must not die on one."""
    df = parse_short_volume_file(
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n")
    assert df.empty
    assert list(df.columns) == ["date", "ticker", "short_ratio",
                                "short_volume", "total_volume"]


def test_junk_is_rejected_rather_than_silently_empty() -> None:
    """An HTML error page served with status 200 must not parse to 'no shorts
    today', which would be indistinguishable from a real quiet session."""
    with pytest.raises(ValueError, match="header"):
        parse_short_volume_file("<html><body>403 Forbidden</body></html>")


# ---------------------------------------------------------------------------
# Point-in-time discipline
# ---------------------------------------------------------------------------


def _panel() -> pd.DataFrame:
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=10)):
        for t, base in (("AAA", 0.30), ("BBB", 0.60)):
            rows.append({"date": d, "ticker": t,
                         "short_ratio": base + 0.01 * i,
                         "short_volume": 100.0, "total_volume": 1000.0})
    return pd.DataFrame(rows)


def test_the_feature_is_lagged_by_a_publication_delay() -> None:
    """A file dated D is published after D's close. The signal already uses D's
    close and fills at D+1, so D's short volume is known in time -- but a
    vendor that publishes late would turn that into lookahead for free. One
    session of lag is cheap insurance on a slow-moving feature."""
    assert DEFAULT_PUBLICATION_LAG >= 1
    panel = _panel()
    feat = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=10),
                         "ticker": "AAA"})
    out = add_short_volume_features(feat, panel)
    merged = out.set_index("date")["short_ratio"]
    # The value on session D is the one reported for D-1.
    assert merged.loc[pd.Timestamp("2026-01-06")] == pytest.approx(0.30)
    assert pd.isna(merged.loc[pd.Timestamp("2026-01-05")]), (
        "the first session used same-day data")


def test_a_ticker_absent_from_the_panel_gets_nan_not_zero() -> None:
    """A name FINRA does not list is unmeasured, not unshorted. LightGBM
    handles NaN natively; a zero would be a confident wrong answer."""
    feat = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=4),
                         "ticker": "ZZZ"})
    out = add_short_volume_features(feat, _panel())
    assert out["short_ratio"].isna().all()


def test_features_do_not_reach_across_tickers() -> None:
    """A rolling mean computed without grouping bleeds one name into the next,
    which is the bug class this project has paid for repeatedly."""
    feat = pd.DataFrame(
        {"date": list(pd.bdate_range("2026-01-05", periods=10)) * 2,
         "ticker": ["AAA"] * 10 + ["BBB"] * 10}
    )
    out = add_short_volume_features(feat, _panel())
    a = out[out["ticker"] == "AAA"]["short_ratio"].dropna()
    b = out[out["ticker"] == "BBB"]["short_ratio"].dropna()
    assert a.max() < 0.45, f"AAA picked up BBB's level: {a.max()}"
    assert b.min() > 0.55, f"BBB picked up AAA's level: {b.min()}"


def test_the_row_count_is_unchanged() -> None:
    """A feature merge that drops rows silently shrinks the training set."""
    feat = pd.DataFrame(
        {"date": list(pd.bdate_range("2026-01-05", periods=10)) * 2,
         "ticker": ["AAA"] * 10 + ["ZZZ"] * 10}
    )
    out = add_short_volume_features(feat, _panel())
    assert len(out) == len(feat)


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------


def test_the_expected_feature_columns_are_added() -> None:
    feat = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=10),
                         "ticker": "AAA"})
    out = add_short_volume_features(feat, _panel())
    for col in ("short_ratio", "short_ratio_5d", "short_ratio_chg_5d",
                "short_ratio_xs_z"):
        assert col in out.columns, f"missing {col}"


def test_the_cross_sectional_z_is_computed_per_date() -> None:
    """Ranking against the whole history instead of the session's peers is a
    look across time; the model trades a cross-section."""
    feat = pd.DataFrame(
        {"date": list(pd.bdate_range("2026-01-05", periods=10)) * 2,
         "ticker": ["AAA"] * 10 + ["BBB"] * 10}
    )
    out = add_short_volume_features(feat, _panel()).dropna(subset=["short_ratio_xs_z"])
    # Two names per date, so z-scores are symmetric about zero on every date.
    by_date = out.groupby("date")["short_ratio_xs_z"].sum()
    assert by_date.abs().max() == pytest.approx(0.0, abs=1e-9)


def test_an_empty_panel_leaves_the_features_missing_not_absent() -> None:
    """Running without the data must not change the model's column set, or a
    model trained with it cannot score without it."""
    feat = pd.DataFrame({"date": pd.bdate_range("2026-01-05", periods=4),
                         "ticker": "AAA"})
    out = add_short_volume_features(feat, pd.DataFrame())
    assert "short_ratio" in out.columns
    assert out["short_ratio"].isna().all()


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def test_the_panel_keeps_only_the_requested_tickers() -> None:
    """The files carry ~11,000 symbols; the universe is ~800. Keeping the rest
    makes every groupby in the feature path an order of magnitude slower."""
    files = {pd.Timestamp("2026-01-05"): RAW}
    panel = short_volume_panel(files, tickers={"AAA", "DDD"})
    assert set(panel["ticker"]) == {"AAA", "DDD"}


def test_the_panel_is_sorted_and_deduplicated() -> None:
    """A session published twice must not double-count into the rolling mean."""
    files = {pd.Timestamp("2026-01-05"): RAW,
             pd.Timestamp("2026-01-06"): RAW.replace("20260105", "20260106")}
    panel = short_volume_panel({**files, pd.Timestamp("2026-01-07"): RAW},
                               tickers={"AAA"})
    assert panel["date"].is_monotonic_increasing
    assert not panel.duplicated(subset=["date", "ticker"]).any()
