"""Feature-engineering stage order: time-series → PIT filter → cross-sectional.

Regression tests for the ordering bug where ``filter_panel_to_pit`` ran
*before* the rolling per-ticker features, so lagged windows silently spanned
index-membership gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import (
    add_cross_sectional_features,
    add_timeseries_features,
    build_feature_panel,
    build_labeled_panel,
)

DATES = pd.bdate_range("2024-01-01", periods=120)


class _OfflineProvider:
    """Provider stub so build_feature_panel never touches the network."""

    def download_macro(self, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame({
            "date": DATES,
            "vix": np.linspace(15.0, 25.0, len(DATES)),
            "tnx_yield": np.linspace(3.0, 4.0, len(DATES)),
            "irx_yield": np.linspace(2.0, 2.5, len(DATES)),
        })


def _prices(tickers: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(tickers, index=DATES)


def _volume(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: np.full(len(DATES), 1_000_000.0) for c in cols}, index=DATES,
    )


def _sector_map(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": tickers, "sector": ["Industrials"] * len(tickers)})


def _stints(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [r[0] for r in rows],
        "start_date": pd.to_datetime([r[1] for r in rows]),
        "end_date": pd.to_datetime([r[2] for r in rows]),
    })


# ---------------------------------------------------------------------------
# Membership gaps must not corrupt rolling windows
# ---------------------------------------------------------------------------


def test_ret_1d_across_a_membership_gap_is_a_real_one_day_return() -> None:
    """Regression: KDP left the index in 2018 and rejoined in 2022; its first
    row back showed ret_1d = +92.8% — a four-year return fed to the model as
    a one-day feature."""
    # GAP is flat at 100 through index 40, ramps to 200 over indices 40-70
    # while it is OUT of the index, then sits flat at 200.
    px = np.concatenate([
        np.full(40, 100.0), np.linspace(100.0, 200.0, 31)[1:], np.full(50, 200.0),
    ])
    assert len(px) == len(DATES)
    adj_close = _prices({"GAP": px, "CONT": np.full(len(DATES), 50.0)})
    stints = _stints([
        ("GAP", "2020-01-01", str(DATES[40].date())),   # out from index 40…
        ("GAP", str(DATES[80].date()), None),           # …back in at index 80
        ("CONT", "2020-01-01", None),
    ])

    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    panel, _ = build_feature_panel(
        labeled, _volume(["GAP", "CONT"]),
        start=str(DATES[0].date()), end=None,
        skip_earnings=True, earnings_workers=1,
        provider=_OfflineProvider(), macro_merge=False,
        stints=stints, sector_map=_sector_map(["GAP", "CONT"]),
    )

    gap = panel[panel["ticker"] == "GAP"].sort_values("date")
    # The out-of-index rows are excluded from the panel…
    assert not ((gap["date"] > DATES[40]) & (gap["date"] < DATES[80])).any()

    # …but the features on the re-entry row still reflect real history.
    reentry = gap[gap["date"] >= DATES[80]].iloc[0]
    assert reentry["date"] == DATES[80]
    assert reentry["ret_1d"] == pytest.approx(0.0, abs=1e-9)   # flat 200 → 200
    assert reentry["ret_5d"] == pytest.approx(0.0, abs=1e-9)

    # 21 sessions back from index 80 is index 59, mid-ramp. Filtering first
    # would instead reach back to index ~20 (price 100) and report +100%.
    expected = px[80] / px[59] - 1.0
    assert reentry["ret_21d"] == pytest.approx(expected, rel=1e-9)
    assert reentry["ret_21d"] < 0.5, "ret_21d spans the membership gap"


def test_long_lookback_features_survive_late_index_entry() -> None:
    """A ticker added to the index late still gets a real ret_21d on its
    first in-index day, because features saw its pre-membership history."""
    adj_close = _prices({
        "LATE": np.linspace(100.0, 160.0, len(DATES)),
        "OLD": np.linspace(80.0, 90.0, len(DATES)),
    })
    stints = _stints([
        ("LATE", str(DATES[100].date()), None),  # joins near the end
        ("OLD", "2020-01-01", None),
    ])

    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    panel, _ = build_feature_panel(
        labeled, _volume(["LATE", "OLD"]),
        start=str(DATES[0].date()), end=None,
        skip_earnings=True, earnings_workers=1,
        provider=_OfflineProvider(), macro_merge=False,
        stints=stints, sector_map=_sector_map(["LATE", "OLD"]),
    )

    late = panel[panel["ticker"] == "LATE"].sort_values("date")
    assert late["date"].min() >= DATES[100]
    first = late.iloc[0]
    assert not np.isnan(first["ret_21d"]), "21d lookback lost to the PIT filter"
    assert not np.isnan(first["vol_21d"])
    assert first["ret_21d"] > 0


# ---------------------------------------------------------------------------
# Cross-sectional features must see only in-index names
# ---------------------------------------------------------------------------


def test_cross_sectional_ranks_ignore_out_of_index_tickers() -> None:
    """A symbol that is downloaded but not an index member on a date must not
    shift that date's ranks or 'market' regime median."""
    adj_close = _prices({
        "IN1": np.linspace(100.0, 110.0, len(DATES)),
        "IN2": np.linspace(100.0, 105.0, len(DATES)),
        "OUT": np.linspace(100.0, 400.0, len(DATES)),  # huge outlier, not a member
    })
    stints = _stints([
        ("IN1", "2020-01-01", None),
        ("IN2", "2020-01-01", None),
        ("OUT", "1990-01-01", "2000-01-01"),  # membership long expired
    ])

    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    panel, _ = build_feature_panel(
        labeled, _volume(["IN1", "IN2", "OUT"]),
        start=str(DATES[0].date()), end=None,
        skip_earnings=True, earnings_workers=1,
        provider=_OfflineProvider(), macro_merge=False,
        stints=stints, sector_map=_sector_map(["IN1", "IN2", "OUT"]),
    )

    assert set(panel["ticker"].unique()) == {"IN1", "IN2"}
    day = panel[panel["date"] == panel["date"].max()]
    # Two members only: ranks are 0.5 and 1.0, and the median regime return
    # is the average of the two — the outlier never enters the cross-section.
    assert sorted(day["ret_21d_rank"].round(6)) == [0.5, 1.0]
    assert day["market_ret_5d"].nunique() == 1
    assert day["market_ret_5d"].iloc[0] == pytest.approx(day["ret_5d"].median())


# ---------------------------------------------------------------------------
# Stage helpers in isolation
# ---------------------------------------------------------------------------


def test_add_timeseries_features_is_order_insensitive_to_input_sorting() -> None:
    adj_close = _prices({
        "AAA": np.linspace(100.0, 120.0, len(DATES)),
        "BBB": np.linspace(60.0, 55.0, len(DATES)),
    })
    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    vol = _volume(["AAA", "BBB"])

    ordered = add_timeseries_features(labeled, vol)
    shuffled = add_timeseries_features(
        labeled.sample(frac=1.0, random_state=0), vol,
    )
    key = ["ticker", "date"]
    a = ordered.sort_values(key).reset_index(drop=True)
    b = shuffled.sort_values(key).reset_index(drop=True)
    pd.testing.assert_series_equal(a["ret_5d"], b["ret_5d"])
    pd.testing.assert_series_equal(a["volume_zscore"], b["volume_zscore"])


def test_add_cross_sectional_features_adds_expected_columns() -> None:
    adj_close = _prices({
        "AAA": np.linspace(100.0, 120.0, len(DATES)),
        "BBB": np.linspace(60.0, 55.0, len(DATES)),
    })
    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    ts = add_timeseries_features(labeled, _volume(["AAA", "BBB"]))
    out = add_cross_sectional_features(ts, _sector_map(["AAA", "BBB"]))
    for col in (
        "market_ret_5d", "market_ret_21d", "ret_21d_rank",
        "vol_10d_rank", "volume_zscore_rank",
        "ret_5d_vs_sector", "ret_21d_vs_sector", "vol_vs_sector",
    ):
        assert col in out.columns, col
    assert str(out["sector"].dtype) == "category"
