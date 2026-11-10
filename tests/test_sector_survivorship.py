"""The sector map must not tell the model which companies left the index.

The map is scraped from today's constituent list, so only *current* members
have a sector. Every departed member had ``sector = NaN`` and NaN
``*_vs_sector`` features, on every historical row -- a flag reading "this
company will be removed before the data was downloaded", visible years before
it happened. On the 2010-2026 baseline it was the model's top feature by gain
(14%), flagged names returned -8.3% per 63 sessions against the rest in
2023-2026, and they made up 30% of the short decile against a 12% base rate.
A live run never sees the flag: every candidate is a current member.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.training import (
    add_cross_sectional_features,
    add_timeseries_features,
    build_feature_panel,
    build_labeled_panel,
)

DATES = pd.bdate_range("2024-01-01", periods=120)
SECTOR_SPREADS = ("ret_5d_vs_sector", "ret_21d_vs_sector", "vol_vs_sector")


class _OfflineProvider:
    def download_macro(self, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame({
            "date": DATES,
            "vix": np.linspace(15.0, 25.0, len(DATES)),
            "tnx_yield": np.linspace(3.0, 4.0, len(DATES)),
            "irx_yield": np.linspace(2.0, 2.5, len(DATES)),
        })


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {}
    for i, t in enumerate(["KEEP1", "KEEP2", "KEEP3", "GONE"]):
        steps = rng.normal(0.0005 * (i + 1), 0.01, len(DATES))
        cols[t] = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(cols, index=DATES)


def _volume(cols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {c: rng.uniform(5e5, 2e6, len(DATES)) for c in cols}, index=DATES,
    )


def _survivors_only_map() -> pd.DataFrame:
    """What Wikipedia returns: the names still in the index, and nobody else."""
    return pd.DataFrame({
        "ticker": ["KEEP1", "KEEP2", "KEEP3"],
        "sector": ["Tech", "Tech", "Energy"],
    })


def _cross_sectional() -> pd.DataFrame:
    adj_close = _prices()
    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    ts = add_timeseries_features(labeled, _volume(list(adj_close.columns)))
    return add_cross_sectional_features(ts, _survivors_only_map())


def test_sector_spreads_are_not_missing_for_a_name_without_a_sector() -> None:
    out = _cross_sectional()
    gone = out[out["ticker"] == "GONE"]
    for spread, base in zip(SECTOR_SPREADS, ("ret_5d", "ret_21d", "vol_10d")):
        has_base = gone[base].notna()
        assert has_base.any()
        # Missing exactly where the underlying return is, and nowhere else.
        assert gone.loc[has_base, spread].notna().all(), spread


def test_unknown_sector_falls_back_to_the_market_peer_group() -> None:
    out = _cross_sectional()
    day = out[out["date"] == DATES[-1]].set_index("ticker")
    for spread, base in zip(SECTOR_SPREADS, ("ret_5d", "ret_21d", "vol_10d")):
        expected = day.at["GONE", base] - day[base].median()
        assert np.isclose(day.at["GONE", spread], expected), spread


def test_known_sector_spreads_are_unchanged_by_the_fallback() -> None:
    out = _cross_sectional()
    day = out[out["date"] == DATES[-1]].set_index("ticker")
    tech_median = day.loc[["KEEP1", "KEEP2"], "ret_21d"].median()
    assert np.isclose(
        day.at["KEEP1", "ret_21d_vs_sector"], day.at["KEEP1", "ret_21d"] - tech_median,
    )
    # A one-name sector is its own median.
    assert np.isclose(day.at["KEEP3", "ret_21d_vs_sector"], 0.0)


def _panel() -> tuple[pd.DataFrame, list[str]]:
    adj_close = _prices()
    labeled = build_labeled_panel(adj_close, None, horizon=5, threshold=0.05)
    return build_feature_panel(
        labeled, _volume(list(adj_close.columns)),
        start="2024-01-01", end=None, skip_earnings=True, earnings_workers=1,
        provider=_OfflineProvider(), macro_merge=False,
        sector_map=_survivors_only_map(),
    )


def test_raw_sector_is_not_a_model_feature() -> None:
    features, feature_cols = _panel()
    assert "sector" not in feature_cols
    # Still on the panel: the sector-neutral book and the reports read it.
    assert "sector" in features.columns


def test_no_model_feature_reveals_which_names_lack_a_sector() -> None:
    """The property itself, over the whole feature list rather than by name.

    Per date, no feature may be missing for the sector-less name while present
    for every name that has one -- that pattern is the flag, whichever column
    carries it.
    """
    features, feature_cols = _panel()
    warm = features[features["date"] >= DATES[70]]
    numeric = [c for c in feature_cols if c != "sector"]
    gone_nan = warm[warm["ticker"] == "GONE"].set_index("date")[numeric].isna()
    kept_nan = (
        warm[warm["ticker"] != "GONE"].groupby("date")[numeric]
        .agg(lambda s: s.isna().any())
    )
    telltale = gone_nan & ~kept_nan.reindex(gone_nan.index)
    offenders = [c for c in numeric if telltale[c].any()]
    assert offenders == [], offenders
