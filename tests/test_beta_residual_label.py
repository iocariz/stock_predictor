"""A ranking label that does not pay the model for holding beta.

Graded on raw forward return, the ranker learns that high-beta names top the
list in up markets and bottom it in down ones -- and since markets rose over
most of the sample, it learns to rank beta and volatility. The long-short book
built from it carried beta +0.20 and was long technology against financials;
with the sector leak removed, its alpha across rebalance schedules centred on
zero.

``beta_resid`` grades on ``fwd_ret - beta * market_fwd_ret`` instead: what a
name did beyond what its market exposure explains. Within a date the market
term is the same number for everyone, so only the *beta-scaled* part reorders
names -- which is exactly the part that is not selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import (
    BETA_BOUNDS,
    BETA_COL,
    BETA_MIN_OBS,
    LABEL_TARGETS,
    add_market_beta,
    add_price_features,
    add_rank_labels,
    build_feature_panel,
    build_labeled_panel,
    label_target_series,
)

DATES = pd.bdate_range("2022-01-03", periods=400)
LOADINGS = {"LOW": 0.5, "MID": 1.0, "HIGH": 1.5}


def _factor_prices(seed: int = 0, noise: float = 0.0) -> pd.DataFrame:
    """Long price panel where each name is a fixed multiple of one factor."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0004, 0.01, len(DATES))
    rows = []
    for ticker, loading in LOADINGS.items():
        ret = loading * factor + rng.normal(0.0, noise, len(DATES))
        px = 100.0 * np.cumprod(1.0 + ret)
        rows.append(pd.DataFrame({"date": DATES, "ticker": ticker, "adj_close": px}))
    return pd.concat(rows, ignore_index=True).sort_values(["ticker", "date"])


def _with_beta(prices: pd.DataFrame) -> pd.DataFrame:
    return add_market_beta(add_price_features(prices.copy()))


def _last(df: pd.DataFrame, ticker: str) -> float:
    return float(df[df["ticker"] == ticker][BETA_COL].iloc[-1])


# --- the trailing beta --------------------------------------------------------


def test_beta_recovers_each_names_loading_on_the_market() -> None:
    """The market here is the cross-sectional median, which is MID's return."""
    out = _with_beta(_factor_prices())
    for ticker, loading in LOADINGS.items():
        assert _last(out, ticker) == pytest.approx(loading, abs=1e-6)


def test_beta_is_undefined_until_there_is_enough_history() -> None:
    out = _with_beta(_factor_prices())
    one = out[out["ticker"] == "HIGH"].reset_index(drop=True)
    # ret_1d needs one prior session, so observation k lands on row k.
    assert one[BETA_COL].iloc[:BETA_MIN_OBS].isna().all()
    assert one[BETA_COL].iloc[BETA_MIN_OBS:].notna().all()


def test_beta_uses_only_the_past() -> None:
    """Rewriting every price after a date must not move the beta on it."""
    prices = _factor_prices(noise=0.004)
    cut = DATES[300]
    shocked = prices.copy()
    later = shocked["date"] > cut
    shocked.loc[later & (shocked["ticker"] == "HIGH"), "adj_close"] *= 3.0
    a = _with_beta(prices)
    b = _with_beta(shocked)
    on_or_before = a["date"] <= cut
    pd.testing.assert_series_equal(
        a.loc[on_or_before, BETA_COL], b.loc[on_or_before, BETA_COL])


def test_one_corrupt_print_does_not_move_everyone_elses_beta() -> None:
    """Dead symbols in this panel carry prints thousands of times too large.
    A mean market return would hand that to every name's beta for a year."""
    prices = _factor_prices(noise=0.004)
    extra = prices[prices["ticker"] == "MID"].assign(ticker="DEAD").copy()
    extra.loc[extra["date"] == DATES[200], "adj_close"] *= 5000.0
    clean = _with_beta(prices)
    dirty = _with_beta(pd.concat([prices, extra]).sort_values(["ticker", "date"]))
    assert _last(dirty, "HIGH") == pytest.approx(_last(clean, "HIGH"), abs=0.05)


def test_a_gap_in_one_name_does_not_blank_its_beta_forever() -> None:
    prices = _factor_prices(noise=0.004)
    hole = (prices["ticker"] == "LOW") & prices["date"].isin(DATES[150:153])
    prices.loc[hole, "adj_close"] = np.nan
    out = _with_beta(prices)
    assert np.isfinite(_last(out, "LOW"))


# --- the label ----------------------------------------------------------------


def _label_panel() -> pd.DataFrame:
    """Two dates. On each, forward return is exactly beta times the market's,
    except one name that beat its exposure and one that lagged it by as much
    -- a pair, so the cross-sectional median stays the market's return."""
    rows = []
    for date, mkt in ((DATES[0], 0.10), (DATES[1], -0.10)):
        for ticker, beta in (("B05", 0.5), ("B10", 1.0), ("B15", 1.5)):
            rows.append({"date": date, "ticker": ticker, BETA_COL: beta,
                         "fwd_ret": beta * mkt, "vol_21d": 0.02})
        for ticker, own in (("SKILL", 0.03), ("LAG", -0.03)):
            rows.append({"date": date, "ticker": ticker, BETA_COL: 1.0,
                         "fwd_ret": mkt + own, "vol_21d": 0.02})
    return pd.DataFrame(rows)


def test_beta_resid_is_offered() -> None:
    assert "beta_resid" in LABEL_TARGETS


def test_the_residual_is_return_minus_beta_times_the_market() -> None:
    panel = _label_panel()
    target = label_target_series(panel, "beta_resid")
    # Market = cross-sectional median forward return: +0.10 and -0.10.
    by_ticker = target.groupby(panel["ticker"]).agg(list)
    for ticker in ("B05", "B10", "B15"):
        assert by_ticker[ticker] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert by_ticker["SKILL"] == pytest.approx([0.03, 0.03])
    assert by_ticker["LAG"] == pytest.approx([-0.03, -0.03])


def test_raw_grades_reward_beta_and_the_residual_does_not() -> None:
    """In the up market raw puts the highest beta on top; in the down market
    it puts it at the bottom. The residual ranks the one name with a return of
    its own first on both dates."""
    panel = _label_panel()
    raw = add_rank_labels(panel, n_grades=5, label_target="raw")
    res = add_rank_labels(panel, n_grades=5, label_target="beta_resid")

    def top(df: pd.DataFrame, date: pd.Timestamp) -> str:
        day = df[df["date"] == date]
        return str(day.loc[day["rank_grade"].idxmax(), "ticker"])

    assert top(raw, DATES[0]) == "B15"
    assert top(raw, DATES[1]) == "B05"
    assert top(res, DATES[0]) == "SKILL"
    assert top(res, DATES[1]) == "SKILL"


def test_a_name_without_a_beta_is_treated_as_the_market_not_as_the_worst() -> None:
    """NaN would be graded 0 -- the worst of the day -- which turns "too new to
    have a beta" into a label. Beta 1 is the neutral assumption."""
    panel = _label_panel()
    panel.loc[panel["ticker"] == "SKILL", BETA_COL] = np.nan
    target = label_target_series(panel, "beta_resid")
    assert target[panel["ticker"] == "SKILL"].tolist() == pytest.approx([0.03, 0.03])


def test_asking_for_it_without_a_beta_column_says_what_is_missing() -> None:
    with pytest.raises(ValueError, match=BETA_COL):
        label_target_series(_label_panel().drop(columns=[BETA_COL]), "beta_resid")


# --- the panel ----------------------------------------------------------------


class _OfflineProvider:
    def download_macro(self, start: str, end: str | None) -> pd.DataFrame:
        return pd.DataFrame({"date": DATES, "vix": 18.0, "tnx_yield": 3.5,
                             "irx_yield": 2.2})


def test_beta_is_on_the_panel_but_is_not_a_model_feature() -> None:
    """It exists to build the label. Handing it to the model as well would
    change two things at once, and the experiment is about the label."""
    wide = _factor_prices(noise=0.004).pivot(
        index="date", columns="ticker", values="adj_close")
    volume = pd.DataFrame(1e6, index=wide.index, columns=wide.columns)
    labeled = build_labeled_panel(wide, None, horizon=5, threshold=0.05)
    features, feature_cols = build_feature_panel(
        labeled, volume, start=str(DATES[0].date()), end=None,
        skip_earnings=True, earnings_workers=1, provider=_OfflineProvider(),
        macro_merge=False,
        sector_map=pd.DataFrame({"ticker": list(LOADINGS), "sector": "Tech"}),
    )
    assert BETA_COL in features.columns
    assert features[BETA_COL].notna().any()
    assert BETA_COL not in feature_cols


# --- corrupt prints -----------------------------------------------------------


def test_a_names_own_corrupt_print_does_not_wreck_its_beta() -> None:
    """Measured on the real panel before this was bounded: betas from -3,564
    to +510, from dead symbols printing thousands of times too high for a
    session. One such return dominates a year of covariance."""
    prices = _factor_prices(noise=0.004)
    clean = _with_beta(prices)
    spiked = prices.copy()
    hit = (spiked["ticker"] == "HIGH") & (spiked["date"] == DATES[380])
    spiked.loc[hit, "adj_close"] *= 5000.0
    assert _last(_with_beta(spiked), "HIGH") == pytest.approx(
        _last(clean, "HIGH"), abs=0.5)


def test_beta_never_leaves_its_stated_bounds() -> None:
    prices = _factor_prices(noise=0.004)
    wild = prices.copy()
    rng = np.random.default_rng(3)
    rows = wild["ticker"] == "LOW"
    wild.loc[rows, "adj_close"] *= np.exp(rng.normal(0, 2.0, int(rows.sum())))
    beta = _with_beta(wild)[BETA_COL].dropna()
    low, high = BETA_BOUNDS
    assert beta.between(low, high).all()
