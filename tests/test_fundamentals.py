"""Point-in-time discipline for EDGAR fundamentals.

The join is the whole game: a figure for the quarter ending March is not
knowable until it is filed, and EDGAR also carries later filings that revise
earlier periods. Both hazards are lookahead bugs if handled naively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.fundamentals import (
    CONCEPT_TAGS,
    FLOW_CONCEPTS,
    FUNDAMENTAL_FEATURE_COLS,
    asof_join_fundamentals,
    extract_concepts,
    trailing_twelve_months,
)


def _facts(tag: str = "Revenues") -> dict:
    """Minimal companyfacts payload: four quarters, each filed ~40 days late."""
    obs = [
        {"start": "2024-01-01", "end": "2024-03-31", "val": 100.0,
         "filed": "2024-05-10", "form": "10-Q", "fy": 2024, "fp": "Q1"},
        {"start": "2024-04-01", "end": "2024-06-30", "val": 110.0,
         "filed": "2024-08-09", "form": "10-Q", "fy": 2024, "fp": "Q2"},
        {"start": "2024-07-01", "end": "2024-09-30", "val": 120.0,
         "filed": "2024-11-08", "form": "10-Q", "fy": 2024, "fp": "Q3"},
        {"start": "2024-10-01", "end": "2024-12-31", "val": 130.0,
         "filed": "2025-02-14", "form": "10-Q", "fy": 2024, "fp": "Q4"},
        # Not a periodic report: must be ignored.
        {"start": "2024-10-01", "end": "2024-12-31", "val": 999.0,
         "filed": "2025-02-20", "form": "8-K", "fy": 2024, "fp": "Q4"},
    ]
    return {"facts": {"us-gaap": {tag: {"units": {"USD": obs}}}}}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extraction_keeps_periodic_reports_only() -> None:
    out = extract_concepts(_facts(), "AAA")
    assert len(out) == 4
    assert set(out["form"]) == {"10-Q"}
    assert 999.0 not in set(out["value"])


def test_extraction_carries_the_filed_date() -> None:
    out = extract_concepts(_facts(), "AAA").sort_values("period_end")
    first = out.iloc[0]
    assert first["period_end"] == pd.Timestamp("2024-03-31")
    assert first["filed"] == pd.Timestamp("2024-05-10")
    assert first["filed"] > first["period_end"], "filing must postdate the period"


def test_extraction_falls_back_across_synonym_tags() -> None:
    """Companies tag revenue differently; both must land on 'revenue'."""
    for tag in CONCEPT_TAGS["revenue"][:3]:
        out = extract_concepts(_facts(tag), "AAA")
        assert set(out["concept"]) == {"revenue"}, tag
        assert len(out) == 4


def test_extraction_of_empty_facts_is_an_empty_frame() -> None:
    out = extract_concepts({"facts": {"us-gaap": {}}}, "AAA")
    assert out.empty
    assert "filed" in out.columns


# ---------------------------------------------------------------------------
# The point-in-time join
# ---------------------------------------------------------------------------


def _fund_table() -> pd.DataFrame:
    return trailing_twelve_months(extract_concepts(_facts(), "AAA"))


def _stock_facts(scale: float = 1.0) -> dict:
    """Balance-sheet payload: a level, available the moment it is filed."""
    obs = [
        {"end": "2024-03-31", "val": 1000.0 * scale, "filed": "2024-05-10",
         "form": "10-Q", "fy": 2024, "fp": "Q1"},
        {"end": "2024-06-30", "val": 1100.0 * scale, "filed": "2024-08-09",
         "form": "10-Q", "fy": 2024, "fp": "Q2"},
        {"end": "2024-09-30", "val": 1200.0 * scale, "filed": "2024-11-08",
         "form": "10-Q", "fy": 2024, "fp": "Q3"},
    ]
    return {"facts": {"us-gaap": {"Assets": {"units": {"USD": obs}}}}}


def _panel(dates: list[str], ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": ticker})


def test_figures_are_invisible_before_they_are_filed() -> None:
    """The core hazard: Q1 ends 2024-03-31 but is not filed until 2024-05-10,
    so 2024-04-15 must see nothing at all."""
    fund = trailing_twelve_months(extract_concepts(_stock_facts(), "AAA"))
    joined = asof_join_fundamentals(
        _panel(["2024-04-15", "2024-05-09", "2024-05-10", "2024-06-01",
                "2024-08-09"]),
        fund,
    )
    vals = joined["raw_assets"].tolist()
    assert np.isnan(vals[0]), "period end must not make a figure available"
    assert np.isnan(vals[1]), "the day before filing must still be blind"
    assert vals[2] == 1000.0, "available on the filing date"
    assert vals[3] == 1000.0, "carried forward until the next filing"
    assert vals[4] == 1100.0, "advances when the next report lands"


def test_a_flow_column_never_mixes_quarterly_and_annual_figures() -> None:
    """`raw_revenue` must mean trailing-twelve-months at every date, or be NaN.

    Falling back to a bare quarter before four are available would make the
    column mean one thing early in a company's history and another later — a
    level shift the model would learn as signal.
    """
    joined = asof_join_fundamentals(
        _panel(["2024-06-01", "2024-08-09", "2024-11-08", "2025-02-14"]),
        _fund_table(),
    )
    seen = joined["raw_revenue"].tolist()
    assert all(np.isnan(v) for v in seen[:3]), "no TTM until four quarters are filed"
    assert seen[3] == pytest.approx(100 + 110 + 120 + 130)


def test_restatement_does_not_rewrite_earlier_history() -> None:
    """A revision of Q1 filed in November must not change what May-October saw."""
    base = extract_concepts(_stock_facts(), "AAA")
    restated = pd.DataFrame([{
        "ticker": "AAA", "concept": "assets",
        "period_end": pd.Timestamp("2024-03-31"),
        "period_start": pd.NaT,
        "filed": pd.Timestamp("2024-11-08"), "value": 555.0,
        "form": "10-Q", "fy": 2024, "fp": "Q1",
    }])
    dates = ["2024-05-10", "2024-06-01", "2024-10-01", "2024-11-07", "2024-11-30"]
    without = asof_join_fundamentals(
        _panel(dates), trailing_twelve_months(base),
    )["raw_assets"].to_numpy()
    with_restated = asof_join_fundamentals(
        _panel(dates),
        trailing_twelve_months(pd.concat([base, restated], ignore_index=True)),
    )["raw_assets"].to_numpy()

    pre = np.array([pd.Timestamp(d) < pd.Timestamp("2024-11-08") for d in dates])
    assert not np.isnan(without[pre]).all(), "fixture must expose real values pre-restatement"
    np.testing.assert_allclose(without[pre], with_restated[pre])
    assert 555.0 not in set(with_restated[pre]), "restated figure leaked backwards"


def test_join_is_per_ticker() -> None:
    fund = trailing_twelve_months(pd.concat([
        extract_concepts(_stock_facts(1.0), "AAA"),
        extract_concepts(_stock_facts(10.0), "BBB"),
    ], ignore_index=True))
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
        "ticker": ["AAA", "BBB"],
    })
    joined = asof_join_fundamentals(panel, fund)
    got = dict(zip(joined["ticker"], joined["raw_assets"]))
    assert got["AAA"] == 1000.0
    assert got["BBB"] == 10000.0


def test_join_preserves_panel_rows_and_order() -> None:
    panel = _panel(["2024-06-03", "2024-06-04", "2024-06-05"])
    joined = asof_join_fundamentals(panel, _fund_table())
    assert len(joined) == len(panel)
    pd.testing.assert_series_equal(joined["date"], panel["date"])


def test_empty_fundamentals_returns_the_panel_untouched() -> None:
    panel = _panel(["2024-06-01"])
    out = asof_join_fundamentals(panel, pd.DataFrame())
    pd.testing.assert_frame_equal(out, panel)


# ---------------------------------------------------------------------------
# Trailing twelve months
# ---------------------------------------------------------------------------


def test_ttm_needs_four_quarters_before_it_reports() -> None:
    ttm = _fund_table().sort_values("period_end")
    vals = ttm["ttm"].tolist()
    assert all(np.isnan(v) for v in vals[:3]), "cannot sum a year from three quarters"
    assert vals[3] == pytest.approx(100 + 110 + 120 + 130)


def test_ttm_inherits_the_newest_component_filing_date() -> None:
    """The TTM figure must not be available before its last quarter was filed."""
    ttm = _fund_table().sort_values("period_end")
    complete = ttm[ttm["ttm"].notna()].iloc[0]
    assert complete["filed"] == pd.Timestamp("2025-02-14")


def test_balance_items_are_not_summed() -> None:
    obs = [
        {"end": "2024-03-31", "val": 1000.0, "filed": "2024-05-10",
         "form": "10-Q", "fy": 2024, "fp": "Q1"},
        {"end": "2024-06-30", "val": 1100.0, "filed": "2024-08-09",
         "form": "10-Q", "fy": 2024, "fp": "Q2"},
    ]
    facts = {"facts": {"us-gaap": {"Assets": {"units": {"USD": obs}}}}}
    out = trailing_twelve_months(extract_concepts(facts, "AAA"))
    assert "assets" not in FLOW_CONCEPTS
    assert out["ttm"].isna().all(), "a balance is a level, not a flow"
    joined = asof_join_fundamentals(_panel(["2024-09-01"]), out)
    assert joined["raw_assets"].iloc[0] == 1100.0


def test_feature_columns_are_declared() -> None:
    assert len(FUNDAMENTAL_FEATURE_COLS) >= 8
    assert all(c.startswith("fund_") for c in FUNDAMENTAL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Derived ratios
# ---------------------------------------------------------------------------


def test_ratios_are_scale_free() -> None:
    """Two identical businesses differing only in size must score the same —
    a level feature would just proxy market cap."""
    from stock_predictor.fundamentals import add_fundamental_features

    small = {"adj_close": 10.0, "raw_net_income": 1e6, "raw_equity": 5e6,
             "raw_revenue": 1e7, "raw_assets": 1e7, "raw_shares_diluted": 1e6,
             "raw_cash_ops": 2e6, "raw_capex": 5e5, "raw_gross_profit": 4e6,
             "raw_operating_income": 2e6, "raw_liabilities": 5e6}
    big = {k: (v * 1000 if k != "adj_close" else v) for k, v in small.items()}
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-06-01"), "ticker": "SML", **small},
        {"date": pd.Timestamp("2024-06-01"), "ticker": "BIG", **big},
    ])
    out = add_fundamental_features(df)
    for col in ("fund_roe", "fund_gross_margin", "fund_debt_to_equity"):
        a, b = out.set_index("ticker")[col].loc[["SML", "BIG"]]
        assert a == pytest.approx(b), col


def test_ratios_are_finite_or_nan_never_infinite() -> None:
    from stock_predictor.fundamentals import add_fundamental_features

    df = pd.DataFrame([{
        "date": pd.Timestamp("2024-06-01"), "ticker": "ZER", "adj_close": 10.0,
        "raw_net_income": 1e6, "raw_equity": 0.0, "raw_revenue": 0.0,
        "raw_assets": 0.0, "raw_shares_diluted": 0.0, "raw_cash_ops": 1e6,
        "raw_capex": 0.0, "raw_gross_profit": 0.0,
        "raw_operating_income": 0.0, "raw_liabilities": 1e6,
    }])
    out = add_fundamental_features(df)
    vals = out[FUNDAMENTAL_FEATURE_COLS].to_numpy(dtype=float)
    assert not np.isinf(vals).any(), "zero denominators must yield NaN, not inf"


def test_missing_fundamentals_yields_all_nan_columns() -> None:
    from stock_predictor.fundamentals import add_fundamental_features

    df = pd.DataFrame([{"date": pd.Timestamp("2024-06-01"),
                        "ticker": "AAA", "adj_close": 10.0}])
    out = add_fundamental_features(df)
    for col in FUNDAMENTAL_FEATURE_COLS:
        assert col in out.columns
        assert out[col].isna().all()


def test_negative_equity_does_not_produce_a_ratio() -> None:
    """Book-to-price on negative equity is meaningless, not merely negative."""
    from stock_predictor.fundamentals import add_fundamental_features

    df = pd.DataFrame([{
        "date": pd.Timestamp("2024-06-01"), "ticker": "NEG", "adj_close": 10.0,
        "raw_net_income": 1e6, "raw_equity": -5e6, "raw_revenue": 1e7,
        "raw_assets": 1e7, "raw_shares_diluted": 1e6, "raw_cash_ops": 1e6,
        "raw_capex": 0.0, "raw_gross_profit": 4e6,
        "raw_operating_income": 2e6, "raw_liabilities": 5e6,
    }])
    out = add_fundamental_features(df)
    assert np.isnan(out["fund_book_to_price"].iloc[0])
    assert np.isnan(out["fund_roe"].iloc[0])


# ---------------------------------------------------------------------------
# Did the model use them?
# ---------------------------------------------------------------------------


def test_importances_sum_to_one_and_rank_by_gain() -> None:
    import lightgbm as lgb

    from stock_predictor.training import feature_importances

    rng = np.random.default_rng(0)
    n = 600
    useful = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (useful + 0.05 * rng.normal(size=n) > 0).astype(int)
    X = pd.DataFrame({"useful": useful, "noise": noise})
    clf = lgb.LGBMClassifier(n_estimators=40, verbosity=-1).fit(X, y)

    imp = feature_importances(clf, ["useful", "noise"])
    assert set(imp) == {"useful", "noise"}
    assert sum(imp.values()) == pytest.approx(1.0)
    assert list(imp)[0] == "useful", "the predictive feature must rank first"
    assert imp["useful"] > imp["noise"]


def test_unused_feature_gets_near_zero_importance() -> None:
    """The check that answers 'were the fundamentals used at all?'"""
    import lightgbm as lgb

    from stock_predictor.training import feature_importances, importance_by_group

    rng = np.random.default_rng(1)
    n = 600
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "px_signal": signal,
        "fund_a": np.zeros(n),          # constant: nothing to split on
        "fund_b": rng.normal(size=n),   # pure noise
    })
    y = (signal > 0).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=40, verbosity=-1).fit(X, y)

    imp = feature_importances(clf, list(X.columns))
    grouped = importance_by_group(imp, {"price": ("px_",), "fundamentals": ("fund_",)})
    assert grouped["price"] > 0.5
    assert grouped["fundamentals"] < 0.5
    assert imp.get("fund_a", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_importances_empty_for_a_model_without_a_booster() -> None:
    from stock_predictor.training import feature_importances

    assert feature_importances(object(), ["a", "b"]) == {}


def test_join_survives_mismatched_datetime_resolutions() -> None:
    """Regression: pandas >= 3.0 keeps inferred datetime units, and merge_asof
    refuses M8[s] against M8[us]. A real run failed here after the unit tests
    passed, because the synthetic fixtures happened to agree on units."""
    fund = trailing_twelve_months(extract_concepts(_stock_facts(), "AAA"))
    fund["filed"] = fund["filed"].astype("datetime64[us]")

    panel = _panel(["2024-06-03"])
    panel["date"] = panel["date"].astype("datetime64[s]")
    assert panel["date"].dtype != fund["filed"].dtype, "fixture must mismatch"

    joined = asof_join_fundamentals(panel, fund)
    assert joined["raw_assets"].iloc[0] == 1000.0


def test_join_handles_every_common_resolution_pair() -> None:
    for panel_unit in ("datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"):
        for fund_unit in ("datetime64[s]", "datetime64[us]", "datetime64[ns]"):
            fund = trailing_twelve_months(extract_concepts(_stock_facts(), "AAA"))
            fund["filed"] = fund["filed"].astype(fund_unit)
            panel = _panel(["2024-06-03"])
            panel["date"] = panel["date"].astype(panel_unit)
            out = asof_join_fundamentals(panel, fund)
            assert out["raw_assets"].iloc[0] == 1000.0, (panel_unit, fund_unit)
