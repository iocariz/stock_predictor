"""Derive fundamentals that filers report indirectly.

Two features were far below the rest on coverage — gross margin at 45.5% and
debt-to-equity at 66.2% — not because the data is absent but because the tag
we looked for is. Accenture and Airbnb file `CostOfRevenue` and never
`GrossProfit`; plenty of filers omit `Liabilities` while reporting both sides
of the balance sheet.

Both recoveries are accounting identities, not estimates:

    gross_profit = revenue - cost_of_revenue
    liabilities  = assets  - equity

The filed value always wins where it exists; the identity only fills gaps.
Airlines, utilities and insurers genuinely have no gross-profit line, so a
residual gap there is structural and must stay NaN rather than be invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_predictor.fundamentals import CONCEPT_TAGS, FLOW_CONCEPTS, add_fundamental_features


def _panel(**raw) -> pd.DataFrame:
    n = len(next(iter(raw.values())))
    base = {"date": pd.bdate_range("2024-01-01", periods=n),
            "ticker": ["AAA"] * n, "adj_close": [100.0] * n,
            "raw_revenue": [1000.0] * n, "raw_net_income": [100.0] * n,
            "raw_equity": [500.0] * n, "raw_assets": [900.0] * n,
            "raw_shares_diluted": [10.0] * n}
    base.update({f"raw_{k}": v for k, v in raw.items()})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Gross margin
# ---------------------------------------------------------------------------


def test_cost_of_revenue_is_a_tracked_concept() -> None:
    assert "cost_of_revenue" in CONCEPT_TAGS
    assert "CostOfRevenue" in CONCEPT_TAGS["cost_of_revenue"]
    assert "cost_of_revenue" in FLOW_CONCEPTS, "it is a per-period flow, like revenue"


def test_gross_margin_is_derived_when_the_filer_omits_gross_profit() -> None:
    """Accenture's case: CostOfRevenue filed, GrossProfit never."""
    out = add_fundamental_features(_panel(cost_of_revenue=[600.0]))
    assert out["fund_gross_margin"].iloc[0] == 0.4, "(1000 - 600) / 1000"


def test_a_filed_gross_profit_beats_the_derivation() -> None:
    """Where the filer states it, use it — the identity is only a fallback."""
    out = add_fundamental_features(
        _panel(gross_profit=[350.0], cost_of_revenue=[600.0]),
    )
    assert out["fund_gross_margin"].iloc[0] == 0.35


def test_the_derivation_fills_only_the_rows_that_need_it() -> None:
    out = add_fundamental_features(
        _panel(gross_profit=[350.0, np.nan], cost_of_revenue=[600.0, 600.0]),
    )
    assert list(out["fund_gross_margin"]) == [0.35, 0.4]


def test_no_cost_of_revenue_leaves_gross_margin_missing() -> None:
    """Airlines and insurers have no gross-profit line. NaN is the honest
    answer; inventing one would be worse than the gap."""
    out = add_fundamental_features(_panel(operating_income=[80.0]))
    assert out["fund_gross_margin"].isna().all()


# ---------------------------------------------------------------------------
# Debt to equity
# ---------------------------------------------------------------------------


def test_liabilities_are_derived_from_the_balance_sheet_identity() -> None:
    out = add_fundamental_features(_panel(cost_of_revenue=[600.0]))
    # assets 900 - equity 500 = 400; 400 / 500
    assert out["fund_debt_to_equity"].iloc[0] == 0.8


def test_filed_liabilities_beat_the_identity() -> None:
    out = add_fundamental_features(_panel(liabilities=[450.0]))
    assert out["fund_debt_to_equity"].iloc[0] == 0.9


def test_the_identity_needs_both_sides() -> None:
    p = _panel(liabilities=[np.nan])
    p["raw_assets"] = np.nan
    out = add_fundamental_features(p)
    assert out["fund_debt_to_equity"].isna().all()


def test_a_negative_derived_liability_is_rejected() -> None:
    """Equity above assets means the two came from different filings, not a
    company with negative liabilities."""
    p = _panel(liabilities=[np.nan])
    p["raw_equity"] = [900.0]
    p["raw_assets"] = [500.0]
    out = add_fundamental_features(p)
    assert out["fund_debt_to_equity"].isna().all()


# ---------------------------------------------------------------------------
# Nothing else moves
# ---------------------------------------------------------------------------


def test_other_features_are_unchanged_by_the_derivations() -> None:
    a = add_fundamental_features(_panel(gross_profit=[350.0], liabilities=[450.0]))
    b = add_fundamental_features(_panel(cost_of_revenue=[650.0]))
    for col in ("fund_earnings_yield", "fund_book_to_price", "fund_sales_to_price",
                "fund_roe", "fund_accruals"):
        assert a[col].iloc[0] == b[col].iloc[0] or (
            pd.isna(a[col].iloc[0]) and pd.isna(b[col].iloc[0])
        ), col
