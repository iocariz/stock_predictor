"""You cannot rank what you cannot price, and you cannot rank a truncated list.

Two faults in the live path:

* `score_universe` dropped a row only when **all** its ticker-level features
  were NaN. Calendar features like `days_to_fomc` are date-level constants but
  are not in `MACRO_FEATURE_COLS`, so they counted as ticker-level and kept an
  unpriced row alive. It was then ranked, entered `latest_prices` as NaN, and
  inflated the cross-section width that `min_cross_section` measures.

* Fixed-hold passed only `scored.head(top_n * 2)` to the shared selection.
  With `top_n=5, rank_offset=10` the full cross-section selects ranks 11-15;
  the truncated input selects **nothing** — ten rows cannot survive an offset
  of ten, and ten is below the cross-section floor of fifteen. Rank-hold
  already passed the full ranking; fixed-hold was the odd one out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.portfolio import PortfolioState, generate_orders
from stock_predictor.predict import score_universe

DATES = pd.bdate_range(end="2024-06-14", periods=200)
AS_OF = "2024-06-14"
FEATS = ["ret_1d", "mom_21d", "vix", "days_to_fomc"]


class _Ranker:
    def predict(self, X):
        return np.arange(len(X), dtype=float)[::-1]


def _panel(bad_price=np.nan) -> pd.DataFrame:
    d = pd.Timestamp(AS_OF)
    rows = []
    for i in range(6):
        first = i == 0
        rows.append({
            "date": d, "ticker": f"T{i:02d}",
            "adj_close": bad_price if first else 100.0 + i,
            "ret_1d": np.nan if first else 0.01,
            "mom_21d": np.nan if first else 0.05,
            # Date-level constants: present for every row, priced or not.
            "vix": 16.0, "days_to_fomc": 7.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A price is required to rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
def test_a_row_without_a_usable_price_is_not_ranked(bad) -> None:
    out = score_universe(_Ranker(), _panel(bad), FEATS)
    assert "T00" not in set(out["ticker"])


def test_calendar_features_do_not_keep_an_unpriced_row_alive() -> None:
    """days_to_fomc is date-level but was counted as a ticker-level feature."""
    out = score_universe(_Ranker(), _panel(), FEATS)
    assert out["adj_close"].notna().all()
    assert (out["adj_close"] > 0).all()


def test_priced_names_are_all_still_scored() -> None:
    out = score_universe(_Ranker(), _panel(), FEATS)
    assert set(out["ticker"]) == {f"T{i:02d}" for i in range(1, 6)}


def test_dropping_the_unpriced_row_does_not_empty_the_panel() -> None:
    out = score_universe(_Ranker(), _panel(), FEATS)
    assert len(out) == 5


def test_a_fully_unpriced_day_is_an_error_not_a_silent_empty() -> None:
    panel = _panel()
    panel["adj_close"] = np.nan
    with pytest.raises(ValueError):
        score_universe(_Ranker(), panel, FEATS)


# ---------------------------------------------------------------------------
# The whole cross-section reaches selection
# ---------------------------------------------------------------------------


def _scored(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)],
        "prob": np.linspace(0.99, 0.01, n),
        "adj_close": [100.0] * n,
    })


def _buys(picks) -> list[str]:
    scored = _scored()
    orders, _ = generate_orders(
        PortfolioState(cash=100_000.0), picks,
        dict(zip(scored.ticker, scored.adj_close, strict=True)),
        top_n=5, max_cohorts=2, holding_days=10, slippage_bps=0.0,
        as_of=AS_OF, trading_dates=DATES.to_numpy(), rank_offset=10, force=True,
    )
    return sorted(o.ticker for o in orders if o.action == "BUY")


def test_a_rank_band_needs_the_full_cross_section() -> None:
    """The reproduction: ranks 11-15 with everything, nothing with head(10)."""
    full = _scored().to_dict("records")
    assert _buys(full) == ["T10", "T11", "T12", "T13", "T14"]
    assert _buys(full[: 5 * 2]) == [], "truncation silently selects nothing"


def test_the_live_path_passes_everything_it_scored() -> None:
    """Fixed-hold truncated to top_n*2; rank-hold already did not."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "src" / "stock_predictor" / "predict.py").read_text()
    assert "head(args.top_n * 2)" not in src
    assert "scored.to_dict(\"records\")" in src


def test_a_score_floor_also_needs_the_full_list() -> None:
    """A floor applied to a pre-truncated list is a different basket."""
    scored = _scored()
    prices = dict(zip(scored.ticker, scored.adj_close, strict=True))
    kw = dict(top_n=5, max_cohorts=2, holding_days=10, slippage_bps=0.0,
              as_of=AS_OF, trading_dates=DATES.to_numpy(), min_prob=0.3,
              force=True)
    full, _ = generate_orders(PortfolioState(cash=100_000.0),
                              scored.to_dict("records"), prices, **kw)
    trunc, _ = generate_orders(PortfolioState(cash=100_000.0),
                               scored.head(10).to_dict("records"), prices, **kw)
    assert [o.ticker for o in full if o.action == "BUY"]
    assert len([o for o in full if o.action == "BUY"]) >= \
        len([o for o in trunc if o.action == "BUY"])
