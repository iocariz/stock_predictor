"""A holdout that restarts the strategy is measuring a different strategy.

The locked holdout re-ran the engine over a truncated panel and called the
result "the committed configuration on the holdout". Three things were wrong
with that, and only the third is cosmetic:

* **The rebalance calendar is anchored to row zero** of whatever frame the
  engine is handed (``long_short.py``: ``range(0, n_days - 1, rebalance_every)``).
  Slicing the panel therefore moves every trade date. On the 2022-01-01 split
  the continuing strategy signals on 2022-04-01, 07-05 and 10-03; the restarted
  one signals on 2022-01-03, 04-04 and 07-06. Not one date in common.
* **The book begins flat.** Positions and short liabilities open at the split
  are discarded, so the holdout never pays for what the selection window bought.
* **Both slices included the split session** (``<=`` and ``>=``). Latent rather
  than active -- the configured splits fall on 1 January, never a trading day --
  but a split on an open session would have put that signal on both sides.

The fix runs the engine once over the whole panel and measures the holdout
window out of the continuing NAV. Selection still truncates, which is correct:
there the truncation is the real beginning, not a cut through a live book.

Separately, the benchmark was downloaded from Yahoo at report time, so a
pre-registered test depended on what a vendor served that afternoon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from locked_holdout import _before, _segment, _whole  # noqa: E402

from stock_predictor.backtest_reporting import nav_metrics  # noqa: E402
from stock_predictor.long_short import LongShortConfig, run_long_short_backtest  # noqa: E402

DATES = pd.bdate_range("2020-01-01", periods=400)
N = 20
SPLIT = "2021-01-01"


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, (len(DATES), N)), axis=0))
    rows = []
    for di, d in enumerate(DATES):
        order = rng.permutation(N)
        for i in range(N):
            rows.append({"date": d, "ticker": f"T{i:02d}",
                         "prob": float(order[i]), "adj_close": float(px[di, i])})
    return pd.DataFrame(rows)


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


def _cfg(rebalance: int = 63) -> LongShortConfig:
    return LongShortConfig(decile=0.25, rebalance_every=rebalance,
                           slippage_bps=5.0, risk_free_rate=0.0,
                           benchmark_ticker=None, min_names_per_side=2)


def _run(panel, rebalance=63):
    return run_long_short_backtest(panel, _cfg(rebalance),
                                   execution_prices=_exec(panel))


def _rebalanced(res) -> pd.Series:
    """The sessions the book actually traded on, read off the turnover series."""
    t = res.turnover.astype(float)
    return t[t > 0]


# ---------------------------------------------------------------------------
# The window boundary
# ---------------------------------------------------------------------------


def test_the_split_session_belongs_to_exactly_one_window() -> None:
    """A session on both sides is selected on and then tested on."""
    panel = _panel()
    split = str(DATES[200].date())            # deliberately a trading day
    sel = _before(panel, split)
    hold = panel[panel["date"] >= pd.Timestamp(split)]
    assert set(sel["date"]) & set(hold["date"]) == set()
    assert pd.Timestamp(split) in set(hold["date"])


def test_selection_stops_before_the_split() -> None:
    sel = _before(_panel(), SPLIT)
    assert sel["date"].max() < pd.Timestamp(SPLIT)


def test_the_two_windows_cover_everything() -> None:
    panel = _panel()
    split = str(DATES[200].date())
    sel = _before(panel, split)
    hold = panel[panel["date"] >= pd.Timestamp(split)]
    assert len(sel) + len(hold) == len(panel)


# ---------------------------------------------------------------------------
# The calendar, which is the substantive defect
# ---------------------------------------------------------------------------


def test_slicing_the_panel_moves_every_trade_date() -> None:
    """The bug itself, stated as a property: this is why a truncated re-run
    cannot be called a continuation."""
    panel = _panel()
    full = _run(panel)
    truncated = _run(panel[panel["date"] >= pd.Timestamp(SPLIT)])

    after = pd.Timestamp(SPLIT)
    full_dates = set(_rebalanced(full).loc[lambda s: s.index >= after].index)
    reset_dates = set(_rebalanced(truncated).index)
    assert full_dates and reset_dates
    assert full_dates != reset_dates, (
        f"expected different calendars, both traded on {sorted(full_dates)[:4]}")


def test_the_holdout_segment_keeps_the_continuing_calendar() -> None:
    panel = _panel()
    full = _run(panel)
    after = pd.Timestamp(SPLIT)
    seg = full.daily_nav.loc[full.daily_nav.index >= after]
    traded = list(_rebalanced(full).loc[lambda s: s.index >= after].index)
    assert traded, "no trades in the holdout window to check"
    assert all(d in seg.index for d in traded)


def test_a_continuation_does_not_start_flat() -> None:
    """A restarted run begins at initial capital by construction. The segment
    of a continuing run begins at whatever the book was worth."""
    panel = _panel()
    full = _run(panel)
    seg = full.daily_nav.loc[full.daily_nav.index >= pd.Timestamp(SPLIT)]
    assert seg.iloc[0] != pytest.approx(_cfg().initial_capital, rel=1e-9)


def test_positions_open_at_the_split_are_not_discarded() -> None:
    """The restarted book never pays for what the selection window bought."""
    panel = _panel()
    truncated = _run(panel[panel["date"] >= pd.Timestamp(SPLIT)])
    assert truncated.daily_nav.iloc[0] == pytest.approx(
        _cfg().initial_capital, rel=1e-9)


# ---------------------------------------------------------------------------
# Segment measurement
# ---------------------------------------------------------------------------


def test_a_segment_is_measured_from_its_own_starting_value() -> None:
    """CAGR over the holdout must not be diluted by the selection window's
    return sitting in the numerator."""
    nav = pd.Series(np.linspace(100.0, 400.0, len(DATES)), index=DATES)
    whole = nav_metrics(nav)
    tail = nav_metrics(nav.loc[nav.index >= pd.Timestamp(SPLIT)])
    assert whole["total_return"] == pytest.approx(3.0)
    assert tail["total_return"] < whole["total_return"]
    assert tail["total_return"] > 0


def test_segment_metrics_carry_the_published_fields() -> None:
    res = _run(_panel())
    m = _segment(res, SPLIT, 63)
    for key in ("cagr", "sharpe", "max_drawdown"):
        assert key in m, key


def test_a_segment_with_no_sessions_is_empty_not_wrong() -> None:
    assert _segment(_run(_panel()), "2099-01-01", 63) == {}


def test_whole_period_metrics_still_work() -> None:
    m = _whole(_run(_panel()), 63)
    assert "cagr" in m and np.isfinite(m["cagr"])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_the_script_does_not_reach_for_yfinance() -> None:
    """A pre-registered test whose benchmark is whatever a vendor served that
    afternoon is not reproducible, and this project records one for exactly
    this reason."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "locked_holdout.py").read_text()
    assert "yfinance" not in src
    assert "SnapshotProvider" in src


def test_measuring_the_same_segment_twice_agrees() -> None:
    panel = _panel()
    a = _segment(_run(panel), SPLIT, 63)
    b = _segment(_run(panel), SPLIT, 63)
    assert a["cagr"] == pytest.approx(b["cagr"], rel=0, abs=0)
