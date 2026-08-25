"""An entry decision cannot depend on the exit quote.

``_build_cohort`` priced the entry *and* the exit at construction time, on the
signal date, and dropped any name whose exit quote was missing or stale. The
exit is ``holding_days`` sessions in the future, so a position that was
perfectly enterable on the entry date vanished because of something that had
not happened yet:

    complete data: cohort entered A on 2024-01-03
    missing exit:  the entire cohort disappeared

That is look-ahead, and it flatters the result twice over — the excluded names
are disproportionately the ones that stopped being quoted, which is to say the
failures. Whether a name can be *bought* on the entry date and what happens
when it cannot be *sold* on the exit date are separate questions, answered at
separate times.

Rank-hold already resolves the second one properly: reject the fill, defer, and
dispose by explicit evidence or a named policy after a grace period
(``specs.md:181``, ``:249``, ``:587``). The cohort engine now uses the same
machinery instead of quietly declining to have opened the position.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.backtest import BacktestConfig, run_backtest
from stock_predictor.delisting import DelistingPolicy

DATES = pd.bdate_range("2024-01-01", periods=40)
TICKERS = ["A", "B", "C", "D", "E", "F"]
HOLD = 5


def _panel(drop: list[tuple[pd.Timestamp, str]] | None = None) -> pd.DataFrame:
    rows = [
        {"date": d, "ticker": t, "prob": float(len(TICKERS) - i),
         "adj_close": 100.0 + i, "fwd_ret": 0.0, "target_5pct": 1.0}
        for d in DATES for i, t in enumerate(TICKERS)
    ]
    df = pd.DataFrame(rows)
    for when, who in (drop or []):
        df = df[~((df["date"] == when) & (df["ticker"] == who))]
    return df


def _cfg(**kw) -> BacktestConfig:
    base = dict(top_n=1, holding_days=HOLD, max_overlapping_cohorts=1,
                slippage_bps=0.0, rebalance_day="Tuesday",
                benchmark_ticker=None, reject_stale_fills=True)
    base.update(kw)
    return BacktestConfig(**base)


def _first(res):
    return res.cohorts[0] if res.cohorts else None


@pytest.fixture
def baseline():
    res = run_backtest(_panel(), _cfg())
    c = _first(res)
    assert c is not None and c.tickers == ("A",)
    return res, c


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


def test_a_missing_future_exit_quote_does_not_prevent_the_entry(baseline) -> None:
    """The reported reproduction, stated as a requirement."""
    _, c0 = baseline
    res = run_backtest(
        _panel(drop=[(pd.Timestamp(c0.exit_date), "A")]), _cfg(),
    )
    entered = [c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)]
    assert entered, "the entry was valid on its own date and must still happen"
    assert "A" in entered[0].tickers


def test_the_cohort_count_is_unchanged_by_a_future_gap(baseline) -> None:
    res_full, c0 = baseline
    res_gap = run_backtest(
        _panel(drop=[(pd.Timestamp(c0.exit_date), "A")]), _cfg(),
    )
    n_full = len({pd.Timestamp(c.entry_date) for c in res_full.cohorts})
    n_gap = len({pd.Timestamp(c.entry_date) for c in res_gap.cohorts})
    assert n_gap == n_full


# ---------------------------------------------------------------------------
# Entry-date information is still enforced
# ---------------------------------------------------------------------------


def test_a_missing_entry_quote_still_rejects(baseline) -> None:
    """Not look-ahead: you cannot buy at a price that did not print."""
    _, c0 = baseline
    res = run_backtest(
        _panel(drop=[(pd.Timestamp(c0.entry_date), "A")]), _cfg(),
    )
    entered = [c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)]
    assert not entered or "A" not in entered[0].tickers


def test_complete_data_is_unaffected(baseline) -> None:
    """Regression guard: the fix must not move results when nothing is missing."""
    res_full, _ = baseline
    again = run_backtest(_panel(), _cfg())
    assert [c.tickers for c in again.cohorts] == [c.tickers for c in res_full.cohorts]
    assert again.daily_nav.iloc[-1] == pytest.approx(res_full.daily_nav.iloc[-1])


# ---------------------------------------------------------------------------
# The exit is resolved later, the way rank-hold resolves it
# ---------------------------------------------------------------------------


def test_a_deferred_exit_settles_at_the_next_real_quote(baseline) -> None:
    _, c0 = baseline
    exit_date = pd.Timestamp(c0.exit_date)
    res = run_backtest(_panel(drop=[(exit_date, "A")]), _cfg())
    leg = next(c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)
               and "A" in c.tickers)
    assert pd.Timestamp(leg.exit_date) > exit_date, "it sold later, not never"


def test_a_deferred_exit_is_reported(baseline) -> None:
    """specs.md:249 — missing exits must appear in the diagnostics."""
    _, c0 = baseline
    res = run_backtest(_panel(drop=[(pd.Timestamp(c0.exit_date), "A")]), _cfg())
    assert res.metrics.get("exits_deferred", 0) >= 1


def test_a_name_that_never_prints_again_is_written_off_not_sold_at_entry(
    baseline,
) -> None:
    """A gap is not proof of a delisting, but it cannot be ignored forever
    either. After the grace period the conservative fallback applies."""
    _, c0 = baseline
    exit_date = pd.Timestamp(c0.exit_date)
    gone = [(d, "A") for d in DATES if d >= exit_date]
    res = run_backtest(
        _panel(drop=gone),
        _cfg(delisting_policy=DelistingPolicy(fallback="write_off",
                                              grace_sessions=3)),
    )
    leg = next(c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)
               and "A" in c.tickers)
    assert leg.exit_prices[0] == pytest.approx(0.0), "written off, not exited flat"
    assert leg.net_return == pytest.approx(-1.0)


def test_explicit_delisting_evidence_is_used(baseline) -> None:
    _, c0 = baseline
    exit_date = pd.Timestamp(c0.exit_date)
    gone = [(d, "A") for d in DATES if d >= exit_date]
    evidence = pd.DataFrame({"ticker": ["A"], "date": [exit_date],
                             "proceeds": [58.5]})
    res = run_backtest(
        _panel(drop=gone), _cfg(), delisting_proceeds=evidence,
    )
    leg = next(c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)
               and "A" in c.tickers)
    assert leg.exit_prices[0] == pytest.approx(58.5)


def test_holding_indefinitely_leaves_the_capital_visibly_stuck(baseline) -> None:
    _, c0 = baseline
    exit_date = pd.Timestamp(c0.exit_date)
    gone = [(d, "A") for d in DATES if d >= exit_date]
    res = run_backtest(
        _panel(drop=gone), _cfg(delisting_policy=DelistingPolicy(fallback="hold")),
    )
    leg = next(c for c in res.cohorts
               if pd.Timestamp(c.entry_date) == pd.Timestamp(c0.entry_date)
               and "A" in c.tickers)
    assert pd.Timestamp(leg.exit_date) > DATES[-1], "never settled, and it shows"
