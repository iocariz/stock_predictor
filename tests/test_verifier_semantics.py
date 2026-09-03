"""Five ways the checks were weaker or looser than the things they check.

None of these changes a published number. All of them are places where the
apparatus reported more confidence than it had earned.

* **Membership boundaries disagreed.** Production filters ``[start_date,
  end_date)`` (``pit.py:153``); the verifier accepted ``date <= end_date``. A
  regression that scored a company on its first *non*-member session would have
  passed the point-in-time gate.
* **The rename concurrency test could never fire.** It needs both symbols in
  the price panel, and canonicalisation renames predecessors before the panel
  is downloaded -- 0 of 15 predecessors are present. The verifier reported all
  15 as "checked" when only successor coverage had been evaluated.
* **Accounting checked one identity, not every session.** ``specs.md:414``
  requires cash, holdings and costs to reconcile *on every session*. The gate
  compared final NAV to the trade ledger and nothing in between, and the
  long-short book was skipped entirely.
* **Cohort commissions were charged at the wrong time.** The round trip is
  inside ``net_return``, credited at exit, so a cohort's NAV carried no
  commission drag while it was open. Zero in the baseline, which trades at zero
  commission; wrong as soon as anyone sets one.
* **Long-short ignored its own CLI flags.** ``--no-benchmark`` was dropped, and
  the early ``return`` meant ``--plots-dir`` and ``--compare-with`` did nothing.

Separately, replay re-detected recycled symbols from an already-cleaned
snapshot and overwrote the manifest with the empty result -- see
:mod:`tests.test_replay_provenance`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_baseline import gate_accounting, gate_pit, gate_renames  # noqa: E402

from stock_predictor.backtest import BacktestConfig, run_backtest  # noqa: E402
from stock_predictor.long_short import (  # noqa: E402
    LongShortConfig,
    run_long_short_backtest,
)
from stock_predictor.pit import filter_panel_to_pit  # noqa: E402

DATES = pd.bdate_range("2024-01-01", periods=140)
TICKERS = [f"T{i:02d}" for i in range(12)]


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, (len(DATES), len(TICKERS))),
                                axis=0))
    return pd.DataFrame([
        {"date": d, "ticker": t,
         "prob": float(np.random.default_rng(200 + di).permutation(len(TICKERS))[i]),
         "adj_close": float(px[di, i])}
        for di, d in enumerate(DATES) for i, t in enumerate(TICKERS)
    ])


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


# ---------------------------------------------------------------------------
# Membership boundary
# ---------------------------------------------------------------------------


def _stints(end: str) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "start_date": DATES[0],
                          "end_date": pd.Timestamp(end)} for t in TICKERS])


def test_production_treats_end_date_as_the_first_non_member_day() -> None:
    """The convention the verifier has to match: pit.py:153 is half-open."""
    end = DATES[100]
    kept = filter_panel_to_pit(_panel(), _stints(str(end.date())))
    assert end not in set(kept["date"]), "end_date is exclusive in production"
    assert DATES[99] in set(kept["date"])


def test_the_gate_rejects_a_row_on_its_first_non_member_day() -> None:
    """A regression scoring a company the day its membership ended used to
    pass, because the gate accepted ``date <= end_date``."""
    end = DATES[100]
    panel = _panel()
    scored = panel[panel["date"] <= end]         # one session too far
    g = gate_pit(scored, _exec(panel), 21, _stints(str(end.date())))
    assert not g.passed


def test_the_gate_accepts_the_last_real_member_session() -> None:
    end = DATES[100]
    panel = _panel()
    scored = panel[panel["date"] < end]
    g = gate_pit(scored, _exec(panel), 21, _stints(str(end.date())))
    assert g.passed, g.notes


def test_an_open_stint_is_unbounded() -> None:
    st = pd.DataFrame([{"ticker": t, "start_date": DATES[0], "end_date": pd.NaT}
                       for t in TICKERS])
    panel = _panel()
    assert gate_pit(panel, _exec(panel), 21, st).passed


# ---------------------------------------------------------------------------
# Rename concurrency
# ---------------------------------------------------------------------------


def test_the_rename_gate_does_not_claim_a_test_it_could_not_run() -> None:
    """0 of 15 predecessors are in the real panel, so the falsifier never
    fires. Saying "15 checked" overstates what was established."""
    st = pd.DataFrame([{"ticker": "ELV", "start_date": pd.Timestamp("2010-01-01"),
                        "end_date": pd.NaT, "alias": "ANTM"}])
    prices = pd.DataFrame({"ELV": 100.0}, index=DATES)   # no ANTM column
    g = gate_renames(st, prices)
    joined = " ".join(g.notes).lower()
    assert "concurren" in joined
    assert "0" in joined, "the count of testable pairs must be reported"


def test_concurrent_trading_after_the_effective_date_still_fails() -> None:
    """When both symbols *are* present the falsifier must still bite."""
    st = pd.DataFrame([{"ticker": "ELV", "start_date": pd.Timestamp("2010-01-01"),
                        "end_date": pd.NaT, "alias": "ANTM"}])
    prices = pd.DataFrame({"ELV": 100.0, "ANTM": 90.0},
                          index=pd.bdate_range("2023-01-01", periods=60))
    assert not gate_renames(st, prices).passed


# ---------------------------------------------------------------------------
# Per-session accounting
# ---------------------------------------------------------------------------


def _cfg(**kw) -> BacktestConfig:
    base = dict(top_n=4, holding_days=21, max_overlapping_cohorts=2,
                slippage_bps=5.0, benchmark_ticker=None,
                rebalance_day="Friday", reject_stale_fills=False)
    base.update(kw)
    return BacktestConfig(**base)


def test_the_engine_reports_its_ledger_every_session() -> None:
    """specs.md:414 asks for reconciliation on every session, which needs the
    per-session ledger to exist at all."""
    res = run_backtest(_panel(), _cfg(), execution_prices=_exec(_panel()))
    assert len(res.daily_cash) == len(res.daily_nav)
    assert len(res.daily_positions) == len(res.daily_nav)


def test_nav_equals_cash_plus_holdings_on_every_session() -> None:
    res = run_backtest(_panel(), _cfg(), execution_prices=_exec(_panel()))
    recomputed = res.daily_cash + res.daily_positions
    pd.testing.assert_series_equal(recomputed, res.daily_nav,
                                   check_names=False, rtol=0, atol=1e-9)


def test_the_accounting_gate_checks_every_session_not_just_the_last() -> None:
    """A curve that ends right but wanders in between used to pass."""
    res = run_backtest(_panel(), _cfg(), execution_prices=_exec(_panel()))
    assert gate_accounting(res, _cfg(), "cohort").passed

    # Break the ledger identity mid-series without moving NAV: the terminal
    # figure is untouched and the day-to-day returns are untouched, so neither
    # the final-identity check nor the implausible-jump check can see it. Only
    # a per-session reconciliation can.
    bad_positions = res.daily_positions.copy()
    bad_positions.iloc[len(bad_positions) // 2] += 500.0
    broken = res.__class__(**{**res.__dict__, "daily_positions": bad_positions})
    g = gate_accounting(broken, _cfg(), "cohort")
    assert not g.passed
    assert any("cash + holdings" in n for n in g.notes), g.notes


def test_long_short_accounting_is_gated_rather_than_skipped() -> None:
    panel = _panel()
    res = run_long_short_backtest(
        panel,
        LongShortConfig(decile=0.25, rebalance_every=21, slippage_bps=5.0,
                        benchmark_ticker=None, risk_free_rate=0.0,
                        min_names_per_side=2),
        execution_prices=_exec(panel))
    recomputed = res.daily_cash + res.daily_positions
    pd.testing.assert_series_equal(recomputed, res.daily_nav,
                                   check_names=False, rtol=0, atol=1e-6)


def test_cash_never_goes_negative_in_a_long_only_book() -> None:
    """specs.md:417 -- cash must not be driven negative by fees applied after
    sizing."""
    res = run_backtest(_panel(), _cfg(commission_per_order=5.0,
                                      commission_per_share=0.01),
                       execution_prices=_exec(_panel()))
    assert (res.daily_cash >= -1e-9).all()


# ---------------------------------------------------------------------------
# Commission timing
# ---------------------------------------------------------------------------


def test_entry_commission_is_charged_at_entry() -> None:
    """The round trip lived inside net_return and was credited only at exit,
    so an open cohort carried no commission drag at all."""
    panel = _panel()
    free = run_backtest(panel, _cfg(), execution_prices=_exec(panel))
    charged = run_backtest(panel, _cfg(commission_per_order=25.0),
                           execution_prices=_exec(panel))
    # Look at a session while the first cohort is still open.
    i = 5
    assert charged.daily_nav.iloc[i] < free.daily_nav.iloc[i], (
        "commission not visible while the position is open")


def test_zero_commission_changes_nothing() -> None:
    """Regression guard: the baseline trades at zero commission and its
    numbers must not move."""
    panel = _panel()
    a = run_backtest(panel, _cfg(), execution_prices=_exec(panel))
    b = run_backtest(panel, _cfg(commission_per_order=0.0,
                                 commission_per_share=0.0),
                     execution_prices=_exec(panel))
    pd.testing.assert_series_equal(a.daily_nav, b.daily_nav)


def test_the_round_trip_total_is_unchanged() -> None:
    """Charging earlier must move *when* the fee lands, not how much."""
    panel = _panel()
    charged = run_backtest(panel, _cfg(commission_per_order=25.0),
                           execution_prices=_exec(panel))
    closed = [c for c in charged.cohorts
              if pd.Timestamp(c.exit_date) <= charged.daily_nav.index[-1]]
    assert closed, "fixture closed no cohorts"
    recomputed = charged.daily_cash + charged.daily_positions
    assert float(recomputed.iloc[-1]) == pytest.approx(
        float(charged.daily_nav.iloc[-1]), rel=0, abs=1e-9)


# ---------------------------------------------------------------------------
# Long-short CLI flags
# ---------------------------------------------------------------------------


def test_no_benchmark_is_honoured_by_the_long_short_path() -> None:
    import inspect

    from stock_predictor import backtest

    src = inspect.getsource(backtest.main)
    ls = src.split("ls_config = LongShortConfig(")[1].split(
        "run_long_short_backtest(")[0]
    assert "args.no_benchmark" in ls, (
        "--no-benchmark is dropped on the long-short path")


def test_the_long_short_path_does_not_return_before_the_other_flags() -> None:
    import inspect

    from stock_predictor import backtest

    src = inspect.getsource(backtest.main)
    after = src.split("print_long_short_report(ls_result)")[1]
    assert "plots_dir" in after, "--plots-dir ignored after the long-short report"
