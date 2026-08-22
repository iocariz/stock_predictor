"""``predict-sp500`` end to end, offline.

The live path is the one that spends money, and it was the least covered part
of the repo. Every network edge is stubbed; everything else — state loading,
the kill switch, order generation, the signal report, and the dry-run/confirm
split — is the real code.

The dry-run default matters most here: a run that silently persisted state
would double-count a cohort on the next invocation.
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stock_predictor import predict as P
from stock_predictor.portfolio import init_state, load_state, save_state

N_TICKERS = 30
DATES = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=320)
TICKERS = [f"T{i:02d}" for i in range(N_TICKERS)]


class RankByTickerModel:
    """Deterministic stand-in for a fitted ranker.

    Defined at module level so it pickles; ``model_scores`` only needs
    ``predict``.
    """

    def predict(self, X):
        return np.arange(len(X), dtype=float)[::-1]


class _FakeProvider:
    def download_equity_ohlcv(self, tickers, start, end):
        rng = np.random.default_rng(1)
        keep = [t for t in tickers if t in TICKERS]
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(DATES), len(keep))), axis=0))
        return (pd.DataFrame(px, index=DATES, columns=keep),
                pd.DataFrame(1e6, index=DATES, columns=keep))

    def download_macro(self, start, end):
        return pd.DataFrame({"date": DATES, "vix": 15.0,
                             "tnx_yield": 3.0, "irx_yield": 4.5})

    def download_benchmark(self, ticker, start, end):
        return pd.Series(1.0, index=DATES, name=ticker)


def _stints() -> pd.DataFrame:
    return pd.DataFrame({"ticker": TICKERS,
                         "start_date": [pd.Timestamp("2014-01-01")] * N_TICKERS,
                         "end_date": [pd.NaT] * N_TICKERS})


def _sectors() -> pd.DataFrame:
    return pd.DataFrame({"ticker": TICKERS,
                         "sector": ["Tech", "Health", "Energy"] * (N_TICKERS // 3)})


@pytest.fixture
def rig(tmp_path):
    """A model whose feature list matches what the live panel actually builds."""
    from stock_predictor.predict import build_inference_panel

    provider = _FakeProvider()
    adj, vol = provider.download_equity_ohlcv(TICKERS, None, None)
    with patch("stock_predictor.training.download_sector_map", return_value=_sectors()):
        panel, cols = build_inference_panel(
            adj, vol, _stints(), start=str(DATES[0].date()), end=None,
            skip_earnings=True, provider=provider, macro_merge=False,
        )
    model_path = tmp_path / "m.pkl"
    with open(model_path, "wb") as fh:
        # train_end is not optional in practice — build_model_meta always
        # writes it — and the freshness gate treats an unknown age as stale.
        pickle.dump({"model": RankByTickerModel(),
                     "meta": {"feature_cols": cols, "horizon": 21,
                              "seed": 42, "sample_n": N_TICKERS,
                              "train_end": str(DATES[-1].date())}}, fh)
    state_path = tmp_path / "state.json"
    save_state(init_state(100_000.0), state_path)
    return model_path, state_path


def _run(rig, *extra: str):
    model_path, state_path = rig
    argv = ["predict-sp500", "--model", str(model_path), "--state", str(state_path),
            "--sample-n", str(N_TICKERS), "--skip-earnings", "--no-macro-merge",
            "--min-coverage", "0", "--top-n", "5", *extra]
    with (
        patch("sys.argv", argv),
        patch.object(P, "get_provider", return_value=_FakeProvider()),
        patch.object(P, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
    ):
        P.main()
    return state_path


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_a_signal_is_produced_end_to_end(rig, capsys) -> None:
    _run(rig)
    out = capsys.readouterr().out
    assert "DAILY SIGNAL" in out
    assert "Scored" in out


def test_the_default_is_a_dry_run_that_does_not_touch_state(rig, capsys) -> None:
    """Persisting without --confirm would double-count the cohort next run."""
    model_path, state_path = rig
    before = state_path.read_text()
    _run(rig)
    assert state_path.read_text() == before
    assert "Dry run" in capsys.readouterr().out


def test_confirm_persists_the_new_state(rig) -> None:
    model_path, state_path = rig
    _run(rig, "--confirm", "--force-rebalance")
    after = load_state(state_path)
    assert len(after.positions) > 0, "a confirmed run must open the picks it printed"
    assert after.cash < 100_000.0


def test_rank_hold_mode_also_runs(rig, capsys) -> None:
    _run(rig, "--hold-mode", "rank", "--force-rebalance")
    assert "DAILY SIGNAL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_a_missing_state_file_exits_with_guidance(rig, tmp_path) -> None:
    model_path, _ = rig
    argv = ["predict-sp500", "--model", str(model_path),
            "--state", str(tmp_path / "nope.json")]
    with patch("sys.argv", argv), pytest.raises(SystemExit) as exc:
        P.main()
    assert "--init" in str(exc.value)


def test_init_creates_a_portfolio(tmp_path, capsys) -> None:
    state_path = tmp_path / "new.json"
    with patch("sys.argv", ["predict-sp500", "--init", "--state", str(state_path),
                            "--initial-capital", "50000"]):
        P.main()
    assert state_path.exists()
    assert json.loads(state_path.read_text())["cash"] == 50_000.0


def test_a_universe_mismatch_is_warned_about_not_silent(rig, capsys) -> None:
    """The model was fitted on a 30-name draw; asking for 20 changes every
    cross-sectional feature it reads."""
    _run(rig, "--sample-n", "20")
    assert "differs from training" in capsys.readouterr().err


def test_model_is_required_for_anything_but_init(tmp_path) -> None:
    """--model stopped being argparse-required so `--init` alone works; main
    must still refuse a scoring run without one."""
    state_path = tmp_path / "s.json"
    save_state(init_state(100_000.0), state_path)
    with patch("sys.argv", ["predict-sp500", "--state", str(state_path)]), \
         pytest.raises(SystemExit) as exc:
        P.main()
    assert "--model is required" in str(exc.value)


# ---------------------------------------------------------------------------
# Unrankable names are excluded from the live decision
# ---------------------------------------------------------------------------


class _HoleyProvider(_FakeProvider):
    """A vendor that stops quoting T00 for the last three weeks.

    Exactly what happened to AVB and EQR: continuous index members, fully
    "covered" by the download check, present on 1 of the last 20 sessions.
    """

    def download_equity_ohlcv(self, tickers, start, end):
        adj, vol = super().download_equity_ohlcv(tickers, start, end)
        if "T00" in adj.columns:
            adj.iloc[-19:, adj.columns.get_loc("T00")] = np.nan
        return adj, vol


def _run_holey(rig, *extra: str):
    model_path, state_path = rig
    argv = ["predict-sp500", "--model", str(model_path), "--state", str(state_path),
            "--sample-n", str(N_TICKERS), "--skip-earnings", "--no-macro-merge",
            "--min-coverage", "0", "--top-n", "5", *extra]
    with (
        patch("sys.argv", argv),
        patch.object(P, "get_provider", return_value=_HoleyProvider()),
        patch.object(P, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
    ):
        P.main()


def test_a_name_with_a_recent_hole_is_not_ranked(rig, capsys) -> None:
    """T00 is the model's top pick by construction, so if it still appears the
    guard did not reach the decision."""
    _run_holey(rig, "--force-rebalance")
    out = capsys.readouterr()
    assert "Excluding 1 name(s)" in out.err
    assert "T00" in out.err
    picks = out.out.split("NEW PICKS")[-1]
    assert "T00" not in picks, "an unrankable name must not be bought"


def test_the_guard_can_be_switched_off(rig, capsys) -> None:
    """Deliberately, and visibly — not by accident."""
    _run_holey(rig, "--force-rebalance", "--min-recent-coverage", "0")
    assert "Excluding" not in capsys.readouterr().err


def test_a_clean_panel_excludes_nothing(rig, capsys) -> None:
    _run(rig, "--force-rebalance")
    assert "Excluding" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Stale inputs block the run
# ---------------------------------------------------------------------------


def _stale_model(tmp_path, train_end: str):
    """A model identical to the rig's, but trained long ago."""
    from stock_predictor.predict import build_inference_panel

    provider = _FakeProvider()
    adj, vol = provider.download_equity_ohlcv(TICKERS, None, None)
    with patch("stock_predictor.training.download_sector_map", return_value=_sectors()):
        _, cols = build_inference_panel(
            adj, vol, _stints(), start=str(DATES[0].date()), end=None,
            skip_earnings=True, provider=provider, macro_merge=False,
        )
    p = tmp_path / "stale.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"model": RankByTickerModel(),
                     "meta": {"feature_cols": cols, "horizon": 21, "seed": 42,
                              "sample_n": N_TICKERS, "train_end": train_end}}, fh)
    return p


def test_a_stale_model_blocks_the_run(rig, tmp_path, capsys) -> None:
    """The real incident: a model 3.6 years past its training data, used to
    pick trades, with nothing in the code path objecting."""
    _, state_path = rig
    model = _stale_model(tmp_path, "2015-01-01")
    argv = ["predict-sp500", "--model", str(model), "--state", str(state_path),
            "--sample-n", str(N_TICKERS), "--skip-earnings", "--no-macro-merge",
            "--min-coverage", "0"]
    with (
        patch("sys.argv", argv),
        patch.object(P, "get_provider", return_value=_FakeProvider()),
        patch.object(P, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
        pytest.raises(SystemExit) as exc,
    ):
        P.main()
    assert "stale" in str(exc.value).lower()
    assert "model_age" in capsys.readouterr().err


def test_a_stale_model_can_be_overridden_visibly(rig, tmp_path, capsys) -> None:
    _, state_path = rig
    model = _stale_model(tmp_path, "2015-01-01")
    argv = ["predict-sp500", "--model", str(model), "--state", str(state_path),
            "--sample-n", str(N_TICKERS), "--skip-earnings", "--no-macro-merge",
            "--min-coverage", "0", "--allow-stale"]
    with (
        patch("sys.argv", argv),
        patch.object(P, "get_provider", return_value=_FakeProvider()),
        patch.object(P, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
    ):
        P.main()
    err = capsys.readouterr().err
    assert "model_age" in err and "proceeding anyway" in err


def test_a_fresh_model_is_not_blocked(rig, capsys) -> None:
    """A recent train_end passes the default policy without any override."""
    _run(rig)
    assert "DAILY SIGNAL" in capsys.readouterr().out


def test_the_gate_fires_before_any_order_is_generated(rig, tmp_path, capsys) -> None:
    """Blocking after printing a signal would invite acting on it."""
    _, state_path = rig
    model = _stale_model(tmp_path, "2015-01-01")
    argv = ["predict-sp500", "--model", str(model), "--state", str(state_path),
            "--sample-n", str(N_TICKERS), "--skip-earnings", "--no-macro-merge",
            "--min-coverage", "0", "--confirm"]
    before = state_path.read_text()
    with (
        patch("sys.argv", argv),
        patch.object(P, "get_provider", return_value=_FakeProvider()),
        patch.object(P, "load_sp500_stints", return_value=_stints()),
        patch("stock_predictor.training.download_sector_map", return_value=_sectors()),
        pytest.raises(SystemExit),
    ):
        P.main()
    assert "DAILY SIGNAL" not in capsys.readouterr().out
    assert state_path.read_text() == before, "a blocked run must not touch state"


# ---------------------------------------------------------------------------
# The kill switch outranks --force-rebalance
# ---------------------------------------------------------------------------


def _halted_state(state_path, capital: float = 100_000.0):
    """A book whose high-water mark is far above its current value."""
    from stock_predictor.portfolio import PortfolioState
    save_state(PortfolioState(initial_capital=capital, cash=capital,
                              high_watermark=capital * 4), state_path)


def test_force_rebalance_cannot_trade_through_the_kill_switch(rig, capsys) -> None:
    """`allow_buys` and `force` reach generate_orders independently, so this
    is the seam where a precedence bug in their combination shows up."""
    _, state_path = rig
    _halted_state(state_path)
    _run(rig, "--force-rebalance", "--confirm", "--max-drawdown", "0.15")

    out = capsys.readouterr().out
    assert "HALTED" in out
    assert "KILL-SWITCH ENGAGED" in out
    assert "NEW PICKS" not in out
    assert "while halted" not in out, "no orders should have been generated at all"
    assert load_state(state_path).positions == (), "a halted book bought anyway"


def test_a_healthy_book_still_honours_force_rebalance(rig) -> None:
    """The fix must not disarm force for its actual purpose."""
    _, state_path = rig
    _run(rig, "--force-rebalance", "--confirm")
    assert len(load_state(state_path).positions) > 0
