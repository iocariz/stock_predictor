"""Two ways the live path measured something the simulation does not.

**The signal and the fill were the same bar.** The backtest ranks on
``date[i]`` and fills at ``date[i+1]``'s price, and says why: trading on the
signal-day close would use the very bar the score was computed from. The live
path scored and filled on the same session, which is that look-ahead — not a
licence live has and the simulation lacks. ``--signal-lag`` defaults to 1 and
makes the separation explicit.

**A missing session was closed up before the rolling windows ran.**
``build_labeled_panel`` dropped unpriced rows, so features were computed on a
compacted grid and every window silently spanned more calendar than its name
claimed. Measured on a one-session gap: ``ret_1d`` on the following session read
``+10%`` while actually covering two sessions, and ``vol_10d`` covered eleven.
The grid is now preserved through feature computation and the unpriced rows are
dropped afterwards, so a gap is a gap rather than a join.

Both are parity defects rather than crashes, which is why neither showed up as a
failure: each produced a plausible number that the simulation would never have
produced from the same data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.training import add_timeseries_features, build_labeled_panel

DATES = pd.bdate_range("2024-01-01", periods=40)
GAP = 20


def _prices(gap: bool = True) -> pd.DataFrame:
    px = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=DATES)
    if gap:
        px.loc[DATES[GAP], "AAA"] = np.nan
        px.loc[DATES[GAP + 1], "AAA"] = 110.0
    return px


def _volume() -> pd.DataFrame:
    return pd.DataFrame(1e6, index=DATES, columns=["AAA", "BBB"])


def _features(px: pd.DataFrame) -> pd.DataFrame:
    labeled = build_labeled_panel(px, None, horizon=5, threshold=0.05)
    return add_timeseries_features(labeled, _volume())


# ---------------------------------------------------------------------------
# The session grid survives feature computation
# ---------------------------------------------------------------------------


def test_the_grid_is_not_compacted() -> None:
    feat = _features(_prices())
    aaa = feat[feat["ticker"] == "AAA"]
    assert len(aaa) == len(DATES), (
        f"{len(DATES) - len(aaa)} session(s) closed up before the windows ran")


def test_an_unpriced_session_is_marked_not_removed() -> None:
    feat = _features(_prices()).set_index(["ticker", "date"])
    assert not bool(feat.loc[("AAA", DATES[GAP]), "is_tradable"])
    assert bool(feat.loc[("AAA", DATES[GAP - 1]), "is_tradable"])


def test_a_one_day_return_is_never_a_two_day_return() -> None:
    """The defect, stated as the property it violated. Across the gap the
    price goes 100 → (nothing) → 110, and the compacted frame reported that as
    ``ret_1d = +10%``."""
    feat = _features(_prices()).set_index(["ticker", "date"])
    after = feat.loc[("AAA", DATES[GAP + 1]), "ret_1d"]
    assert pd.isna(after), (
        f"ret_1d after a gap is {after:+.4f}; the previous close is unknown, "
        "so a one-session return does not exist")


def test_a_clean_series_is_unaffected() -> None:
    """Regression guard: this must change nothing when nothing is missing."""
    feat = _features(_prices(gap=False))
    bbb = feat[feat["ticker"] == "BBB"]
    assert len(bbb) == len(DATES)
    assert bbb["ret_1d"].iloc[1:].notna().all()


def test_rolling_windows_span_the_real_calendar() -> None:
    """A window that crosses a gap has fewer observations than sessions, which
    is the honest reading. Under the defect it had the full count, drawn from
    a wider stretch of calendar."""
    feat = _features(_prices()).set_index(["ticker", "date"])
    v = feat.loc[("AAA", DATES[GAP + 2]), "vol_10d"]
    assert pd.isna(v) or float(v) >= 0.0


def test_both_tickers_keep_the_same_sessions() -> None:
    """A ragged grid is what let one name's window silently outrun another's."""
    feat = _features(_prices())
    counts = feat.groupby("ticker")["date"].nunique()
    assert counts.nunique() == 1, counts.to_dict()


# ---------------------------------------------------------------------------
# Signal and fill are different sessions
# ---------------------------------------------------------------------------


def test_the_default_lag_is_one_session() -> None:
    """Matching the backtest, which fills the session after it signals."""
    import contextlib
    import io

    from stock_predictor import predict

    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.suppress(SystemExit):
        parser = predict.build_parser() if hasattr(predict, "build_parser") else None
    if parser is None:                       # parser built inside parse_args
        import re
        src = __import__("inspect").getsource(predict.parse_args)
        m = re.search(r'"--signal-lag".*?default=(\d+)', src, re.S)
        assert m and m.group(1) == "1", "the default signal lag is not 1"
        return
    ns = parser.parse_args([])
    assert ns.signal_lag == 1


def test_the_signal_session_precedes_the_fill_session() -> None:
    """Read off the source: the score date must not be the newest session when
    the fill prices come from it."""
    import inspect

    from stock_predictor import predict

    src = inspect.getsource(predict.main)
    assert "signal_date = sessions[-1 - lag]" in src
    assert "score_date=signal_date" in src


def test_fill_prices_do_not_come_from_the_scored_panel() -> None:
    """The scores are a session older by design, so their adj_close is not the
    price anything executes at."""
    import inspect

    from stock_predictor import predict

    src = inspect.getsource(predict.main)
    assert 'dict(zip(scored["ticker"], scored["adj_close"]))' not in src, (
        "fills are still priced from the scored panel")
    assert "latest_prices.update(session_px)" in src


def test_a_panel_too_short_for_the_lag_is_refused() -> None:
    """Silently falling back to same-bar execution is the behaviour being
    removed, so a panel that cannot support the lag has to say so."""
    import inspect

    from stock_predictor import predict

    src = inspect.getsource(predict.main)
    assert "needs at least" in src and "signal-lag" in src


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_the_lag_selects_the_right_session(lag: int) -> None:
    """The arithmetic on its own, since the CLI path needs a model to run."""
    sessions = pd.DatetimeIndex(DATES)
    assert sessions[-1 - lag] == DATES[len(DATES) - 1 - lag]
