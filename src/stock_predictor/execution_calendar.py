"""Trading-day calendar helpers shared by backtest and live order generation.

The calendar is the sorted set of session dates present in the OHLC/scored
panel (not a full exchange holiday calendar). Entry and exit assumptions match
:class:`stock_predictor.backtest.BacktestConfig` cohort logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trading_dates_from_index(index: pd.Index) -> np.ndarray:
    """Return sorted unique normalized dates from a DataFrame index (numpy datetime64)."""
    idx = pd.DatetimeIndex(index).normalize().unique().sort_values()
    return idx.to_numpy()


def extend_calendar(trading_dates: np.ndarray, n_future: int) -> np.ndarray:
    """Append *n_future* business days after the last known session.

    Live order generation needs entry/expiry sessions that lie beyond the
    downloaded price history (which necessarily ends "today").  Future
    sessions are approximated with business days — an exchange holiday can
    shift the true exit by a session, which is harmless because expiries are
    settled by ``expiry_date <= as_of`` on a later run.
    """
    ts = pd.DatetimeIndex(trading_dates)
    if len(ts) == 0 or n_future <= 0:
        return ts.to_numpy()
    future = pd.bdate_range(ts[-1] + pd.Timedelta(days=1), periods=n_future)
    return ts.append(future).to_numpy()


def next_trading_day(date: pd.Timestamp, trading_dates: np.ndarray) -> pd.Timestamp | None:
    """First session strictly after *date*."""
    ts = pd.DatetimeIndex(trading_dates)
    idx = ts.searchsorted(pd.Timestamp(date).normalize(), side="right")
    if idx >= len(ts):
        return None
    return pd.Timestamp(ts[idx])


def offset_trading_days(
    date: pd.Timestamp, n: int, trading_dates: np.ndarray,
) -> pd.Timestamp | None:
    """Session *n* trading days after *date* (same convention as the backtest exit)."""
    ts = pd.DatetimeIndex(trading_dates)
    idx = ts.searchsorted(pd.Timestamp(date).normalize(), side="left")
    target = idx + n
    if target >= len(ts):
        return None
    return pd.Timestamp(ts[target])


def entry_on_or_after(as_of: str | pd.Timestamp, trading_dates: np.ndarray) -> pd.Timestamp | None:
    """First trading session on or after *as_of* (normalized), for assumed fill day."""
    d = pd.Timestamp(as_of).normalize()
    ts = pd.DatetimeIndex(trading_dates)
    i = ts.searchsorted(d, side="left")
    if i >= len(ts):
        return None
    return pd.Timestamp(ts[i])


def exit_date_iso_after_hold(
    entry: pd.Timestamp,
    holding_days: int,
    trading_dates: np.ndarray,
) -> str | None:
    """ISO date (YYYY-MM-DD) of the exit session *holding_days* after *entry*."""
    exit_ts = offset_trading_days(entry, holding_days, trading_dates)
    if exit_ts is None:
        return None
    return exit_ts.strftime("%Y-%m-%d")
