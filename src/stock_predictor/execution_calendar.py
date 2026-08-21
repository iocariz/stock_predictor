"""Trading-day calendar helpers shared by backtest and live order generation.

Historical sessions come from the data itself — the sorted set of dates present
in the OHLC/scored panel, which is ground truth for the past. *Future* sessions
come from the exchange calendar.

That distinction used to be missing: future dates were projected with
``pd.bdate_range``, which counts every weekday as a session. Measured on this
repo's own panel, **156 of 4,337 business days were not sessions — 3.6%, about
9.1 a year**. A live entry date could therefore land on a closed market, and a
63-session expiry drifted roughly two sessions early because about two holidays
fall in any given quarter.

A hand-rolled rule set does not fix this. Good Friday is a market holiday but
not a federal one; Juneteenth only became a holiday in 2022; July 4th shifts
when it falls at a weekend; the day after Thanksgiving is open on an early
close; and 2018-12-05 was an ad-hoc closure for a national day of mourning that
no rule predicts. :mod:`exchange_calendars` carries all of it and ships its
rules offline.

Early closes (1pm sessions) are deliberately not modelled: a shortened session
still has a close, and every fill here is a close. They would matter only for
intraday execution.

Entry and exit assumptions match :class:`stock_predictor.backtest.BacktestConfig`
cohort logic.
"""

from __future__ import annotations

import functools

import exchange_calendars as xcals
import numpy as np
import pandas as pd

DEFAULT_EXCHANGE = "XNYS"
"""NYSE. The universe is the S&P 500, so this is the relevant calendar."""


@functools.lru_cache(maxsize=4)
def _calendar(exchange: str = DEFAULT_EXCHANGE):
    """Cached calendar; construction parses decades of rules and is not cheap."""
    return xcals.get_calendar(exchange)


def exchange_sessions(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    exchange: str = DEFAULT_EXCHANGE,
) -> np.ndarray:
    """Real sessions in ``[start, end]``, holidays and ad-hoc closures removed."""
    sessions = _calendar(exchange).sessions_in_range(
        pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(),
    )
    return pd.DatetimeIndex(sessions).tz_localize(None).normalize().to_numpy()


def is_trading_session(
    date: str | pd.Timestamp, *, exchange: str = DEFAULT_EXCHANGE,
) -> bool:
    """Is the exchange open on *date*?"""
    return bool(_calendar(exchange).is_session(pd.Timestamp(date).normalize()))


def trading_dates_from_index(index: pd.Index) -> np.ndarray:
    """Return sorted unique normalized dates from a DataFrame index (numpy datetime64)."""
    idx = pd.DatetimeIndex(index).normalize().unique().sort_values()
    return idx.to_numpy()


def extend_calendar(
    trading_dates: np.ndarray, n_future: int, *, exchange: str = DEFAULT_EXCHANGE,
) -> np.ndarray:
    """Append the next *n_future* real exchange sessions to the known history.

    Live order generation needs entry and expiry dates beyond the downloaded
    price history, which necessarily ends today. Those are genuine sessions,
    not weekdays: an entry placed on a holiday is an order that cannot fill,
    and an expiry counted in weekdays closes the position early.
    """
    ts = pd.DatetimeIndex(trading_dates)
    if len(ts) == 0 or n_future <= 0:
        return ts.to_numpy()
    # Ask for a generous calendar span and take the first n_future sessions
    # after the last known one. Weekends and holidays cost roughly 30% of
    # calendar days, so 2x plus a margin always covers the request.
    start = ts[-1] + pd.Timedelta(days=1)
    span_days = int(n_future * 2) + 30
    future = pd.DatetimeIndex(
        exchange_sessions(start, start + pd.Timedelta(days=span_days),
                          exchange=exchange)
    )[:n_future]
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
