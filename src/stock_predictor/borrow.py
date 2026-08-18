"""Per-name short borrow rates.

A single flat borrow rate is the weakest assumption in the long-short engine,
and it is optimistic in a specific direction: this model partly ranks
volatility, so its short book is drawn from exactly the names that are
expensive to borrow. Charging general collateral on a book concentrated in
specials understates the cost.

Three ways to supply rates, in order of preference:

1. **Real data.** Put a ``borrow_rate`` column (annualized, 0.05 = 5%) on the
   scored panel. Borrow is a traded price; if you have it, nothing here beats
   it. Vendors: IHS Markit, S3 Partners, or a prime broker's locate file.
2. **A stylised proxy** — :func:`estimate_borrow_rates`. Maps each name's
   cross-sectional volatility percentile to a rate through a step schedule.
3. **A flat rate**, the previous behaviour.

The proxy exists to answer "how much does borrow concentration cost me?",
not "what is the borrow rate". Its schedule is a stylised representation of
the well-documented shape of the borrow market — a large general-collateral
mass and a thin expensive tail — not a measurement. Treat its output as a
sensitivity, and do not quote a number from it without saying so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENERAL_COLLATERAL = 0.005
"""Typical borrow for a liquid large cap, annualized."""

# (upper percentile bound, annualized rate). Most large caps sit at general
# collateral and a thin tail is dear; the shape is well documented even
# though the exact levels move constantly.
DEFAULT_BORROW_SCHEDULE: tuple[tuple[float, float], ...] = (
    (0.80, GENERAL_COLLATERAL),
    (0.90, 0.015),
    (0.95, 0.03),
    (0.99, 0.08),
    (1.00, 0.20),
)

VOL_WINDOW = 63


def realized_volatility(
    panel: pd.DataFrame,
    *,
    window: int = VOL_WINDOW,
    price_col: str = "adj_close",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Trailing realized volatility per (ticker, date), aligned to *panel*.

    Trailing by construction: the rate charged on a given day may only use
    price history up to that day.
    """
    out = panel.sort_values([ticker_col, date_col], kind="stable")
    ret = out.groupby(ticker_col, sort=False)[price_col].pct_change()
    vol = ret.groupby(out[ticker_col], sort=False).transform(
        lambda s: s.rolling(window, min_periods=max(5, window // 4)).std()
    )
    return (vol * np.sqrt(252)).reindex(panel.index)


def borrow_from_percentile(
    pct: pd.Series,
    schedule: tuple[tuple[float, float], ...] = DEFAULT_BORROW_SCHEDULE,
) -> pd.Series:
    """Map a 0-1 cross-sectional percentile to an annualized borrow rate."""
    bounds = [b for b, _ in schedule]
    rates = [r for _, r in schedule]
    idx = np.searchsorted(np.asarray(bounds), pct.to_numpy(dtype=float), side="left")
    idx = np.clip(idx, 0, len(rates) - 1)
    out = pd.Series(np.asarray(rates)[idx], index=pct.index, dtype=float)
    return out.where(pct.notna(), rates[0])


def estimate_borrow_rates(
    panel: pd.DataFrame,
    *,
    schedule: tuple[tuple[float, float], ...] = DEFAULT_BORROW_SCHEDULE,
    window: int = VOL_WINDOW,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Stylised per-name borrow rates from trailing volatility.

    Returns ``(date, ticker, borrow_rate)``. Volatility is ranked *within each
    date*, so the schedule describes the shape of the borrow market on that
    day rather than drifting with market-wide volatility — in a calm year the
    top percentile is still the top percentile.

    This is a proxy. See the module docstring.
    """
    work = panel[[date_col, ticker_col, "adj_close"]].copy()
    work["_vol"] = realized_volatility(
        work, window=window, date_col=date_col, ticker_col=ticker_col,
    )
    pct = work.groupby(date_col)["_vol"].rank(pct=True)
    work["borrow_rate"] = borrow_from_percentile(pct, schedule)
    return work[[date_col, ticker_col, "borrow_rate"]]


def resolve_borrow_rates(
    panel: pd.DataFrame,
    *,
    flat_rate: float,
    per_name: bool | pd.DataFrame = False,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame | None:
    """Pick a borrow source: panel column, supplied frame, proxy, or flat.

    Returns a ``(date, ticker, borrow_rate)`` frame, or ``None`` to mean
    "use the flat rate", which lets the engine keep its fast path.
    """
    if isinstance(per_name, pd.DataFrame):
        missing = {date_col, ticker_col, "borrow_rate"} - set(per_name.columns)
        if missing:
            raise ValueError(f"borrow rate frame missing columns: {sorted(missing)}")
        return per_name[[date_col, ticker_col, "borrow_rate"]]
    if "borrow_rate" in panel.columns:
        # Real data on the panel always wins over a proxy or a flat rate.
        return panel[[date_col, ticker_col, "borrow_rate"]]
    if per_name:
        return estimate_borrow_rates(panel, date_col=date_col, ticker_col=ticker_col)
    _ = flat_rate
    return None


def borrow_concentration(
    short_names: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> dict[str, float]:
    """How much dearer is the short book than the universe it is drawn from?

    The number that justifies this module: a ratio above 1 means a flat
    general-collateral rate understates the true cost.
    """
    merged = short_names.merge(rates, on=[date_col, ticker_col], how="left")
    book = float(merged["borrow_rate"].mean())
    universe = float(rates["borrow_rate"].mean())
    return {
        "short_book_mean_rate": book,
        "universe_mean_rate": universe,
        "concentration_ratio": book / universe if universe > 0 else float("nan"),
    }
