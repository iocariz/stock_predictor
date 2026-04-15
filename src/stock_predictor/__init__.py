"""S&P 500 forward-return research toolkit."""

__version__ = "0.1.0"

from stock_predictor.calendar_features import (
    CALENDAR_FEATURE_COLS,
    FOMC_STATEMENT_DATES,
    add_calendar_features,
)
from stock_predictor.pit import (
    SP500_STINTS_URL,
    filter_panel_to_pit,
    load_sp500_stints,
    tickers_overlapping_window,
)

__all__ = [
    "__version__",
    "CALENDAR_FEATURE_COLS",
    "FOMC_STATEMENT_DATES",
    "SP500_STINTS_URL",
    "add_calendar_features",
    "filter_panel_to_pit",
    "load_sp500_stints",
    "tickers_overlapping_window",
]
