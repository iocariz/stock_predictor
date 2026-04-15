"""Calendar features: month encoding, quarter-end, FOMC proximity (statement dates)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# FOMC policy statement release dates (approx. second day for two-day meetings).
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (verify for production).
FOMC_STATEMENT_DATES: tuple[str, ...] = (
    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",
    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-30",
    "2019-12-11",
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10",
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
)

CALENDAR_FEATURE_COLS: list[str] = [
    "cal_month_sin",
    "cal_month_cos",
    "cal_is_quarter_end_month",
    "cal_days_to_calendar_qe",
    "cal_days_to_next_fomc",
    "cal_fomc_within_5d",
]


def _fomc_np() -> np.ndarray:
    return np.array(FOMC_STATEMENT_DATES, dtype="datetime64[D]")


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    fomc_dates: np.ndarray | None = None,
    fomc_cap_days: int = 60,
) -> pd.DataFrame:
    """Add month (cyclical), quarter-end calendar proximity, and FOMC countdown features."""
    out = df.copy()
    d = pd.to_datetime(out[date_col]).dt.normalize()
    month = d.dt.month.astype(float)
    out["cal_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["cal_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    out["cal_is_quarter_end_month"] = d.dt.month.isin([3, 6, 9, 12]).astype("int8")

    qe = d + pd.offsets.QuarterEnd(0)
    out["cal_days_to_calendar_qe"] = (qe - d).dt.days.astype("int16")

    fomc = fomc_dates if fomc_dates is not None else _fomc_np()
    d_np = d.values.astype("datetime64[D]")
    idx = np.searchsorted(fomc, d_np, side="left")
    past = idx >= len(fomc)
    raw_days = np.full(len(d_np), fomc_cap_days, dtype=np.int16)
    ok = ~past
    raw_days[ok] = (fomc[idx[ok]] - d_np[ok]).astype("timedelta64[D]").astype(int)
    raw_days = np.clip(raw_days, 0, fomc_cap_days).astype("int16")
    out["cal_days_to_next_fomc"] = raw_days
    out["cal_fomc_within_5d"] = (raw_days <= 5).astype("int8")

    return out
