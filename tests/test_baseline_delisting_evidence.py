"""The baseline has to be measured with what delisted names actually paid.

Every disposal the engines made on the recorded baseline was an acquisition:
TSS, VIAB, AGN, NBL, CXO, TIF, FLIR, VAR, CERN, JNPR. None of them went to
zero. With no evidence the policy settles a dark long at zero once its grace
period ends, so the long-short book booked four total losses -- VIAB, NBL,
CXO, CERN -- on names whose holders were paid in cash or in the acquirer's
stock. The evidence mechanism existed and the live book used it; the baseline
was measured without it.

Evidence is recorded *with* the baseline, like the benchmark and the
vendor-absent set, so the verdict on fixed artifacts does not depend on what
the repo's data file says later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_baseline import (  # noqa: E402
    EVIDENCE_FILE,
    load_baseline_evidence,
    measure_engines,
)

from stock_predictor.delisting import load_proceeds  # noqa: E402

DATES = pd.bdate_range("2024-01-01", periods=140)
N = 60
DARK = "T00"
GOES_DARK = 45
ACQUIRED = ("TSS", "VIAB", "AGN", "NBL", "CXO", "TIF", "FLIR", "VAR", "CERN",
            "JNPR", "DISCK",
            # The rebuild with corrupt prints quarantined is a different draw of
            # the model and held different names as they left the index.
            "ALXN", "NLSN", "TWTR", "ANSS", "HES", "DAY", "HOLX")


def _panel() -> pd.DataFrame:
    """Sixty flat names; the best-ranked one stops printing for good."""
    rows = []
    for di, d in enumerate(DATES):
        for i in range(N):
            t = f"T{i:02d}"
            if t == DARK and di >= GOES_DARK:
                continue
            rows.append({"date": d, "ticker": t, "prob": float(N - i),
                         "adj_close": 100.0})
    return pd.DataFrame(rows)


def _exec(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(index="date", columns="ticker",
                             values="adj_close", aggfunc="first").reindex(DATES)


def _final_navs(evidence: pd.DataFrame | None) -> dict[str, float]:
    panel = _panel()
    out = measure_engines(
        panel, _exec(panel), provider=None, top_n=3, horizon=20, max_cohorts=2,
        slippage_bps=0.0, rebalance_day="Friday", exit_rank=6,
        delisting_proceeds=evidence,
    )
    return {k: float(v["result"].daily_nav.iloc[-1]) for k, v in out.items()}


def _paid_in_full() -> pd.DataFrame:
    return pd.DataFrame({"ticker": [DARK], "date": [DATES[GOES_DARK]],
                         "proceeds": [100.0], "source": ["test"]})


# --- the measurement ----------------------------------------------------------


def test_without_evidence_a_bought_out_long_is_a_total_loss() -> None:
    """The behaviour being corrected, pinned so the next test means something."""
    navs = _final_navs(None)
    assert all(nav < 100_000 * 0.99 for nav in navs.values()), navs


def test_evidence_reaches_all_three_engines() -> None:
    """Nothing lost value on this panel: the dark name was paid 100 for a
    position marked at 100. Each engine must end where a clean run ends."""
    without = _final_navs(None)
    paid = _final_navs(_paid_in_full())
    for engine, nav in paid.items():
        assert nav > without[engine], f"{engine} ignored the evidence"
        # Long-short earns the cash rate on its balance; the long-only engines
        # on this flat panel end exactly where they started.
        assert nav >= 100_000 * 0.999, (engine, nav)


# --- recorded with the baseline -----------------------------------------------


def test_a_baseline_without_evidence_reports_none(tmp_path: Path) -> None:
    assert load_baseline_evidence(tmp_path) is None


def test_evidence_is_read_from_the_baseline_directory(tmp_path: Path) -> None:
    _paid_in_full().to_csv(tmp_path / EVIDENCE_FILE, index=False)
    frame = load_baseline_evidence(tmp_path)
    assert frame is not None and list(frame["ticker"]) == [DARK]


def test_malformed_evidence_is_an_error_not_no_evidence(tmp_path: Path) -> None:
    """Silently measuring without it is how this went unnoticed."""
    (tmp_path / EVIDENCE_FILE).write_text("ticker,price\nT00,100\n")
    with pytest.raises(ValueError, match="proceeds"):
        load_baseline_evidence(tmp_path)


# --- the committed file -------------------------------------------------------


def _committed() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "delisting_proceeds.csv")


def test_every_acquisition_the_baseline_disposes_of_has_a_row() -> None:
    evidence = load_proceeds(_committed())
    assert [t for t in ACQUIRED if t not in evidence] == []


def test_every_row_says_where_it_came_from() -> None:
    frame = _committed()
    assert frame["ticker"].is_unique
    assert (frame["proceeds"] > 0).all()
    assert frame["source"].astype(str).str.len().gt(20).all()
    assert pd.to_datetime(frame["date"]).notna().all()
