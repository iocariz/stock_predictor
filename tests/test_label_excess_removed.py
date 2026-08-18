"""label_target='excess' was mathematically a no-op (P2-3).

Rank grades are computed *within each date*. Subtracting that date's median
shifts every value by the same constant, so the ordering — and therefore the
grades — are identical to `raw`. It was shipped as a distinct, documented
"beta-neutral" option that could not differ from the default.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_predictor.training import LABEL_TARGETS, add_rank_labels, label_target_series


def _panel() -> pd.DataFrame:
    d = pd.bdate_range("2024-01-01", periods=4)
    return pd.DataFrame([
        {"date": x, "ticker": f"T{i}", "fwd_ret": (i * 7 + j) % 11 / 100,
         "vol_21d": 0.01 + i / 500}
        for j, x in enumerate(d) for i in range(10)
    ])


def test_excess_is_no_longer_offered() -> None:
    assert "excess" not in LABEL_TARGETS
    assert "excess_vol_adj" in LABEL_TARGETS, "the vol-adjusted variant is real"


def test_selecting_it_explains_why_rather_than_failing_opaquely() -> None:
    with pytest.raises(ValueError, match="identical"):
        label_target_series(_panel(), "excess")


def test_the_surviving_targets_are_genuinely_distinct() -> None:
    """Guard against shipping another no-op: every option must change grades."""
    panel = _panel()
    grades = {
        t: add_rank_labels(panel, n_grades=5, label_target=t)["rank_grade"].tolist()
        for t in LABEL_TARGETS
    }
    seen: dict[tuple, str] = {}
    for name, g in grades.items():
        key = tuple(g)
        assert key not in seen, f"{name} produces identical grades to {seen.get(key)}"
        seen[key] = name


def test_excess_vol_adj_still_differs_from_raw() -> None:
    panel = _panel()
    raw = add_rank_labels(panel, n_grades=5, label_target="raw")["rank_grade"]
    eva = add_rank_labels(panel, n_grades=5, label_target="excess_vol_adj")["rank_grade"]
    assert not raw.equals(eva), "dividing by vol does change the ordering"
