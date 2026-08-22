"""Archived notebooks must not display results from an invalid methodology.

Both notebooks implement, with stored outputs, the exact defects the README
documents as invalidating results:

  cell 7   sample = tickers[:SAMPLE_N]                   alphabetical truncation
  cell 9   labeled = long.dropna(subset=["fwd_ret_10d"]) unlabelled rows dropped
           labeled = filter_panel_to_pit(labeled, stints) PIT applied before
                                                          time-series features
  cell 25  TimeSeriesSplit(n_splits=TS_CV_SPLITS)        row-based and unpurged

On the same rules the truncated universe showed **+76.6% and Sharpe 1.05**
where the corrected panel shows **+22.2% and Sharpe 0.16**. Numbers of that
shape sitting in a committed notebook are not neutral history; someone opening
the repo reads them as findings.

The notebooks are kept as a record of how the project started. They keep the
code and lose the outputs, and the banner names the specific defects rather
than saying the work is merely old.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _cells(path: Path):
    return json.loads(path.read_text())["cells"]


def test_there_are_notebooks_to_check() -> None:
    assert NOTEBOOKS, "guard would pass vacuously with no notebooks"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_stored_outputs(path: Path) -> None:
    """Stored outputs are what turn invalid code into apparent results."""
    offenders = [
        i for i, c in enumerate(_cells(path))
        if c["cell_type"] == "code" and c.get("outputs")
    ]
    assert not offenders, f"{path.name} has outputs in cells {offenders}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_execution_counts(path: Path) -> None:
    """An execution count implies a run that produced numbers."""
    offenders = [
        i for i, c in enumerate(_cells(path))
        if c["cell_type"] == "code" and c.get("execution_count") is not None
    ]
    assert not offenders, f"{path.name} has execution counts in cells {offenders}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_banner_names_the_defects_not_just_the_age(path: Path) -> None:
    """"Predates the reworked pipeline" reads as "a bit old". The reader needs
    to know the results are wrong and why."""
    head = "".join(_cells(path)[0]["source"]).lower()
    assert "archived" in head
    assert "invalid" in head or "not valid" in head
    for term in ("truncat", "point-in-time", "purg"):
        assert term in head, f"{path.name} banner does not mention {term!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_the_banner_points_at_what_replaced_it(path: Path) -> None:
    head = "".join(_cells(path)[0]["source"]).lower()
    assert "train-sp500" in head or "stock_predictor" in head
