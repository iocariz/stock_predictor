"""Nothing checked the baseline against what was published about it.

Every gate in ``verify_baseline.py`` checks the artifacts against *themselves*:
that the NAV reconciles, that fills are real, that the bytes match what was
recorded. That is a complete description of internal consistency and says
nothing about whether the numbers in BASELINE.md came from these files.

At ``c656df9`` the baseline was rebuilt with 48 reused ticker symbols removed.
New run id, new commit, every snapshot hash different, 37,156 fewer labelled
rows. The document's provenance table and survivorship section were updated;
its two results tables were not. They went on describing the artifact that had
just been replaced -- and the difference was not cosmetic:

    rank-hold CAGR   published 26.41% ± 1.46%   actual 18.83%
    rank-hold alpha  published +6.80%           actual +0.45%
    rank-hold HAC t  published +0.84            actual +0.07

The published figure was measured on the *contaminated* panel. Cleaning it
removed most of the engine's apparent return, and every gate still passed,
because none of them was looking at the published numbers.

Two things were missing and both are here:

* the figures a reader quotes are pinned next to the artifacts they came from,
  and recomputed on every verification;
* the benchmark is recorded in the snapshot, so beta, alpha and the HAC
  t-statistic can be checked at all. Verification used to pass
  ``benchmark_ticker=None``, which made exactly the numbers the conclusions
  rest on the ones nothing gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_baseline import (  # noqa: E402
    DEFAULT_TOLERANCE,
    PINNED_METRICS,
    gate_expected_metrics,
)

from stock_predictor.replay import SnapshotIncomplete, SnapshotProvider  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "artifacts" / "baseline"

MEASURED = {
    "cohort": {"metrics": {"cagr": 0.2317, "sharpe": 0.74, "max_drawdown": -0.4470,
                           "beta": 1.21, "alpha_ann": 0.0472, "alpha_t": 0.78}},
    "long-short": {"metrics": {"cagr": 0.1547, "sharpe": 1.02, "max_drawdown": -0.1202,
                               "beta": 0.18, "alpha_ann": 0.0798, "alpha_t": 2.60}},
}


def _pin(tmp_path: Path, engines: dict | None = None, **top) -> Path:
    pin = {
        "run_id": None,
        "pinned_at_utc": "2026-09-02T00:00:00+00:00",
        "pinned_at_commit": "abc123",
        "provenance": "measured-from-these-artifacts",
        "engines": engines if engines is not None else {
            k: dict(v["metrics"]) for k, v in MEASURED.items()
        },
    }
    pin.update(top)
    (tmp_path / "expected_metrics.json").write_text(json.dumps(pin))
    return tmp_path


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_matching_metrics_pass(tmp_path: Path) -> None:
    assert gate_expected_metrics(_pin(tmp_path), MEASURED).passed


def test_an_unpinned_baseline_fails(tmp_path: Path) -> None:
    """Absence is not innocence. An unpinned baseline's published figures are
    exactly as unchecked as they were before this existed."""
    g = gate_expected_metrics(tmp_path, MEASURED)
    assert not g.passed
    assert any("expected_metrics" in n for n in g.notes)


@pytest.mark.parametrize("metric", PINNED_METRICS)
def test_drift_in_any_published_metric_fails(tmp_path: Path, metric: str) -> None:
    """Each of these is quoted somewhere. Each has to be checkable."""
    drifted = {k: {"metrics": dict(v["metrics"])} for k, v in MEASURED.items()}
    drifted["cohort"]["metrics"][metric] += 10 * DEFAULT_TOLERANCE[metric]
    g = gate_expected_metrics(_pin(tmp_path), drifted)
    assert not g.passed
    assert any(f"cohort.{metric}" in n for n in g.notes)


def test_the_real_drift_would_have_been_caught(tmp_path: Path) -> None:
    """The actual event: rank-hold published at 26.41%, artifact at 18.83%."""
    pinned = {"rank-hold": {"cagr": 0.2641, "sharpe": 0.73, "max_drawdown": -0.498,
                            "beta": 1.43, "alpha_ann": 0.0680, "alpha_t": 0.84}}
    actual = {"rank-hold": {"metrics": {
        "cagr": 0.1883, "sharpe": 0.58, "max_drawdown": -0.5157,
        "beta": 1.41, "alpha_ann": 0.0045, "alpha_t": 0.07}}}
    g = gate_expected_metrics(_pin(tmp_path, engines=pinned), actual)
    assert not g.passed
    assert any("rank-hold.cagr" in n for n in g.notes)


def test_noise_within_tolerance_passes(tmp_path: Path) -> None:
    """Float summation order across library versions must not fail a build."""
    jittered = {k: {"metrics": dict(v["metrics"])} for k, v in MEASURED.items()}
    for m in jittered.values():
        for key in PINNED_METRICS:
            m["metrics"][key] += 0.4 * DEFAULT_TOLERANCE[key]
    assert gate_expected_metrics(_pin(tmp_path), jittered).passed


def test_a_pin_from_another_artifact_is_refused(tmp_path: Path) -> None:
    """This is the failure mode itself: numbers pinned to a run that is no
    longer on disk look like verification and assert nothing."""
    d = _pin(tmp_path, run_id="20260830T204011Z_f814aa3d")
    (d / "snapshot").mkdir(exist_ok=True)
    (d / "snapshot" / "manifest.json").write_text(
        json.dumps({"run_id": "20260831T085207Z_8e4abc34"}))
    g = gate_expected_metrics(d, MEASURED)
    assert not g.passed
    assert any("20260830T204011Z_f814aa3d" in n for n in g.notes)


def test_a_metric_pinned_but_unmeasurable_fails(tmp_path: Path) -> None:
    """Turning the benchmark off used to make alpha simply absent. Silently
    skipping a pinned figure is how it stayed unchecked."""
    blind = {"cohort": {"metrics": {"cagr": 0.2317, "sharpe": 0.74,
                                    "max_drawdown": -0.4470}}}
    g = gate_expected_metrics(_pin(tmp_path, engines={
        "cohort": dict(MEASURED["cohort"]["metrics"])}), blind)
    assert not g.passed
    assert any("alpha_t" in n for n in g.notes)


def test_an_engine_pinned_but_not_run_fails(tmp_path: Path) -> None:
    g = gate_expected_metrics(_pin(tmp_path), {"cohort": MEASURED["cohort"]})
    assert not g.passed
    assert any("long-short" in n for n in g.notes)


def test_an_unreadable_pin_fails(tmp_path: Path) -> None:
    (tmp_path / "expected_metrics.json").write_text("{not json")
    assert not gate_expected_metrics(tmp_path, MEASURED).passed


def test_an_empty_pin_fails(tmp_path: Path) -> None:
    assert not gate_expected_metrics(_pin(tmp_path, engines={}), MEASURED).passed


# ---------------------------------------------------------------------------
# The recorded benchmark
# ---------------------------------------------------------------------------


BENCH_DATES = pd.bdate_range("2024-01-01", periods=30)


def _snapshot(tmp_path: Path, *, benchmark: bool) -> Path:
    """A minimal replayable snapshot, optionally carrying a benchmark."""
    snap = tmp_path / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame([
        {"date": d, "ticker": t, "close": 100.0 + i + di, "volume": 1e6}
        for di, d in enumerate(BENCH_DATES) for i, t in enumerate(["AAA", "BBB"])
    ])
    long.to_parquet(snap / "equity_prices_long.parquet", index=False)
    if benchmark:
        pd.DataFrame({"date": BENCH_DATES, "ticker": "SPY",
                      "close": [float(v) for v in range(100, 130)]}).to_parquet(
            snap / "benchmark.parquet", index=False)
    (snap / "manifest.json").write_text(json.dumps({"snapshots": {}}))
    return tmp_path


def test_a_snapshot_without_a_benchmark_says_so(tmp_path: Path) -> None:
    """It must refuse rather than reach for the network -- the alternative is
    verifying against a series that moves."""
    with pytest.raises(SnapshotIncomplete, match="SPY"):
        SnapshotProvider(_snapshot(tmp_path, benchmark=False)).download_benchmark(
            "SPY", "2024-01-01", "2024-02-09")


def test_a_recorded_benchmark_is_served(tmp_path: Path) -> None:
    s = SnapshotProvider(_snapshot(tmp_path, benchmark=True)).download_benchmark(
        "SPY", "2024-01-01", "2024-02-09")
    assert len(s) == 30
    assert isinstance(s.index, pd.DatetimeIndex)
    assert float(s.iloc[0]) == 100.0


def test_the_recorded_benchmark_respects_the_window(tmp_path: Path) -> None:
    s = SnapshotProvider(_snapshot(tmp_path, benchmark=True)).download_benchmark(
        "SPY", "2024-01-15", "2024-01-31")
    assert s.index.min() >= pd.Timestamp("2024-01-15")
    assert s.index.max() <= pd.Timestamp("2024-01-31")


def test_benchmark_is_named_among_the_inputs_replay_needs(tmp_path: Path) -> None:
    from stock_predictor.replay import OPTIONAL  # noqa: PLC0415

    assert "benchmark" in OPTIONAL


# ---------------------------------------------------------------------------
# The published document
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (BASELINE / "expected_metrics.json").exists(),
                    reason="no local baseline")
def test_baseline_md_quotes_the_pinned_figures() -> None:
    """The document and the pin must not be able to drift apart again.

    BASELINE.md carries a generated block; this compares every number in it to
    ``expected_metrics.json``. Hand-maintaining a results table next to
    artifacts that get replaced is the specific thing that failed.
    """
    doc = (Path(__file__).resolve().parents[1] / "BASELINE.md").read_text()
    block = doc.split("<!-- pinned-metrics:start -->")[1].split(
        "<!-- pinned-metrics:end -->")[0]
    pin = json.loads((BASELINE / "expected_metrics.json").read_text())

    for label, expected in pin["engines"].items():
        row = next((ln for ln in block.splitlines()
                    if ln.strip().startswith(f"| {label} |")), None)
        assert row is not None, f"{label} missing from the published table"
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        got_cagr = float(cells[1].rstrip("%")) / 100
        assert got_cagr == pytest.approx(expected["cagr"], abs=5e-5), (
            f"{label}: BASELINE.md says {cells[1]}, pin says "
            f"{expected['cagr']:.2%}")
        got_t = float(cells[6])
        assert got_t == pytest.approx(expected["alpha_t"], abs=5e-3), (
            f"{label}: BASELINE.md says t {cells[6]}, pin says "
            f"{expected['alpha_t']:+.2f}")
