"""A run's scores and its execution prices must belong to each other.

``run_pipeline.sh`` defaults ``EXECUTION_PRICES`` to a parquet path and uses it
*if the file exists* — but nothing in the package, the scripts, or CI ever wrote
that file. Where it existed at all it was a leftover from an ad-hoc fetch, and
the pipeline paired it with freshly generated scores without comment:

    scores            409 sessions, 2025-01-02 -> 2026-08-20
    execution prices 4181 sessions, 2010-01-04 -> 2026-08-18

Two scored sessions had no execution row. Absence degraded just as quietly: the
backtest fell back to forward-filled prices, which on the rank-hold engine is
the difference between +17.28% and +22.95% — a gap that looks like a strategy
result and is really a missing file.

So coverage is checked rather than assumed. The execution panel is expected to
be *wider and longer* than the scored panel — it is the full unfiltered
download, while the scored panel is point-in-time filtered — so only the
reverse, a scored row with no execution row, is a fault.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype


class BundleMismatch(RuntimeError):
    """The scores and the execution prices do not describe the same run."""


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str


def _preview(items, n: int = 6) -> str:
    shown = ", ".join(str(x) for x in list(items)[:n])
    return shown + (" …" if len(list(items)) > n else "")


def validate_execution_panel(
    scored: pd.DataFrame,
    execution: pd.DataFrame | None,
    *,
    strict: bool = False,
    max_missing_sessions: int = 0,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> list[Finding]:
    """Check that *execution* can price every row of *scored*.

    *max_missing_sessions* tolerates a known settlement lag — a deliberate
    decision rather than an accident. Set *strict* to raise instead of
    returning findings.
    """
    findings: list[Finding] = []

    if execution is None or not len(execution):
        findings.append(Finding(
            "missing",
            "no execution price panel; fills will fall back to forward-filled "
            "quotes from the point-in-time scored panel",
        ))
        if strict:
            raise BundleMismatch(describe_bundle(findings))
        return findings

    idx = execution.index
    if not isinstance(idx, pd.DatetimeIndex):
        findings.append(Finding(
            "schema", f"execution index is {type(idx).__name__}, not a DatetimeIndex",
        ))
    elif idx.has_duplicates:
        dupes = idx[idx.duplicated()].unique()
        findings.append(Finding(
            "schema", f"execution panel has duplicate sessions: {_preview(dupes)}",
        ))

    non_numeric = [c for c in execution.columns if not is_numeric_dtype(execution[c])]
    if non_numeric:
        findings.append(Finding(
            "schema", f"non-numeric execution columns: {_preview(non_numeric)}",
        ))

    if isinstance(idx, pd.DatetimeIndex):
        want = pd.DatetimeIndex(sorted(pd.to_datetime(scored[date_col]).unique()))
        missing = want.difference(idx.normalize())
        if len(missing) > max_missing_sessions:
            findings.append(Finding(
                "date_coverage",
                f"{len(missing)} scored session(s) have no execution row: "
                f"{_preview(d.date() for d in missing)}",
            ))

    absent = sorted(set(scored[ticker_col].astype(str)) - set(map(str, execution.columns)))
    if absent:
        findings.append(Finding(
            "ticker_coverage",
            f"{len(absent)} scored ticker(s) absent from the execution panel: "
            f"{_preview(absent)}",
        ))

    if strict and findings:
        raise BundleMismatch(describe_bundle(findings))
    return findings


def describe_bundle(findings: list[Finding]) -> str:
    """Operator-facing summary; empty string when the bundle is coherent."""
    if not findings:
        return ""
    return "\n".join(
        ["Execution price panel does not match the scored panel:"]
        + [f"  {f.kind}: {f.detail}" for f in findings]
    )
