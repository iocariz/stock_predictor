"""Refuse to trade on inputs that have gone stale.

A live run consumes three things that rot at different rates: a fitted model, a
price panel, and the state file. Nothing checked any of them.

That is not hypothetical here. The model deployed in this repo until 2026-08-21
had ``train_end`` of **2022-12-31 — 3.64 years before it was used to pick
trades** — at a 10-day horizon while the strategy traded 63. It was caught by
reading the metadata by hand, not by anything in the code path.

Market data rots the same way and is harder to see: a vendor outage that serves
yesterday's snapshot looks exactly like a quiet market until you compare dates.
Session counts come from the exchange calendar rather than calendar days, so a
long weekend does not read as an outage.

This module only reports. The caller decides whether a finding warns or blocks,
because that is a policy question about a particular account.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_predictor.execution_calendar import exchange_sessions

DEFAULT_MAX_MODEL_AGE_YEARS = 2.0
"""Beyond this, a model has not seen the regime it is trading."""

DEFAULT_MAX_DATA_AGE_SESSIONS = 3
"""Enough slack for a long weekend plus a late vendor, not for an outage."""


@dataclass(frozen=True)
class FreshnessPolicy:
    """Limits for a live run. A limit of 0 disables that check."""

    max_model_age_years: float = DEFAULT_MAX_MODEL_AGE_YEARS
    max_data_age_sessions: int = DEFAULT_MAX_DATA_AGE_SESSIONS

    def __post_init__(self) -> None:
        if self.max_model_age_years < 0:
            raise ValueError("max_model_age_years must be >= 0")
        if self.max_data_age_sessions < 0:
            raise ValueError("max_data_age_sessions must be >= 0")


@dataclass(frozen=True)
class Finding:
    """One stale input, with the number that made it stale."""

    kind: str
    detail: str
    value: float
    limit: float


def _model_age_years(meta: dict, as_of: pd.Timestamp) -> float | None:
    """Years between the model's last training label and *as_of*.

    ``None`` means the age could not be established, which is a finding in its
    own right — an unknown age is not the same as a fresh one.
    """
    # fitted_through is what the model actually learned through; train_end is
    # only what was requested, and purging pushes the two apart by the label
    # horizon. Reading train_end understated staleness by a whole quarter.
    raw = meta.get("fitted_through") or meta.get("train_end")
    if raw is None:
        return None
    stamp = pd.to_datetime(raw, errors="coerce")
    if pd.isna(stamp):
        return None
    return float((as_of - stamp).days) / 365.25


def check_freshness(
    model_meta: dict,
    sessions: pd.DatetimeIndex,
    *,
    as_of: pd.Timestamp | str | None = None,
    policy: FreshnessPolicy | None = None,
) -> list[Finding]:
    """Everything stale about this run's inputs, worst first.

    *sessions* is the price panel's own session index. An empty one means the
    data never arrived, which is reported rather than treated as up to date.
    """
    policy = policy or FreshnessPolicy()
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    now = now.normalize()
    out: list[Finding] = []

    if policy.max_model_age_years > 0:
        age = _model_age_years(model_meta, now)
        if age is None:
            out.append(Finding(
                "model_age",
                "model metadata has no usable train_end; age unknown",
                float("inf"), policy.max_model_age_years,
            ))
        elif age > policy.max_model_age_years:
            out.append(Finding(
                "model_age",
                f"model fitted through "
                f"{model_meta.get('fitted_through') or model_meta.get('train_end')} — "
                f"{age:.2f} years before this run",
                age, policy.max_model_age_years,
            ))

    if policy.max_data_age_sessions > 0:
        idx = pd.DatetimeIndex(sessions)
        if len(idx) == 0:
            out.append(Finding(
                "data_age", "price panel is empty",
                float("inf"), float(policy.max_data_age_sessions),
            ))
        else:
            last = idx.max().normalize()
            # Exchange sessions, not calendar days: counting days would flag
            # every Monday after a holiday as an outage.
            behind = max(0, len(exchange_sessions(last, now)) - 1)
            if behind > policy.max_data_age_sessions:
                out.append(Finding(
                    "data_age",
                    f"last priced session {last.date()} — {behind} sessions behind",
                    float(behind), float(policy.max_data_age_sessions),
                ))

    return sorted(out, key=lambda f: f.value / f.limit if f.limit else 0, reverse=True)


def describe(findings: list[Finding]) -> str:
    """Operator-facing summary; empty string when everything is fresh."""
    if not findings:
        return ""
    lines = ["Stale inputs:"]
    lines += [
        f"  {f.kind}: {f.detail} (limit {f.limit:g})" for f in findings
    ]
    return "\n".join(lines)
