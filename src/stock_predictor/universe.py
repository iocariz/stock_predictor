"""Universe selection and download-coverage validation.

Two guards that keep the traded universe honest:

* :func:`sample_tickers` draws a *seeded random* subset when a run caps the
  universe.  Slicing a sorted ticker list (``tickers[:n]``) silently returns
  an alphabetical prefix, which biases every cross-sectional feature —
  ranks, "market" regime medians, sector-relative spreads — and the
  investable universe of the backtest along with them.
* :func:`check_download_coverage` fails a run whose price download came back
  materially incomplete, instead of letting a rate-limited partial response
  masquerade as the S&P 500.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

# Applies to *current* index members, which vendors serve reliably; the
# departed-member gap is reported separately, never gated.
DEFAULT_MIN_COVERAGE = 0.98

DEFAULT_RECENT_SESSIONS = 20
DEFAULT_MIN_RECENT_COVERAGE = 0.8
"""Fraction of the most recent sessions a name needs before it can be ranked.

Coverage and rankability are different questions. A ticker that came back from
the vendor counts as covered even if its last three weeks are missing -- and a
momentum feature computed across a three-week hole is not a slightly wrong
number, it is a different one. Measured on this repo's panel, AVB and EQR were
both continuous S&P 500 members present on 1 and 2 of the 20 most recent
sessions, and both counted as fully covered."""


class DownloadCoverageError(RuntimeError):
    """Raised when a price download returned too few of the requested tickers."""


def sample_tickers(tickers: list[str], max_n: int, *, seed: int) -> list[str]:
    """Return at most *max_n* tickers, drawn without replacement.

    The draw is seeded, so a run is reproducible, and unbiased with respect
    to symbol name — unlike ``sorted(tickers)[:max_n]``, which returns
    whatever the front of the alphabet happens to be.

    Output is sorted for stable downstream ordering.
    """
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")
    unique = sorted(set(tickers))
    if len(unique) <= max_n:
        return unique
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.array(unique, dtype=object), size=max_n, replace=False)
    return sorted(picked.tolist())


def _preview(names: list[str], n: int = 10) -> str:
    head = ", ".join(names[:n])
    return head + (f" (+{len(names) - n} more)" if len(names) > n else "")


def check_download_coverage(
    requested: list[str],
    adj_close: pd.DataFrame,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    active: set[str] | None = None,
    label: str = "price download",
) -> float:
    """Validate that a wide price frame actually covers the requested tickers.

    A column that is present but entirely NaN counts as missing — vendors
    return those for symbols they could not serve.

    Two failure modes look identical in a flat coverage number but mean
    opposite things:

    * A **current** index member is missing → the request broke or was
      throttled. Yahoo serves current members reliably, so this is a fault.
    * A **departed** member is missing → the symbol was acquired, renamed, or
      delisted and the vendor no longer carries it. On the real S&P universe
      that is ~97 of 691 tickers, so a flat 90% gate would block every
      full-universe run. This is survivorship bias, not a download fault: it
      flatters backtest results and belongs in a warning, not an exception.

    Pass *active* (see :func:`stock_predictor.pit.current_members`) to apply
    *min_coverage* to current members only and report the departed gap
    separately. Without it, *min_coverage* applies to the whole request.

    Returns overall coverage. Raises :class:`DownloadCoverageError` when the
    gated cohort falls short.
    """
    want = sorted(set(requested))
    if not want:
        raise ValueError("requested ticker list is empty")

    if adj_close is None or adj_close.empty:
        got: set[str] = set()
    else:
        non_empty = adj_close.columns[adj_close.notna().any(axis=0)]
        got = set(map(str, non_empty)) & set(want)

    coverage = len(got) / len(want)
    missing = [t for t in want if t not in got]

    gated, gated_label = want, "requested"
    if active:
        gated = [t for t in want if t in active]
        gated_label = "current index members"

    gated_got = [t for t in gated if t in got]
    gated_missing = [t for t in gated if t not in got]
    gated_coverage = len(gated_got) / len(gated) if gated else coverage

    if gated_coverage < min_coverage:
        raise DownloadCoverageError(
            f"{label}: only {len(gated_got)}/{len(gated)} {gated_label} returned "
            f"({gated_coverage:.1%}), below the {min_coverage:.0%} threshold.\n"
            f"Missing: {_preview(gated_missing)}\n"
            "Vendors serve current members reliably, so this is a broken or "
            "throttled download, not survivorship. A partial download silently "
            "changes the traded universe and every cross-sectional feature "
            "computed from it. Retry, lower --batch-size (Yahoo throttles large "
            "batches), switch --provider, or lower --min-coverage to accept it."
        )

    print(
        f"  {label}: {len(got)}/{len(want)} tickers ({coverage:.1%} overall)"
        + (f", {gated_coverage:.1%} of {gated_label}" if active else "")
    )
    if active:
        departed_missing = [t for t in missing if t not in active]
        if departed_missing:
            n_departed = len([t for t in want if t not in active])
            print(
                f"  Survivorship gap: {len(departed_missing)}/{n_departed} departed "
                f"index members are unavailable from this vendor and are absent "
                f"from the panel, which flatters results. "
                f"Missing: {_preview(departed_missing, 5)}"
            )
    elif missing:
        print(f"  Warning: missing {len(missing)}: {_preview(missing)}")

    return coverage


def recent_coverage(
    adj_close: pd.DataFrame, *, sessions: int = DEFAULT_RECENT_SESSIONS,
) -> pd.Series:
    """Fraction of the last *sessions* rows for which each ticker has a price.

    A zero is treated as missing: it is a placeholder, not a quote, and it
    divides badly in every ratio the feature set builds.
    """
    if len(adj_close) == 0:
        return pd.Series(dtype=float)
    window = adj_close.tail(sessions)
    priced = window.notna() & (window > 0)
    return priced.sum(axis=0) / len(window)


def thin_recent_names(
    adj_close: pd.DataFrame,
    *,
    sessions: int = DEFAULT_RECENT_SESSIONS,
    min_fraction: float = DEFAULT_MIN_RECENT_COVERAGE,
) -> list[str]:
    """Tickers too sparsely priced lately to be ranked, sorted for stable reports.

    The floor is inclusive: a name exactly at *min_fraction* is kept.
    """
    cov = recent_coverage(adj_close, sessions=sessions)
    return sorted(str(t) for t in cov.index[cov < min_fraction])


def universe_hash(tickers: Iterable[str]) -> str:
    """Stable digest of a ticker set, order-independent.

    SHA-256 rather than :func:`hash`, whose string seed is randomised per
    process, so the same universe hashes the same way across runs.
    """
    payload = "|".join(sorted(str(t) for t in tickers))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_live_universe(
    training_universe: Iterable[str], current: Iterable[str],
) -> list[str]:
    """Names the model was fitted on that the index still holds.

    Training samples from the union covering the whole training window;
    inference used to resample from a 400-day window with the same seed, which
    is a different draw from a different population -- at ``sample_n=500``,
    only 307 of 500 names overlapped. Every cross-sectional feature is a rank
    within the universe, so that changes the inputs the model reads.

    Recording what was actually drawn removes the dependence on reproducing a
    sample. Members added since training are excluded rather than silently
    joining the cross-section: the model has never ranked them, and a monthly
    retrain is what brings them in.
    """
    keep = sorted(set(str(t) for t in training_universe) & set(str(c) for c in current))
    if not keep:
        raise ValueError(
            "no overlap between the model's training universe and current "
            "index membership; the model is too old to trade"
        )
    return keep


def universe_drift(
    training_universe: Iterable[str], current: Iterable[str],
) -> dict[str, float]:
    """How far current membership has moved from what the model was fitted on."""
    train = {str(t) for t in training_universe}
    now = {str(c) for c in current}
    tradable = train & now
    return {
        "tradable": float(len(tradable)),
        "current_not_in_training": float(len(now - train)),
        "training_no_longer_current": float(len(train - now)),
        "coverage_of_current": float(len(tradable) / len(now)) if now else 0.0,
    }
