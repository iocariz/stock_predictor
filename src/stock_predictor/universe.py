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

import numpy as np
import pandas as pd

DEFAULT_MIN_COVERAGE = 0.9


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


def check_download_coverage(
    requested: list[str],
    adj_close: pd.DataFrame,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    label: str = "price download",
) -> float:
    """Validate that a wide price frame actually covers the requested tickers.

    A column that is present but entirely NaN counts as missing — Yahoo
    returns those for symbols it could not serve.

    Returns the coverage fraction.  Raises :class:`DownloadCoverageError`
    when it falls below *min_coverage*; otherwise prints a warning listing
    what is missing so a partial universe is never silent.
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

    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        detail = (
            f"{label}: {len(got)}/{len(want)} tickers returned "
            f"({coverage:.1%} coverage); missing {len(missing)}: {preview}{more}"
        )
        if coverage < min_coverage:
            raise DownloadCoverageError(
                f"{detail}\n"
                f"Coverage is below the {min_coverage:.0%} threshold. A partial "
                "download silently changes the traded universe and every "
                "cross-sectional feature computed from it. Retry (Yahoo "
                "rate-limits large batches), narrow --sample-n, switch "
                "--provider, or lower --min-coverage to accept this run."
            )
        print(f"  Warning: {detail}")

    return coverage
