"""Does the signal work where the strategy actually trades?

A backtest answers "did this configuration make money", which confounds
ranking skill with market beta, costs, and position sizing. These helpers
answer the narrower question directly: sort each date's universe by score
and ask what the top *k* names went on to return, before any portfolio
construction.

The diagnostic that matters is the *shape*. A ranker with skill puts its
best forward returns at the top of the list. A ranker whose top decile
underperforms its own middle band has no exploitable edge no matter how a
backtest of it happens to land.

Forward returns overlap (a `horizon`-session label sampled daily), so every
t-statistic here is HAC-corrected — see :mod:`stock_predictor.stats`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_predictor.stats import hac_mean_tstat

DEFAULT_BUCKETS: tuple[int, ...] = (5, 10, 15, 25, 50, 100)


@dataclass(frozen=True)
class DepthRow:
    """Forward-return statistics for one selection depth."""

    label: str
    n_names: int
    mean_fwd_ret: float
    excess_vs_universe: float
    excess_t: float
    n_days: int


def _ranked(panel: pd.DataFrame, score_col: str, date_col: str) -> pd.DataFrame:
    out = panel.sort_values([date_col, score_col], ascending=[True, False]).copy()
    out["_rank"] = out.groupby(date_col).cumcount() + 1
    return out


def daily_bucket_returns(
    panel: pd.DataFrame,
    top_k: int,
    *,
    score_col: str = "prob",
    fwd_col: str = "fwd_ret",
    date_col: str = "date",
    from_bottom: bool = False,
) -> pd.Series:
    """Equal-weighted mean forward return of the top (or bottom) *top_k* per date."""
    ranked = _ranked(panel, score_col, date_col)
    if from_bottom:
        depth = ranked.groupby(date_col)["_rank"].transform("max")
        sel = ranked[ranked["_rank"] > depth - top_k]
    else:
        sel = ranked[ranked["_rank"] <= top_k]
    return sel.groupby(date_col)[fwd_col].mean()


def depth_table(
    panel: pd.DataFrame,
    *,
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    score_col: str = "prob",
    fwd_col: str = "fwd_ret",
    date_col: str = "date",
    horizon: int = 10,
    include_bottom: bool = True,
) -> list[DepthRow]:
    """Mean forward return by selection depth, against the universe average.

    *horizon* is the label's forward window; it sets the HAC lag floor because
    consecutive daily observations share ``horizon - 1`` sessions of return.
    """
    for col in (score_col, fwd_col, date_col):
        if col not in panel.columns:
            raise ValueError(f"panel is missing required column {col!r}")

    universe = panel.groupby(date_col)[fwd_col].mean()
    rows: list[DepthRow] = []

    for k in buckets:
        series = daily_bucket_returns(
            panel, k, score_col=score_col, fwd_col=fwd_col, date_col=date_col,
        )
        if series.empty:
            continue
        spread = (series - universe).dropna()
        _, t_stat, _ = hac_mean_tstat(spread.to_numpy(), overlap=horizon)
        rows.append(DepthRow(
            label=f"top {k}",
            n_names=k,
            mean_fwd_ret=float(series.mean()),
            excess_vs_universe=float(spread.mean()),
            excess_t=float(t_stat),
            n_days=len(series),
        ))

    if include_bottom and buckets:
        k = max(buckets)
        series = daily_bucket_returns(
            panel, k, score_col=score_col, fwd_col=fwd_col,
            date_col=date_col, from_bottom=True,
        )
        if not series.empty:
            spread = (series - universe).dropna()
            _, t_stat, _ = hac_mean_tstat(spread.to_numpy(), overlap=horizon)
            rows.append(DepthRow(
                label=f"bottom {k}",
                n_names=k,
                mean_fwd_ret=float(series.mean()),
                excess_vs_universe=float(spread.mean()),
                excess_t=float(t_stat),
                n_days=len(series),
            ))

    rows.append(DepthRow(
        label="universe",
        n_names=int(panel.groupby(date_col).size().mean()),
        mean_fwd_ret=float(universe.mean()),
        excess_vs_universe=0.0,
        excess_t=float("nan"),
        n_days=len(universe),
    ))
    return rows


def rank_ic(
    panel: pd.DataFrame,
    *,
    score_col: str = "prob",
    fwd_col: str = "fwd_ret",
    date_col: str = "date",
    horizon: int = 10,
) -> dict[str, float]:
    """Per-date Spearman rank correlation between score and forward return."""
    ic = panel.groupby(date_col).apply(
        lambda g: g[score_col].corr(g[fwd_col], method="spearman"),
        include_groups=False,
    ).dropna()
    if ic.empty:
        return {"mean": float("nan"), "std": float("nan"), "t": float("nan"), "n_days": 0}
    mean, t_stat, lags = hac_mean_tstat(ic.to_numpy(), overlap=horizon)
    return {
        "mean": mean,
        "std": float(ic.std()),
        "t": t_stat,
        "n_days": len(ic),
        "hac_lags": float(lags),
    }


def is_signal_monotone(rows: list[DepthRow], *, tolerance: float = 0.0) -> bool:
    """True when deeper buckets do not beat the tightest bucket.

    A ranker with usable skill concentrates its best forward returns at the
    top. If ``top 50`` outperforms ``top 5``, the extreme of the ranking is
    not where the edge lives — whatever a backtest of it shows.
    """
    ranked = [r for r in rows if r.label.startswith("top ")]
    if len(ranked) < 2:
        return True
    tightest = min(ranked, key=lambda r: r.n_names)
    return all(
        tightest.mean_fwd_ret >= r.mean_fwd_ret - tolerance
        for r in ranked
        if r.n_names > tightest.n_names
    )


def format_depth_table(rows: list[DepthRow]) -> str:
    """Render :func:`depth_table` output as a fixed-width table."""
    head = (
        f"{'bucket':>12s}  {'mean fwd ret':>13s}  {'vs universe':>12s}"
        f"  {'HAC t':>7s}  {'days':>5s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        t = "     —" if r.excess_t != r.excess_t else f"{r.excess_t:+7.2f}"
        vs = "        —" if r.label == "universe" else f"{r.excess_vs_universe:+12.4%}"
        lines.append(
            f"{r.label:>12s}  {r.mean_fwd_ret:+13.4%}  {vs:>12s}  {t:>7s}  {r.n_days:>5d}"
        )
    return "\n".join(lines)


def depth_frame(rows: list[DepthRow]) -> pd.DataFrame:
    """:func:`depth_table` output as a DataFrame (for CSV export)."""
    return pd.DataFrame([
        {
            "bucket": r.label,
            "n_names": r.n_names,
            "mean_fwd_ret": r.mean_fwd_ret,
            "excess_vs_universe": r.excess_vs_universe,
            "excess_hac_t": r.excess_t,
            "n_days": r.n_days,
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# The same quantity, shaped as a tuning objective
# ---------------------------------------------------------------------------


def top_n_excess_score(
    dates: np.ndarray,
    scores: np.ndarray,
    fwd_ret: np.ndarray,
    top_n: int,
    *,
    risk_adjusted: bool = False,
) -> float:
    """Mean per-date excess forward return of the top-*top_n* scored names.

    This is the trading rule expressed as a number: rank each date, take the
    best *top_n*, equal-weight them, and measure the result against that
    date's universe average (which removes market direction, so folds from
    different regimes are comparable).

    Written for a tuning loop rather than a report, hence numpy arrays and no
    DataFrame. It is deliberately **indifferent to ordering below the
    cutoff** — the strategy never trades those names. NDCG over the full list
    is not: it rewards arranging the whole tail correctly, which is why tuning
    NDCG@15 flattened the traded end of the ranking.

    *risk_adjusted* divides by the standard deviation of the daily excess
    series, turning the objective into an information ratio and penalising an
    edge that comes from a handful of lucky dates.

    Returns 0.0 when the input cannot express a preference (a single date, or
    a basket that spans the whole universe).
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    d = np.asarray(dates)
    sc = np.asarray(scores, dtype=float)
    fr = np.asarray(fwd_ret, dtype=float)

    order = np.argsort(d, kind="stable")
    d, sc, fr = d[order], sc[order], fr[order]
    bounds = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    bounds = np.r_[bounds, len(d)]

    spreads: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        day_fr = fr[lo:hi]
        day_sc = sc[lo:hi]
        ok = np.isfinite(day_fr) & np.isfinite(day_sc)
        if ok.sum() < 2:
            continue
        day_fr, day_sc = day_fr[ok], day_sc[ok]
        k = min(top_n, len(day_fr))
        if k >= len(day_fr):
            # The basket is the universe; there is no preference to express.
            spreads.append(0.0)
            continue
        picked = np.argpartition(-day_sc, k - 1)[:k]
        spreads.append(float(day_fr[picked].mean() - day_fr.mean()))

    if not spreads:
        return 0.0
    arr = np.asarray(spreads, dtype=float)
    mean = float(arr.mean())
    if not risk_adjusted:
        return mean if np.isfinite(mean) else 0.0
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if sd <= 1e-12:
        return 0.0
    out = mean / sd
    return out if np.isfinite(out) else 0.0
