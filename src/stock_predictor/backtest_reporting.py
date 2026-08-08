"""Reporting and plotting for backtest results."""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from stock_predictor.backtest import BacktestResult

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(v: float) -> str:
    return f"{v:+.1%}" if not np.isnan(v) else "N/A"


def _fmt_f(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if not np.isnan(v) else "N/A"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


# ---------------------------------------------------------------------------
# Relative-return (active vs benchmark) metrics
# ---------------------------------------------------------------------------


def relative_metrics(strategy_nav: pd.Series, bench_nav: pd.Series) -> dict[str, float]:
    """Active-return statistics of a strategy vs a benchmark.

    Frames the strategy as an overlay on the benchmark: daily active return
    ``r_strat - r_bench`` (a dollar-neutral long-strategy / short-benchmark
    book, rebalanced daily). Returns annualized active return, tracking
    error, information ratio, CAPM beta/alpha, up/down capture, and the
    overlay equity's total return and max drawdown.
    """
    if strategy_nav is None or bench_nav is None:
        return {}
    idx = strategy_nav.index.intersection(bench_nav.index).sort_values()
    if len(idx) < 3:
        return {}
    rs = strategy_nav.reindex(idx).astype(float).pct_change()
    rb = bench_nav.reindex(idx).astype(float).pct_change()
    both = pd.concat([rs.rename("s"), rb.rename("b")], axis=1).dropna()
    if len(both) < 2:
        return {}
    rs, rb = both["s"], both["b"]
    active = rs - rb

    te = float(active.std() * np.sqrt(252))
    active_ann = float(active.mean() * 252)
    ir = float(active.mean() / active.std() * np.sqrt(252)) if active.std() > 0 else float("nan")

    var_b = float(rb.var())
    beta = float(rs.cov(rb) / var_b) if var_b > 0 else float("nan")
    alpha_ann = (
        float((rs.mean() - beta * rb.mean()) * 252) if beta == beta else float("nan")
    )
    # t-stat of daily CAPM alpha: mean/std of regression residuals * sqrt(n).
    # |t| >= ~2 is the usual bar for "not noise".
    alpha_t = float("nan")
    if beta == beta:
        resid = rs - beta * rb
        rstd = float(resid.std())
        # Epsilon guard: identical series leave ~1e-17 float noise in the
        # residuals, whose mean/std ratio is a spurious t-stat.
        if rstd > 1e-12:
            alpha_t = float(resid.mean() / rstd * np.sqrt(len(resid)))

    up = rb > 0
    down = rb < 0
    up_capture = float(rs[up].mean() / rb[up].mean()) if up.any() and rb[up].mean() != 0 else float("nan")
    down_capture = (
        float(rs[down].mean() / rb[down].mean()) if down.any() and rb[down].mean() != 0 else float("nan")
    )

    overlay = (1.0 + active).cumprod()
    overlay_total = float(overlay.iloc[-1] - 1.0)
    overlay_dd = float((overlay / overlay.cummax() - 1.0).min())

    return {
        "active_return_ann": active_ann,
        "tracking_error": te,
        "information_ratio": ir,
        "beta": beta,
        "alpha_ann": alpha_ann,
        "alpha_t": alpha_t,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "overlay_total_return": overlay_total,
        "overlay_max_drawdown": overlay_dd,
    }


def print_relative_report(strategy_nav: pd.Series, bench_nav: pd.Series, bench_label: str) -> None:
    """Print the active-vs-benchmark section (no-op if series don't overlap)."""
    rm = relative_metrics(strategy_nav, bench_nav)
    if not rm:
        return
    print(f"ACTIVE vs {bench_label} (long strategy / short benchmark, daily rebalance)")
    print("-" * 48)
    print(f"{'Active return (ann)':28s} {_fmt_pct(rm['active_return_ann']):>10s}")
    print(f"{'Tracking error (ann)':28s} {_fmt_pct(rm['tracking_error']):>10s}")
    print(f"{'Information ratio':28s} {_fmt_f(rm['information_ratio']):>10s}")
    print(f"{'Beta vs benchmark':28s} {_fmt_f(rm['beta']):>10s}")
    print(f"{'CAPM alpha (ann)':28s} {_fmt_pct(rm['alpha_ann']):>10s}")
    print(f"{'Alpha t-stat':28s} {_fmt_f(rm['alpha_t']):>10s}")
    print(f"{'Up capture':28s} {_fmt_f(rm['up_capture']):>10s}")
    print(f"{'Down capture':28s} {_fmt_f(rm['down_capture']):>10s}")
    print(f"{'Overlay total return':28s} {_fmt_pct(rm['overlay_total_return']):>10s}")
    print(f"{'Overlay max drawdown':28s} {_fmt_pct(rm['overlay_max_drawdown']):>10s}")
    print()


# ---------------------------------------------------------------------------
# Console reports
# ---------------------------------------------------------------------------


def print_report(result: BacktestResult) -> None:
    c = result.config
    m = result.metrics
    s = result.spy_metrics
    hdr = f"BACKTEST REPORT: {result.start_date.date()} to {result.end_date.date()}"
    bench_hdr = f"{c.benchmark_ticker} (B&H)" if c.benchmark_ticker else "\u2014"
    has_bench = bool(c.benchmark_ticker) and bool(s)
    col_w = max(12, len(bench_hdr), len("STRATEGY"))
    print()
    print("=" * (22 + 2 + col_w * 2 + 2))
    print(hdr)
    print("=" * (22 + 2 + col_w * 2 + 2))
    comm = ""
    if c.commission_per_share > 0 or c.commission_per_order > 0:
        comm = (
            f", comm=${c.commission_per_share:.4f}/sh "
            f"+ ${c.commission_per_order:.2f}/order-leg"
        )
    print(
        f"Config: top_n={c.top_n}, holding={c.holding_days}d, "
        f"rebalance={c.rebalance_day}, weighting={c.weighting}, "
        f"slippage={c.slippage_bps:.0f}bps, max_cohorts={c.max_overlapping_cohorts}"
        f"{comm}"
    )
    print()
    print(f"{'':20s} {'STRATEGY':>{col_w}s}  {bench_hdr:>{col_w}s}")
    print("-" * (22 + 2 + col_w * 2 + 2))
    rows = [
        ("Total Return", _fmt_pct(m.get("total_return", float("nan"))),
         _fmt_pct(s.get("total_return", float("nan"))) if has_bench else "N/A"),
        ("CAGR", _fmt_pct(m.get("cagr", float("nan"))),
         _fmt_pct(s.get("cagr", float("nan"))) if has_bench else "N/A"),
        ("Sharpe", _fmt_f(m.get("sharpe", float("nan"))),
         _fmt_f(s.get("sharpe", float("nan"))) if has_bench else "N/A"),
        ("Sortino", _fmt_f(m.get("sortino", float("nan"))),
         _fmt_f(s.get("sortino", float("nan"))) if has_bench else "N/A"),
        ("Max Drawdown", _fmt_pct(m.get("max_drawdown", float("nan"))),
         _fmt_pct(s.get("max_drawdown", float("nan"))) if has_bench else "N/A"),
        ("Calmar", _fmt_f(m.get("calmar", float("nan"))),
         _fmt_f(s.get("calmar", float("nan"))) if has_bench else "N/A"),
        ("Total Costs", _fmt_dollar(m.get("total_costs", 0)),
         _fmt_dollar(0)),
        ("Win Rate", _fmt_pct(m.get("win_rate", float("nan"))), "N/A"),
        ("N Cohorts", str(int(m.get("n_cohorts", 0))), "N/A"),
    ]
    for label, strat, spy in rows:
        print(f"{label:20s} {strat:>{col_w}s}  {spy:>{col_w}s}")
    print("=" * (22 + 2 + col_w * 2 + 2))
    print()
    if has_bench and len(result.spy_daily_nav) > 1:
        print_relative_report(
            result.daily_nav, result.spy_daily_nav, c.benchmark_ticker or "benchmark",
        )


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------


def _nav_only_metrics(nav: pd.Series) -> dict[str, float]:
    """Risk/return stats from a NAV series alone (no cohort breakdown)."""
    nav = nav.astype(float).dropna()
    if len(nav) < 2:
        return {}
    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)
    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
    years = n_days / 252.0
    cagr = (
        float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else float("nan")
    )
    mean_r = float(daily_ret.mean())
    std_r = float(daily_ret.std())
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else float("nan")
    neg = daily_ret[daily_ret < 0]
    downside = float(neg.std()) if len(neg) > 0 else float("nan")
    sortino = (mean_r / downside * np.sqrt(252)) if downside > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def _nav_normalized_overlap(
    nav_a: pd.Series, nav_b: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    idx = nav_a.index.intersection(nav_b.index).sort_values()
    if len(idx) < 2:
        raise ValueError(
            "Need at least two overlapping trading days between strategies for comparison."
        )
    a = nav_a.reindex(idx).astype(float).ffill().bfill()
    b = nav_b.reindex(idx).astype(float).ffill().bfill()
    a = a / float(a.iloc[0])
    b = b / float(b.iloc[0])
    return a, b, idx


def print_strategy_comparison(
    result_a: BacktestResult,
    result_b: BacktestResult,
    *,
    label_a: str = "Strategy A",
    label_b: str = "Strategy B",
) -> None:
    """Print side-by-side metrics on the overlapping trading-day window."""
    a_n, b_n, idx = _nav_normalized_overlap(result_a.daily_nav, result_b.daily_nav)
    m_a = _nav_only_metrics(a_n)
    m_b = _nav_only_metrics(b_n)
    w = max(len(label_a), len(label_b), 10)
    print()
    print("=" * (24 + w * 2 + 4))
    print(
        f"STRATEGY COMPARISON (overlap {idx[0].date()} \u2192 {idx[-1].date()}, "
        f"{len(idx)} days, NAV normalized to 1.0 at start)"
    )
    print("=" * (24 + w * 2 + 4))
    print(f"{'Metric':22s} {label_a:>{w}s}  {label_b:>{w}s}")
    print("-" * (24 + w * 2 + 4))
    pct_metrics = ("total_return", "cagr", "max_drawdown")
    for key, title in [
        ("total_return", "Total Return"),
        ("cagr", "CAGR"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("max_drawdown", "Max Drawdown"),
        ("calmar", "Calmar"),
    ]:
        va, vb = m_a.get(key, float("nan")), m_b.get(key, float("nan"))
        if key in pct_metrics:
            ra, rb = _fmt_pct(va), _fmt_pct(vb)
        else:
            ra, rb = _fmt_f(va), _fmt_f(vb)
        print(f"{title:22s} {ra:>{w}s}  {rb:>{w}s}")
    print(
        f"{'N cohorts (full sample)':22s} {int(result_a.metrics.get('n_cohorts', 0)):>{w}d}  "
        f"{int(result_b.metrics.get('n_cohorts', 0)):>{w}d}"
    )
    print("=" * (24 + w * 2 + 4))
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_strategy_comparison(
    result_a: BacktestResult,
    result_b: BacktestResult,
    output_path: Path,
    *,
    label_a: str = "Strategy A",
    label_b: str = "Strategy B",
) -> None:
    """Overlay normalized equity (growth of $1) on the shared calendar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    a_n, b_n, idx = _nav_normalized_overlap(result_a.daily_nav, result_b.daily_nav)
    fig, ax = plt.subplots(figsize=(13, 5))
    (a_n * result_a.config.initial_capital).plot(ax=ax, label=label_a, linewidth=1.5)
    (b_n * result_b.config.initial_capital).plot(ax=ax, label=label_b, linewidth=1.5)
    c = result_a.config
    spy = result_a.spy_daily_nav
    if len(spy) > 0 and spy.notna().any() and c.benchmark_ticker:
        spy_s = spy.reindex(idx).astype(float).ffill().bfill()
        if not spy_s.isna().all() and float(spy_s.iloc[0]) > 0:
            spy_n = spy_s / float(spy_s.iloc[0])
            (spy_n * c.initial_capital).plot(
                ax=ax,
                label=f"{c.benchmark_ticker} (B&H)",
                linewidth=1.5,
                alpha=0.65,
                linestyle="--",
            )
    ax.set_title("Strategy comparison (aligned calendar, scaled to initial capital)")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved strategy comparison plot to {output_path}")


def plot_backtest(result: BacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    fig, ax = plt.subplots(figsize=(13, 5))
    result.daily_nav.plot(ax=ax, label="Strategy", linewidth=1.5)
    c = result.config
    if len(result.spy_daily_nav) > 0 and result.spy_daily_nav.notna().any():
        bench_label = f"{c.benchmark_ticker} (B&H)" if c.benchmark_ticker else "Benchmark"
        result.spy_daily_nav.plot(ax=ax, label=bench_label, linewidth=1.5, alpha=0.7)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=120)
    plt.close(fig)

    # Drawdown
    dd = result.daily_nav / result.daily_nav.cummax() - 1
    fig2, ax2 = plt.subplots(figsize=(13, 3))
    dd.plot(ax=ax2, color="tomato", linewidth=1)
    ax2.fill_between(dd.index, dd.values, 0, color="tomato", alpha=0.2)
    ax2.set_title("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(output_dir / "drawdown.png", dpi=120)
    plt.close(fig2)

    # Monthly returns heatmap
    monthly = result.daily_nav.resample("ME").last().pct_change().dropna()
    if len(monthly) > 0:
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
        pivot = pivot.reindex(columns=range(1, 13))
        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        pivot.columns = month_labels
        fig3, ax3 = plt.subplots(figsize=(12, max(3, len(pivot) * 0.8)))
        vals = np.ma.masked_invalid(pivot.values.astype(float))
        cmap = plt.colormaps["RdYlGn"].resampled(256)
        cmap.set_bad(color=(0.92, 0.92, 0.92, 1.0))
        im = ax3.imshow(
            vals, cmap=cmap, aspect="auto",
            vmin=-0.10, vmax=0.10,
        )
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax3.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)
        ax3.set_title("Monthly Returns")
        cbar = fig3.colorbar(im, ax=ax3, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        plt.tight_layout()
        fig3.savefig(output_dir / "monthly_returns.png", dpi=120)
        plt.close(fig3)

    print(f"Saved backtest plots to {output_dir}")
