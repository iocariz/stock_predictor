"""Daily inference pipeline: load model, score universe, generate orders.

Example:
  uv run predict-sp500 --model models/latest.pkl --state portfolio.json
  uv run predict-sp500 --model models/latest.pkl --init --initial-capital 50000
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stock_predictor.data_provider import DataProvider, get_provider
from stock_predictor.pit import (
    SP500_STINTS_URL,
    filter_panel_to_pit,
    load_sp500_stints,
    tickers_overlapping_window,
)
from stock_predictor.execution_calendar import trading_dates_from_index
from stock_predictor.portfolio import (
    Order,
    PortfolioState,
    check_kill_switch,
    generate_orders,
    generate_orders_rank_hold,
    init_state,
    load_state,
    portfolio_value,
    save_state,
)
from stock_predictor.training import (
    MACRO_FEATURE_COLS,
    build_feature_panel,
    model_scores,
    wide_field,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: Path) -> tuple[object, dict]:
    """Load model pickle. Returns (model, meta)."""
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    model = payload["model"]
    meta = payload["meta"]
    for key in ("feature_cols", "horizon"):
        if key not in meta:
            raise ValueError(f"Model metadata missing required key: {key}")
    return model, meta


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def download_recent_prices(
    tickers: list[str],
    lookback_days: int = 400,
    provider: DataProvider | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adj_close and volume wide DataFrames for the universe."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    if provider is not None:
        return provider.download_equity_ohlcv(tickers, start.isoformat(), end.isoformat())
    # Legacy yfinance fallback
    from stock_predictor.providers.yfinance_provider import YFinanceProvider

    return YFinanceProvider().download_equity_ohlcv(tickers, start.isoformat(), end.isoformat())


# ---------------------------------------------------------------------------
# Inference panel
# ---------------------------------------------------------------------------


def build_inference_panel(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    stints: pd.DataFrame,
    *,
    start: str,
    end: str | None,
    skip_earnings: bool = True,
    earnings_workers: int = 4,
    provider: DataProvider | None = None,
    provider_name: str = "yfinance",
    macro_merge: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature panel for scoring (no forward return, no label)."""
    long = adj_close.stack(future_stack=True).rename("adj_close").reset_index()
    long.columns = ["date", "ticker", "adj_close"]
    long = long.sort_values(["ticker", "date"])
    long = filter_panel_to_pit(long, stints)
    return build_feature_panel(
        long, volume,
        start=start, end=end,
        skip_earnings=skip_earnings,
        earnings_workers=earnings_workers,
        provider=provider,
        provider_name=provider_name,
        macro_merge=macro_merge,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _forward_fill_date_level_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Forward-fill date-level features (macro) from the most recent available date.

    When Yahoo Finance flakes on ^VIX or other macro downloads, the scoring
    date has NaN for all tickers in those columns.  We fill from the previous
    trading day's values so scoring can proceed.

    Handles both raw (vix, tnx_yield) and derived (vix_ret_5d,
    yield_curve_spread, vix_percentile) macro features.
    """
    # Only macro columns are genuinely date-level (one value shared by every
    # ticker).  Filling a ticker-level column here would smear one ticker's
    # last value across the whole universe, so restrict to MACRO_FEATURE_COLS.
    day = panel[panel["date"] == score_date]
    cols_all_nan = [
        c for c in feature_cols
        if c in MACRO_FEATURE_COLS and c in panel.columns and day[c].isna().all()
    ]
    if not cols_all_nan:
        return panel

    prior = panel[panel["date"] < score_date].sort_values("date")
    if prior.empty:
        return panel

    fill_values: dict[str, float] = {}
    for col in cols_all_nan:
        last_valid = prior[col].dropna()
        if not last_valid.empty:
            fill_values[col] = last_valid.iloc[-1]

    if fill_values:
        mask = panel["date"] == score_date
        for col, val in fill_values.items():
            panel.loc[mask, col] = val
        print(f"  Forward-filled {len(fill_values)} features from prior date: "
              f"{list(fill_values.keys())}")
    return panel


def score_universe(
    model: object,
    panel: pd.DataFrame,
    feature_cols: list[str],
    score_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score the universe on a single date. Returns (ticker, prob, adj_close)."""
    if score_date is None:
        score_date = panel["date"].max()
    day = panel[panel["date"] == score_date].copy()
    if day.empty:
        raise ValueError(f"No data for score date {score_date}")

    missing = set(feature_cols) - set(day.columns)
    if missing:
        raise ValueError(f"Panel missing model features: {missing}")

    # Forward-fill date-level features when macro download partially failed
    would_survive = day.dropna(subset=feature_cols)
    if would_survive.empty:
        panel = _forward_fill_date_level_features(panel, feature_cols, score_date)
        day = panel[panel["date"] == score_date].copy()

    # Drop rows where the ticker itself has no data (all ticker-level features
    # are NaN).  LightGBM handles partial NaN natively (e.g. missing VIX) so
    # we only require that the ticker has price-derived features.
    ticker_cols = [c for c in feature_cols if c not in MACRO_FEATURE_COLS and c != "sector"]
    day = day.dropna(subset=ticker_cols, how="all")
    if day.empty:
        raise ValueError(f"All rows have NaN features on {score_date}")

    nan_pct = day[feature_cols].isna().mean().mean()
    if nan_pct > 0:
        print(f"  Note: {nan_pct:.1%} feature values are NaN (LightGBM handles these natively)")

    # Classifier → probability of +5%; ranker → raw lambdarank score.
    # Both are pick-the-largest signals.
    probs = model_scores(model, day[feature_cols])
    scored = (
        day[["ticker", "adj_close"]]
        .assign(prob=probs)
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )
    return scored


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_signal_report(
    scored: pd.DataFrame,
    orders: tuple[Order, ...],
    state: PortfolioState,
    nav: float,
    drawdown: float,
    halted: bool,
    *,
    top_n: int,
    max_drawdown: float,
) -> None:
    today = date.today().isoformat()
    status = "HALTED" if halted else "ACTIVE"

    print()
    print("=" * 60)
    print(f"DAILY SIGNAL \u2014 {today}")
    print("=" * 60)
    print(
        f"Portfolio: ${nav:,.0f} | "
        f"Drawdown: {drawdown:+.1%} (kill-switch at {-max_drawdown:.0%}) | "
        f"Status: {status}"
    )
    print()

    sells = [o for o in orders if o.action == "SELL"]
    buys = [o for o in orders if o.action == "BUY"]

    if sells:
        print("CLOSING POSITIONS (sell at open):")
        for o in sells:
            # Find matching position for P&L
            pos = next((p for p in state.positions if p.ticker == o.ticker and p.cohort_id == o.cohort_id), None)
            pnl = (o.price - pos.entry_price) * o.shares if pos else 0
            print(f"  SELL {o.shares:>4d} {o.ticker:<6s} @ ~${o.price:.2f}  (P&L {pnl:+,.0f})")
        print()

    if halted:
        print("*** KILL-SWITCH ENGAGED — no new positions ***")
        print()
    elif buys:
        print("NEW PICKS (buy at open):")
        print(f"  {'Rank':>4s}  {'Ticker':<6s}  {'Score':>7s}  {'Shares':>6s}  {'~Cost':>8s}")
        for i, o in enumerate(buys, 1):
            cost = o.shares * o.price
            # Find prob from scored
            row = scored[scored["ticker"] == o.ticker]
            prob = row["prob"].iloc[0] if len(row) else 0
            print(f"  {i:>4d}  {o.ticker:<6s}  {prob:>7.3f}  {o.shares:>6d}  ${cost:>7,.0f}")
        print()
    elif not sells:
        print("No orders today (no expirations, no available cohort slots).")
        print()

    # Active holds
    non_expiring = [p for p in state.positions if p.expiry_date > today]
    if non_expiring:
        print("HOLD (active, not expiring):")
        for p in non_expiring:
            print(f"  {p.shares:>4d} {p.ticker:<6s}  (expires {p.expiry_date})")
        print()

    total_cost = sum(o.shares * o.price for o in buys) + sum(o.shares * o.price for o in sells)
    print(f"Summary: {len(sells)} sells, {len(buys)} buys, {len(non_expiring)} holds")
    print(f"Estimated turnover: ${total_cost:,.0f}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily inference: score S&P 500 universe, generate orders.",
    )
    p.add_argument("--model", type=Path, required=True, help="Path to model .pkl")
    p.add_argument("--state", type=Path, default=Path("portfolio_state.json"),
                   help="Portfolio state JSON (default: portfolio_state.json)")
    p.add_argument("--init", action="store_true", help="Create new portfolio state")
    p.add_argument("--initial-capital", type=float, default=100_000.0, dest="initial_capital")
    p.add_argument("--top-n", type=int, default=15, dest="top_n")
    p.add_argument("--max-cohorts", type=int, default=2, dest="max_cohorts")
    p.add_argument("--holding-days", type=int, default=10, dest="holding_days")
    p.add_argument(
        "--hold-mode",
        default="fixed",
        choices=["fixed", "rank"],
        dest="hold_mode",
        help="fixed: sell after --holding-days sessions (cohort parity); "
        "rank: sell only when a holding's rank decays beyond --exit-rank "
        "(parity with backtest-sp500 --mode rank-hold). Do not switch modes "
        "on an existing state file.",
    )
    p.add_argument(
        "--exit-rank",
        type=int,
        default=40,
        dest="exit_rank",
        help="rank mode: sell held names ranked worse than this (>= top-n)",
    )
    p.add_argument("--max-drawdown", type=float, default=0.15, dest="max_drawdown")
    p.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    p.add_argument("--skip-earnings", action="store_true", dest="skip_earnings")
    p.add_argument("--confirm", action="store_true",
                   help="Update portfolio state with new orders (without this flag: dry run)")
    p.add_argument("--sample-n", type=int, default=500, dest="sample_n",
                   help="Max tickers to download")
    p.add_argument(
        "--provider",
        default="yfinance",
        choices=["yfinance", "tiingo"],
        help="Data provider: yfinance (default) or tiingo (Tiingo equities + FRED macro)",
    )
    p.add_argument("--weighting", default="equal", choices=["equal", "probability"],
                   help="Cohort dollar weights (matches backtest --weighting)")
    p.add_argument(
        "--commission-per-share",
        type=float,
        default=0.0,
        help="Per-share commission per buy/sell leg (same as backtest-sp500)",
    )
    p.add_argument(
        "--commission-per-order",
        type=float,
        default=0.0,
        help="Flat commission per ticker per buy/sell order (same as backtest-sp500)",
    )
    p.add_argument(
        "--no-macro-merge",
        action="store_true",
        help="Disable Yahoo↔FRED macro cross-fill during feature build",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider)

    # Load or init state
    if args.init:
        state = init_state(args.initial_capital)
        save_state(state, args.state)
        print(f"Initialized portfolio: {args.state} (${args.initial_capital:,.0f})")
        if not args.model:
            return

    if not args.state.exists():
        sys.exit(f"Portfolio state not found: {args.state}. Use --init to create one.")

    state = load_state(args.state)
    print(f"Loaded portfolio from {args.state} (cash: ${state.cash:,.0f}, "
          f"{len(state.positions)} positions)")

    # Load model
    print(f"Loading model from {args.model}...")
    model, meta = load_model(args.model)
    feature_cols = meta["feature_cols"]
    print(f"  Features: {len(feature_cols)} columns, horizon={meta.get('horizon', '?')}")

    # Download universe
    print("Loading PIT stints & downloading prices...")
    stints = load_sp500_stints(SP500_STINTS_URL)
    lookback = 400
    start_date = (date.today() - timedelta(days=lookback)).isoformat()
    tickers = tickers_overlapping_window(stints, start_date, None)
    sample = tickers[:args.sample_n]
    print(f"  Universe: {len(sample)} tickers")

    adj_close, volume = download_recent_prices(sample, lookback_days=lookback, provider=provider)
    print(f"  Downloaded: {adj_close.shape[0]} days x {adj_close.shape[1]} tickers")

    # Build features
    print("Computing features...")
    panel, inferred_cols = build_inference_panel(
        adj_close, volume, stints,
        start=start_date, end=None,
        skip_earnings=args.skip_earnings,
        provider=provider,
        provider_name=args.provider,
        macro_merge=not args.no_macro_merge,
    )

    # Validate feature alignment
    missing_feats = set(feature_cols) - set(panel.columns)
    if missing_feats:
        print(f"Warning: model expects features not in panel: {missing_feats}", file=sys.stderr)
        print("Run with matching --skip-earnings flag as training.", file=sys.stderr)

    # Score
    print("Scoring universe...")
    scored = score_universe(model, panel, feature_cols)
    print(f"  Scored {len(scored)} tickers. Top-5:")
    for _, row in scored.head(5).iterrows():
        print(f"    {row['ticker']:<6s}  P(+5%)={row['prob']:.3f}  @ ${row['adj_close']:.2f}")

    # Current prices for valuation
    latest_prices = dict(zip(scored["ticker"], scored["adj_close"]))

    # Kill-switch check
    halted, nav, dd = check_kill_switch(state, latest_prices, args.max_drawdown)

    # Generate orders (calendar = session dates in downloaded OHLC index)
    trading_dates = trading_dates_from_index(adj_close.index)
    if args.hold_mode == "rank":
        # Rank exits need the FULL ranking, not just the top of the list.
        orders, new_state = generate_orders_rank_hold(
            state, scored.to_dict("records"), latest_prices,
            top_n=args.top_n,
            exit_rank=args.exit_rank,
            slippage_bps=args.slippage_bps,
            as_of=date.today().isoformat(),
            trading_dates=trading_dates,
            commission_per_share=args.commission_per_share,
            commission_per_order=args.commission_per_order,
            allow_buys=not halted,
        )
    else:
        picks = scored.head(args.top_n * 2).to_dict("records")  # extra candidates in case some filtered
        orders, new_state = generate_orders(
            state, picks, latest_prices,
            top_n=args.top_n,
            max_cohorts=args.max_cohorts,
            holding_days=args.holding_days,
            slippage_bps=args.slippage_bps,
            as_of=date.today().isoformat(),
            trading_dates=trading_dates,
            weighting=args.weighting,
            commission_per_share=args.commission_per_share,
            commission_per_order=args.commission_per_order,
            allow_buys=not halted,
        )

    print_signal_report(
        scored, orders, state, nav, dd, halted,
        top_n=args.top_n, max_drawdown=args.max_drawdown,
    )

    if args.confirm:
        save_state(new_state, args.state)
        print(f"Portfolio updated: {args.state}")
    else:
        print("Dry run (use --confirm to update portfolio state).")


if __name__ == "__main__":
    main()
