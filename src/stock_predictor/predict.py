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

import pandas as pd

from stock_predictor.data_provider import DataProvider, get_provider
from stock_predictor.execution_calendar import trading_dates_from_index
from stock_predictor.freshness import FreshnessPolicy, check_freshness, describe
from stock_predictor.pit import (
    SP500_STINTS_URL,
    current_members,
    load_sp500_stints,
    tickers_overlapping_window,
)
from stock_predictor.portfolio import (
    Order,
    PortfolioState,
    check_kill_switch,
    generate_orders,
    generate_orders_rank_hold,
    init_state,
    load_state,
    save_state,
    stale_positions,
)
from stock_predictor.quotes import (
    describe_quote_gaps,
    execution_quotes,
    last_quote_dates,
    quote_ages,
    valuation_marks,
)
from stock_predictor.training import (
    MACRO_FEATURE_COLS,
    build_feature_panel,
    model_scores,
    score_label,
)
from stock_predictor.universe import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_RECENT_COVERAGE,
    DEFAULT_RECENT_SESSIONS,
    check_download_coverage,
    resolve_live_universe,
    sample_tickers,
    thin_recent_names,
    universe_drift,
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
    fundamentals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature panel for scoring (no forward return, no label).

    *stints* is handed to :func:`build_feature_panel` rather than applied
    here: rolling features must see each ticker's full contiguous history,
    and the cross-sectional features must see only in-index names — the same
    staging the training pipeline uses.
    """
    long = adj_close.stack(future_stack=True).rename("adj_close").reset_index()
    long.columns = ["date", "ticker", "adj_close"]
    long = long.sort_values(["ticker", "date"])
    return build_feature_panel(
        long, volume,
        start=start, end=end,
        skip_earnings=skip_earnings,
        earnings_workers=earnings_workers,
        provider=provider,
        provider_name=provider_name,
        macro_merge=macro_merge,
        stints=stints,
        fundamentals=fundamentals,
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
    *,
    price_col: str = "adj_close",
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

    # A price is required to rank. Dropping only rows whose ticker-level
    # features are *all* NaN kept unpriced names alive, because date-level
    # calendar features like days_to_fomc are not in MACRO_FEATURE_COLS and so
    # counted as ticker-level. Such a row was then ranked, entered
    # latest_prices as NaN, and inflated the width min_cross_section measures.
    px = pd.to_numeric(day[price_col], errors="coerce")
    day = day[px.notna() & (px > 0)]
    if day.empty:
        raise ValueError(f"No usable price on {score_date}")

    # Beyond the price, LightGBM handles partial NaN natively (e.g. a missing
    # VIX), so a name is kept as long as it has some price-derived feature.
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


DEFAULT_UNIVERSE_SEED = 42


def resolve_universe_seed(explicit: int | None, meta: dict) -> int:
    """Which seed draws the live universe.

    The live sample must be the same draw the model was fitted on: the
    universe defines every cross-sectional feature, so an unrelated sample
    silently changes the inputs the model was trained to read. An explicit
    ``--seed`` wins; otherwise the model's own recorded seed; otherwise the
    default. A stored seed that is not a number is ignored rather than fatal.
    """
    if explicit is not None:
        return int(explicit)
    try:
        return int(meta.get("seed", DEFAULT_UNIVERSE_SEED))
    except (TypeError, ValueError):
        return DEFAULT_UNIVERSE_SEED


def sample_mismatch_warning(sample_n: int, meta: dict) -> str | None:
    """Warn when the live universe size differs from training's, else ``None``."""
    trained = meta.get("sample_n")
    if trained is None or trained == sample_n:
        return None
    return (
        f"--sample-n {sample_n} differs from training ({trained}); the live "
        "universe will not match the panel the model was trained on."
    )


def missing_feature_columns(panel: pd.DataFrame, feature_cols: list[str]) -> set[str]:
    """Features the model expects that the panel does not carry."""
    return set(feature_cols) - set(panel.columns)


def format_signal_report(
    scored: pd.DataFrame,
    orders: tuple[Order, ...],
    state: PortfolioState,
    *,
    nav: float,
    drawdown: float,
    halted: bool,
    top_n: int,
    max_drawdown: float,
) -> list[str]:
    """The daily signal as lines, so it can be asserted on rather than eyeballed."""
    today = date.today().isoformat()
    status = "HALTED" if halted else "ACTIVE"
    out: list[str] = [
        "",
        "=" * 60,
        f"DAILY SIGNAL \u2014 {today}",
        "=" * 60,
        f"Portfolio: ${nav:,.0f} | "
        f"Drawdown: {drawdown:+.1%} (kill-switch at {-max_drawdown:.0%}) | "
        f"Status: {status}",
        "",
    ]

    sells = [o for o in orders if o.action == "SELL"]
    buys = [o for o in orders if o.action == "BUY"]

    if sells:
        out.append("CLOSING POSITIONS (sell at open):")
        for o in sells:
            pos = next(
                (p for p in state.positions
                 if p.ticker == o.ticker and p.cohort_id == o.cohort_id),
                None,
            )
            # State and orders can disagree if state was hand-edited between
            # runs; report the leg rather than failing the whole signal.
            pnl = (o.price - pos.entry_price) * o.shares if pos else 0
            out.append(
                f"  SELL {o.shares:>4d} {o.ticker:<6s} @ ~${o.price:.2f}  (P&L {pnl:+,.0f})"
            )
        out.append("")

    if halted:
        # Stated regardless, but it must not suppress orders that exist. The
        # report once printed this banner beside "5 buys" while --confirm
        # persisted them, because it assumed halted implied no buys.
        out += ["*** KILL-SWITCH ENGAGED — no new positions ***", ""]
        if buys:
            out += [
                f"!!! {len(buys)} BUY order(s) generated while halted — this is "
                "a bug; do not execute:",
                "",
            ]
    if buys:
        out.append("NEW PICKS (buy at open):")
        out.append(f"  {'Rank':>4s}  {'Ticker':<6s}  {'Score':>7s}  {'Shares':>6s}  {'~Cost':>8s}")
        for i, o in enumerate(buys, 1):
            row = scored[scored["ticker"] == o.ticker]
            prob = row["prob"].iloc[0] if len(row) else 0
            out.append(
                f"  {i:>4d}  {o.ticker:<6s}  {prob:>7.3f}  {o.shares:>6d}  "
                f"${o.shares * o.price:>7,.0f}"
            )
        out.append("")
    elif not sells and not halted:
        out += ["No orders today (no expirations, no available cohort slots).", ""]

    non_expiring = [p for p in state.positions if p.expiry_date > today]
    if non_expiring:
        out.append("HOLD (active, not expiring):")
        for p in non_expiring:
            out.append(f"  {p.shares:>4d} {p.ticker:<6s}  (expires {p.expiry_date})")
        out.append("")

    turnover = sum(o.shares * o.price for o in buys) + sum(o.shares * o.price for o in sells)
    out += [
        f"Summary: {len(sells)} sells, {len(buys)} buys, {len(non_expiring)} holds",
        f"Estimated turnover: ${turnover:,.0f}",
        "=" * 60,
        "",
    ]
    return out


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
    """Print the daily signal. Formatting lives in :func:`format_signal_report`."""
    for line in format_signal_report(
        scored, orders, state, nav=nav, drawdown=drawdown, halted=halted,
        top_n=top_n, max_drawdown=max_drawdown,
    ):
        print(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily inference: score S&P 500 universe, generate orders.",
    )
    # Not argparse-required: `--init` alone is a valid invocation that just
    # creates a portfolio file. main() enforces it for every other path.
    p.add_argument("--model", type=Path, default=None, help="Path to model .pkl")
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
    # These reach the shared execution core, so a configuration measured in
    # the backtest can actually be traded. Before, they were simulation-only.
    p.add_argument(
        "--max-model-age-years", type=float, default=None,
        dest="max_model_age_years",
        help="Refuse to trade on a model whose training data ends more than "
             "this long ago (default 2.0). The model deployed here until "
             "2026-08-21 was 3.64 years stale and nothing said so. 0 disables.",
    )
    p.add_argument(
        "--max-data-age-sessions", type=int, default=None,
        dest="max_data_age_sessions",
        help="Refuse to trade on a price panel more than this many exchange "
             "sessions behind (default 3). 0 disables.",
    )
    p.add_argument(
        "--allow-stale", action="store_true", dest="allow_stale",
        help="Downgrade staleness from a block to a warning. Deliberate and "
             "visible, rather than the silent default it used to be.",
    )
    p.add_argument(
        "--min-recent-coverage", type=float, default=DEFAULT_MIN_RECENT_COVERAGE,
        dest="min_recent_coverage",
        help="Fraction of the most recent sessions a name needs before it may "
             "be ranked (default: %(default)s). Coverage and rankability are "
             "different questions: a name whose last three weeks are missing "
             "counts as downloaded but its momentum features span the hole. "
             "Pass 0 to disable.",
    )
    p.add_argument("--rank-offset", type=int, default=0, dest="rank_offset",
                   help="Skip this many top-ranked names before selecting, so the "
                        "book trades the band rank_offset+1..rank_offset+top_n")
    p.add_argument("--min-prob", type=float, default=None, dest="min_prob",
                   help="Score floor: never buy a name scoring below this")
    p.add_argument("--min-cross-section", type=int, default=None,
                   dest="min_cross_section",
                   help="Fewest scored names a date must carry before opening "
                        "positions (default: rank_offset + top_n)")
    p.add_argument("--max-drawdown", type=float, default=0.15, dest="max_drawdown")
    p.add_argument("--slippage-bps", type=float, default=5.0, dest="slippage_bps")
    p.add_argument("--skip-earnings", action="store_true", dest="skip_earnings")
    p.add_argument("--confirm", action="store_true",
                   help="Update portfolio state with new orders (without this flag: dry run)")
    p.add_argument("--sample-n", type=int, default=500, dest="sample_n",
                   help="Cap the universe at this many tickers, drawn as a "
                        "seeded RANDOM sample (not an alphabetical prefix). "
                        "Use the same value as training")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for the universe sample; defaults to the seed "
                        "recorded in the model metadata so the live universe "
                        "matches training")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="Symbols per yfinance request (default 100); lower it "
                        "if Yahoo throttles a large universe")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   dest="min_coverage",
                   help="Fail if the price download returns fewer than this "
                        "fraction of the CURRENT index members (0 = warn only)")
    p.add_argument(
        "--provider",
        default="yfinance",
        choices=["yfinance", "tiingo", "hybrid"],
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
        "--rebalance-day",
        default=None,
        dest="rebalance_day",
        choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        help="Only open a cohort on this weekday, matching the backtest's "
        "one-signal-per-week schedule. Without it any confirmed run may open "
        "a cohort, so a daily cron trades a different strategy from the one "
        "backtested. Expiries always settle either way",
    )
    p.add_argument(
        "--force-rebalance",
        action="store_true",
        dest="force_rebalance",
        help="Bypass the rebalance-day gate and the repeat-signal guard",
    )
    p.add_argument(
        "--one-lot-per-ticker",
        action="store_true",
        dest="one_lot_per_ticker",
        help="Cap each ticker at a single lot. Default matches "
        "backtest-sp500 --mode cohort, which lets a persistently top-ranked "
        "name sit in overlapping cohorts at double weight",
    )
    p.add_argument(
        "--no-macro-merge",
        action="store_true",
        help="Disable Yahoo↔FRED macro cross-fill during feature build",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    provider = get_provider(args.provider, batch_size=args.batch_size)

    # Load or init state
    if args.init:
        state = init_state(args.initial_capital)
        save_state(state, args.state)
        print(f"Initialized portfolio: {args.state} (${args.initial_capital:,.0f})")
        if not args.model:
            return

    if not args.model:
        sys.exit("--model is required unless you are only running --init.")

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
    live_members = current_members(stints)
    trained_on = meta.get("universe")
    if trained_on:
        # Use the draw the model was actually fitted on. Reseeding here samples
        # a 400-day population rather than the training window's union, so the
        # same seed yields a different set -- at sample_n=500 only 307 of 500
        # names overlapped, changing every cross-sectional rank the model reads.
        sample = resolve_live_universe(trained_on, live_members)
        drift = universe_drift(trained_on, live_members)
        print(f"  Universe: {len(sample)} tickers from the model's own draw "
              f"(hash {meta.get('universe_hash', '?')})")
        if drift["current_not_in_training"]:
            print(
                f"  Note: {int(drift['current_not_in_training'])} current index "
                f"member(s) postdate this model and are not traded "
                f"({drift['coverage_of_current']:.0%} of current membership "
                "covered). Retrain to pick them up.",
                file=sys.stderr,
            )
    else:
        # Models trained before the universe was recorded cannot have their
        # draw reproduced; say so rather than pretend the seed is enough.
        seed = resolve_universe_seed(args.seed, meta)
        sample = sample_tickers(tickers, args.sample_n, seed=seed)
        print(f"  Universe: {len(sample)} tickers (reseeded, seed={seed})")
        if args.sample_n < len(tickers):
            print(
                "  Warning: this model records no training universe, and a "
                "capped --sample-n reseeded from a different population is NOT "
                "the draw it was fitted on. Retrain to record it.",
                file=sys.stderr,
            )
        mismatch = sample_mismatch_warning(args.sample_n, meta)
        if mismatch:
            print(f"  Warning: {mismatch}", file=sys.stderr)

    # A holding is a position, not a candidate. It must be priced even after it
    # leaves the index, or its exit falls through to a fabricated fill. The
    # entry universe is the model's draw intersected with current membership,
    # which by construction excludes departed names.
    held = sorted({p.ticker for p in state.positions})
    orphaned = [t for t in held if t not in set(sample)]
    if orphaned:
        print(f"  Adding {len(orphaned)} held ticker(s) outside the entry "
              f"universe so they can be priced: {', '.join(orphaned[:8])}"
              + (" …" if len(orphaned) > 8 else ""))
    download = sorted(set(sample) | set(held))
    adj_close, volume = download_recent_prices(
        download, lookback_days=lookback, provider=provider,
    )
    print(f"  Downloaded: {adj_close.shape[0]} days x {adj_close.shape[1]} tickers")
    check_download_coverage(
        sample, adj_close,
        min_coverage=args.min_coverage,
        active=current_members(stints),
        label="equity download",
    )

    # Required whenever the model was trained with them, or scoring dies on
    # missing feature columns.
    fundamentals = None
    if any(c.startswith("fund_") for c in feature_cols):
        from stock_predictor.fundamentals import fetch_fundamentals

        print("Model uses fundamental features; fetching SEC EDGAR facts...")
        fundamentals = fetch_fundamentals(
            sample, cache_dir=Path("artifacts/edgar_cache"),
        )
        n_tk = fundamentals["ticker"].nunique() if len(fundamentals) else 0
        print(f"  {len(fundamentals):,} facts for {n_tk} tickers")

    # Build features
    print("Computing features...")
    panel, inferred_cols = build_inference_panel(
        adj_close, volume, stints,
        start=start_date, end=None,
        skip_earnings=args.skip_earnings,
        provider=provider,
        provider_name=args.provider,
        macro_merge=not args.no_macro_merge,
        fundamentals=fundamentals,
    )

    # Validate feature alignment
    missing_feats = missing_feature_columns(panel, feature_cols)
    if missing_feats:
        print(f"Warning: model expects features not in panel: {missing_feats}", file=sys.stderr)
        print("Run with matching --skip-earnings flag as training.", file=sys.stderr)

    # A name can be fully "covered" by the download check and still be
    # unrankable today: its recent sessions are what the cross-sectional
    # features are built from. Drop those rather than rank them on a feature
    # computed across the hole.
    unrankable: list[str] = []
    if args.min_recent_coverage > 0:
        # Only names that survived the PIT filter are candidates. Judging the
        # raw download instead flags every delisted symbol at 0% -- 192 of 845
        # on this universe -- which buries the handful that actually matter.
        live_names = [c for c in adj_close.columns if c in set(panel["ticker"])]
        unrankable = thin_recent_names(
            adj_close[live_names], sessions=DEFAULT_RECENT_SESSIONS,
            min_fraction=args.min_recent_coverage,
        )
        if unrankable:
            print(
                f"  Excluding {len(unrankable)} name(s) with under "
                f"{args.min_recent_coverage:.0%} of the last "
                f"{DEFAULT_RECENT_SESSIONS} sessions priced: "
                f"{', '.join(unrankable[:8])}"
                + (" …" if len(unrankable) > 8 else ""),
                file=sys.stderr,
            )
            panel = panel[~panel["ticker"].isin(unrankable)]

    # Score
    print("Scoring universe...")
    scored = score_universe(model, panel, feature_cols)
    label = score_label(model)
    print(f"  Scored {len(scored)} tickers. Top-5:")
    for _, row in scored.head(5).iterrows():
        print(f"    {row['ticker']:<6s}  {label}={row['prob']:.3f}  "
              f"@ ${row['adj_close']:.2f}")

    # Stale inputs are checked after the panel is built, so the data age is
    # measured on what would actually be traded on, and before any order is
    # generated.
    policy_kwargs = {}
    if args.max_model_age_years is not None:
        policy_kwargs["max_model_age_years"] = args.max_model_age_years
    if args.max_data_age_sessions is not None:
        policy_kwargs["max_data_age_sessions"] = args.max_data_age_sessions
    findings = check_freshness(
        meta, pd.DatetimeIndex(adj_close.index), policy=FreshnessPolicy(**policy_kwargs),
    )
    if findings:
        print(describe(findings), file=sys.stderr)
        if not args.allow_stale:
            sys.exit(
                "Refusing to trade on stale inputs. Retrain, refresh the data, "
                "or pass --allow-stale to proceed anyway."
            )
        print("  --allow-stale: proceeding anyway.", file=sys.stderr)

    # Quotes come from the raw download, not the scored panel. The panel is
    # point-in-time filtered, so a departed holding has no row in it and would
    # read as "no quote" even when it was downloaded successfully.
    #
    # Two dictionaries, because there are two questions. `latest_prices`
    # executes and holds only prices that printed on the session being traded;
    # `marks` values and may carry a price forward. Building one forward-filled
    # dict for both is what let a holding sell at a price that did not exist:
    # valid_quote() never saw the missing quote, because ffill had already
    # replaced it upstream.
    latest_prices = dict(zip(scored["ticker"], scored["adj_close"]))
    marks = dict(latest_prices)
    ages: dict[str, int] = {}
    quote_dates: dict[str, pd.Timestamp] = {}
    if len(adj_close):
        session_px = execution_quotes(adj_close)
        marked_px = valuation_marks(adj_close)
        ages = quote_ages(adj_close)
        quote_dates = last_quote_dates(adj_close)
        for t in held:
            if t in session_px:
                latest_prices[t] = session_px[t]
            else:
                # No print this session: it must not be executable. Drop any
                # value the scored panel may have contributed for this name.
                latest_prices.pop(t, None)
            if t in marked_px:
                marks[t] = marked_px[t]

    # Kill-switch check. Valued on `marks`: an unquoted holding must still be
    # markable or it falls back to its entry price and can never show a loss.
    halted, nav, dd = check_kill_switch(state, marks, args.max_drawdown)
    # Staleness measured on the executable dict, which is the one that can be
    # empty. Measured on the forward-filled dict this never fired.
    stale = stale_positions(state, latest_prices)
    gaps = describe_quote_gaps(
        stale, ages, marks, quote_dates, last_session=adj_close.index[-1],
    ) if len(adj_close) and stale else ""
    if gaps:
        print(gaps, file=sys.stderr)

    # Generate orders (calendar = session dates in downloaded OHLC index)
    trading_dates = trading_dates_from_index(adj_close.index)
    if args.hold_mode == "rank":
        # Rank exits need the FULL ranking, not just the top of the list.
        orders, new_state = generate_orders_rank_hold(
            state, scored.to_dict("records"), latest_prices,
            top_n=args.top_n,
            exit_rank=args.exit_rank,
            rank_offset=args.rank_offset,
            min_prob=args.min_prob,
            min_cross_section=args.min_cross_section,
            slippage_bps=args.slippage_bps,
            as_of=date.today().isoformat(),
            trading_dates=trading_dates,
            commission_per_share=args.commission_per_share,
            commission_per_order=args.commission_per_order,
            allow_buys=not halted,
            rebalance_day=args.rebalance_day,
            force=args.force_rebalance,
        )
    else:
        # The complete cross-section, as rank-hold already does. Truncating to
        # top_n * 2 broke every rule that reads beyond the head: a rank_offset
        # of 10 on 10 rows selects nothing, and 10 rows sits below the
        # cross-section floor of rank_offset + top_n.
        picks = scored.to_dict("records")
        orders, new_state = generate_orders(
            state, picks, latest_prices,
            top_n=args.top_n,
            max_cohorts=args.max_cohorts,
            holding_days=args.holding_days,
            rank_offset=args.rank_offset,
            min_prob=args.min_prob,
            min_cross_section=args.min_cross_section,
            slippage_bps=args.slippage_bps,
            as_of=date.today().isoformat(),
            trading_dates=trading_dates,
            weighting=args.weighting,
            commission_per_share=args.commission_per_share,
            commission_per_order=args.commission_per_order,
            allow_buys=not halted,
            allow_duplicate_holdings=not args.one_lot_per_ticker,
            rebalance_day=args.rebalance_day,
            force=args.force_rebalance,
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
