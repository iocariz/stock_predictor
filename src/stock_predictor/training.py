"""Training pipeline: features, Optuna, LightGBM, evaluation."""

from __future__ import annotations

import io
import json
import os
import pickle
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from stock_predictor.calendar_features import CALENDAR_FEATURE_COLS, add_calendar_features
from stock_predictor.data_provider import _load_dotenv
from stock_predictor.macro_merge import download_macro_fred, merge_macro_panels
from stock_predictor.pit import filter_panel_to_pit

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MACRO_YF = ["^VIX", "^TNX", "^IRX"]

PRICE_FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "ret_252d",
    "momentum",
    "vol_10d",
    "vol_21d",
    "rsi_14",
    "price_vs_ma20",
    "price_vs_ma50",
    "bb_pos",
    "high_52w_pct",
    "drawdown_63d",
]
VOLUME_FEATURE_COLS = [
    "volume_zscore",
    "volume_ratio_5d",
    "volume_trend_10d",
    "price_vol_divergence",
]
MACRO_FEATURE_COLS = ["vix", "vix_ret_5d", "tnx_yield", "yield_curve_spread", "vix_percentile"]
REGIME_FEATURE_COLS = ["market_ret_5d", "market_ret_21d"]
RANK_FEATURE_COLS = ["ret_21d_rank", "vol_10d_rank", "volume_zscore_rank"]
EARNINGS_FEATURE_COLS = ["days_since_last_earnings"]

# GICS reclassification effective 2018-09-28: Communication Services was
# carved from parts of IT, Consumer Discretionary, and the old Telecom sector.
# Source: S&P Dow Jones / MSCI GICS structure change, Sep 2018.
GICS_2018_CUTOFF = pd.Timestamp("2018-09-28")
_GICS_2018_PRE_SECTOR: dict[str, str] = {
    # Information Technology → Communication Services
    "ATVI": "Information Technology",
    "EA": "Information Technology",
    "FB": "Information Technology",
    "GOOG": "Information Technology",
    "GOOGL": "Information Technology",
    "META": "Information Technology",
    "TTWO": "Information Technology",
    # Consumer Discretionary → Communication Services
    "CBS": "Consumer Discretionary",
    "CHTR": "Consumer Discretionary",
    "CMCSA": "Consumer Discretionary",
    "DIS": "Consumer Discretionary",
    "DISCA": "Consumer Discretionary",
    "DISCK": "Consumer Discretionary",
    "DISH": "Consumer Discretionary",
    "FOX": "Consumer Discretionary",
    "FOXA": "Consumer Discretionary",
    "IPG": "Consumer Discretionary",
    "NFLX": "Consumer Discretionary",
    "NWS": "Consumer Discretionary",
    "NWSA": "Consumer Discretionary",
    "OMC": "Consumer Discretionary",
    "VIAB": "Consumer Discretionary",
    # Telecommunication Services → Communication Services (sector renamed)
    "CTL": "Telecommunication Services",
    "LUMN": "Telecommunication Services",
    "T": "Telecommunication Services",
    "TMUS": "Telecommunication Services",
    "VZ": "Telecommunication Services",
}


def wide_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        return raw.xs(field, axis=1, level=-1).sort_index()
    return raw[[field]].sort_index()


def _close_wide_macro(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw[["Close"]].sort_index()
    if raw.columns.names[0] == "Price":
        return raw.xs("Close", level=0, axis=1).sort_index()
    return raw.xs("Close", level=-1, axis=1).sort_index()


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    g_price = df.groupby("ticker", group_keys=False)["adj_close"]
    for lag in [1, 5, 10, 21, 63, 252]:
        df[f"ret_{lag}d"] = g_price.transform(lambda s: s.pct_change(lag))
    df["momentum"] = df["ret_21d"] - df["ret_5d"]
    daily_ret = g_price.transform(lambda s: s.pct_change())
    df["vol_10d"] = daily_ret.groupby(df["ticker"]).transform(lambda s: s.rolling(10).std())
    df["vol_21d"] = daily_ret.groupby(df["ticker"]).transform(lambda s: s.rolling(21).std())

    def rsi(s: pd.Series, window: int = 14) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = g_price.transform(rsi)
    df["price_vs_ma20"] = g_price.transform(lambda s: s / s.rolling(20).mean() - 1)
    df["price_vs_ma50"] = g_price.transform(lambda s: s / s.rolling(50).mean() - 1)

    def bb_position(s: pd.Series, window: int = 20) -> pd.Series:
        ma = s.rolling(window).mean()
        std = s.rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (s - lower) / (upper - lower + 1e-9)

    df["bb_pos"] = g_price.transform(bb_position)

    df["high_52w_pct"] = g_price.transform(lambda s: s / s.rolling(252).max())
    df["drawdown_63d"] = g_price.transform(lambda s: s / s.rolling(63).max() - 1)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    g_vol = df.groupby("ticker", group_keys=False)["volume"]
    vol_ma21 = g_vol.transform(lambda s: s.rolling(21).mean())
    vol_std21 = g_vol.transform(lambda s: s.rolling(21).std())
    df["volume_zscore"] = (df["volume"] - vol_ma21) / (vol_std21 + 1e-9)
    vol_ma5 = g_vol.transform(lambda s: s.rolling(5).mean())
    df["volume_ratio_5d"] = df["volume"] / (vol_ma5 + 1e-9)
    df["volume_trend_10d"] = g_vol.transform(
        lambda s: s.rolling(10).mean() / s.rolling(21).mean() - 1
    )
    price_ret_5d = df.groupby("ticker", group_keys=False)["adj_close"].transform(
        lambda s: s.pct_change(5)
    )
    vol_chg_5d = g_vol.transform(lambda s: s.pct_change(5))
    df["price_vol_divergence"] = price_ret_5d - vol_chg_5d
    return df


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    for window in [5, 21]:
        col = f"ret_{window}d"
        sector_median = df.groupby(["date", "sector"])[col].transform("median")
        df[f"{col}_vs_sector"] = df[col] - sector_median
    sector_vol_median = df.groupby(["date", "sector"])["vol_10d"].transform("median")
    df["vol_vs_sector"] = df["vol_10d"] - sector_vol_median
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-wide regime features: cross-sectional median return per date."""
    for window in [5, 21]:
        col = f"ret_{window}d"
        df[f"market_ret_{window}d"] = df.groupby("date")[col].transform("median")
    return df


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank features cross-sectionally per date (percentile 0-1)."""
    for col in ["ret_21d", "vol_10d", "volume_zscore"]:
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)
    return df


def _fetch_earnings_index(ticker: str) -> pd.DatetimeIndex:
    try:
        time.sleep(0.05)
        ed = yf.Ticker(ticker).get_earnings_dates(limit=28)
        if ed is None or len(ed) == 0:
            return pd.DatetimeIndex([])
        return pd.to_datetime(ed.index, utc=False).normalize().sort_values().unique()
    except Exception:
        return pd.DatetimeIndex([])


def build_earnings_map(tickers: list[str], max_workers: int = 8) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_earnings_index, t): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                idx = fut.result()
            except Exception:
                idx = pd.DatetimeIndex([])
            out[t] = idx.values.astype("datetime64[ns]")
    return out


def add_days_since_last_earnings(df: pd.DataFrame, earn_map: dict[str, np.ndarray]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for t, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        earn = earn_map.get(t)
        if earn is None or len(earn) == 0:
            g["days_since_last_earnings"] = np.nan
            pieces.append(g)
            continue
        d_ns = g["date"].values.astype("datetime64[ns]")
        pos = np.searchsorted(earn, d_ns, side="right") - 1
        days = np.full(len(g), np.nan, dtype=float)
        ok = pos >= 0
        days[ok] = (d_ns[ok] - earn[pos[ok]]) / np.timedelta64(1, "D")
        g["days_since_last_earnings"] = days
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def precision_at_k(y_true: pd.Series, y_scores: np.ndarray, k: int) -> float:
    if len(y_true) < k:
        return float("nan")
    top_k_idx = np.argsort(y_scores)[-k:]
    return float(y_true.values[top_k_idx].mean())


def make_objective(
    X: pd.DataFrame, y: np.ndarray, spw: float, tsc: TimeSeriesSplit, seed: int
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        fold_scores: list[float] = []
        for train_idx, val_idx in tsc.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            clf = lgb.LGBMClassifier(
                **params,
                scale_pos_weight=spw,
                metric="average_precision",
                eval_metric="average_precision",
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            )
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(80, verbose=False, first_metric_only=True),
                    lgb.log_evaluation(0),
                ],
            )
            fold_scores.append(average_precision_score(y_va, clf.predict_proba(X_va)[:, 1]))
        return float(np.mean(fold_scores))

    return objective


def _inner_train_val_split(
    train_df: pd.DataFrame, date_col: str, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    udates = np.sort(train_df[date_col].unique())
    if len(udates) < 3:
        return train_df.iloc[:0], train_df.iloc[:0]
    k = max(1, int(len(udates) * val_frac))
    val_dates = set(udates[-k:])
    tr = train_df[~train_df[date_col].isin(val_dates)]
    va = train_df[train_df[date_col].isin(val_dates)]
    return tr, va


def monthly_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    test_start: str,
    lgb_fixed: dict,
    *,
    inner_val_frac: float,
    min_train_rows: int,
    top_k: int,
    random_state: int,
    return_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    first_p = pd.Timestamp(test_start).to_period("M")
    last_p = d[date_col].max().to_period("M")
    periods = pd.period_range(first_p, last_p, freq="M")
    lgb_core = {k: v for k, v in lgb_fixed.items() if k != "n_estimators"}
    n_est_user = int(lgb_fixed.get("n_estimators", 500))
    records: list[dict] = []
    scored_panels: list[pd.DataFrame] = []

    for p in periods:
        m_start = pd.Timestamp(year=p.year, month=p.month, day=1)
        m_end = m_start + pd.offsets.MonthEnd(0)
        train_mask = d[date_col] < m_start
        test_mask = (d[date_col] >= m_start) & (d[date_col] <= m_end)
        train_df = d.loc[train_mask]
        test_df = d.loc[test_mask]
        if len(train_df) < min_train_rows or len(test_df) == 0:
            continue
        tr_in, va_in = _inner_train_val_split(train_df, date_col, inner_val_frac)
        if len(tr_in) == 0 or len(va_in) == 0:
            continue
        y_tr = tr_in[target_col]
        neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        if neg == 0 or pos == 0:
            continue
        spw = neg / pos
        clf = lgb.LGBMClassifier(
            **lgb_core,
            n_estimators=min(2000, max(500, n_est_user * 4)),
            scale_pos_weight=spw,
            metric="average_precision",
            eval_metric="average_precision",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(
            tr_in[feature_cols],
            y_tr,
            eval_set=[(va_in[feature_cols], va_in[target_col])],
            callbacks=[
                lgb.early_stopping(80, verbose=False, first_metric_only=True),
                lgb.log_evaluation(0),
            ],
        )
        bi = clf.best_iteration_
        if bi is None:
            bi = n_est_user - 1
        n_trees = int(bi) + 1
        neg_f, pos_f = int((train_df[target_col] == 0).sum()), int((train_df[target_col] == 1).sum())
        if neg_f == 0 or pos_f == 0:
            continue
        spw_f = neg_f / pos_f
        final = lgb.LGBMClassifier(
            **lgb_core,
            n_estimators=n_trees,
            scale_pos_weight=spw_f,
            metric="average_precision",
            eval_metric="average_precision",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        final.fit(train_df[feature_cols], train_df[target_col])
        y_test = test_df[target_col]
        prob = final.predict_proba(test_df[feature_cols])[:, 1]
        pr = average_precision_score(y_test, prob)
        try:
            roc = roc_auc_score(y_test, prob)
        except ValueError:
            roc = float("nan")
        scored = test_df.assign(prob=prob)
        if return_scores:
            score_cols = [date_col, "ticker", "prob", "adj_close", "fwd_ret", target_col]
            scored_panels.append(scored[score_cols].copy())
        weekly_p = (
            scored.assign(week=lambda x: x[date_col].dt.to_period("W"))
            .groupby("week", observed=True)
            .apply(
                lambda g: precision_at_k(g[target_col], g["prob"].values, k=top_k),
                include_groups=False,
            )
        )
        w_mean = float(np.nanmean(weekly_p.values)) if len(weekly_p) else float("nan")
        records.append(
            {
                "month": str(p),
                "train_end": (m_start - pd.Timedelta(days=1)).date(),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "pr_auc": pr,
                "roc_auc": roc,
                "mean_weekly_precision_at_k": w_mean,
                "pos_rate_test": float(y_test.mean()),
                "n_trees": n_trees,
            }
        )
    metrics_df = pd.DataFrame.from_records(records)
    if return_scores:
        if scored_panels:
            scores_df = pd.concat(scored_panels, ignore_index=True)
        else:
            scores_df = pd.DataFrame(
                columns=[date_col, "ticker", "prob", "adj_close", "fwd_ret", target_col]
            )
        return metrics_df, scores_df
    return metrics_df




def build_labeled_panel(
    adj_close: pd.DataFrame,
    stints: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """Stack wide adj_close into long format, compute forward return & label."""
    long = adj_close.stack(future_stack=True).rename("adj_close").reset_index()
    long.columns = ["date", "ticker", "adj_close"]
    long = long.sort_values(["ticker", "date"])
    long["fwd_ret"] = long.groupby("ticker", group_keys=False)["adj_close"].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    long["target_5pct"] = (long["fwd_ret"] >= threshold).astype("int8")
    labeled = long.dropna(subset=["fwd_ret"])
    return filter_panel_to_pit(labeled, stints)


def _build_macro_panel_yfinance(start: str, end: str | None) -> pd.DataFrame:
    """Fetch macro data via yfinance (legacy inline path).

    Resilient to partial failures: if one or more macro tickers fail to
    download, the missing columns are filled with NaN rather than raising.
    """
    macro_dl = yf.download(
        MACRO_YF, start=start, end=end,
        group_by="ticker", threads=True, auto_adjust=False, progress=False,
    )
    mw = _close_wide_macro(macro_dl)
    rename_map = {"^VIX": "vix", "^TNX": "tnx_yield", "^IRX": "irx_yield"}
    mdf = mw.rename(columns=rename_map).reset_index()
    dcol = mdf.columns[0]
    mdf = mdf.rename(columns={dcol: "date"})
    mdf["date"] = pd.to_datetime(mdf["date"]).dt.normalize()
    # Ensure all expected columns exist even if some tickers failed
    for col in ["vix", "tnx_yield", "irx_yield"]:
        if col not in mdf.columns:
            mdf[col] = np.nan
    return mdf[["date", "vix", "tnx_yield", "irx_yield"]].copy()


def _macro_fallback_panel(
    provider_name: str,
    start: str,
    end: str | None,
    fred_key: str,
) -> pd.DataFrame | None:
    """Second macro source to combine with the provider default (fill NaN dates/columns)."""
    if provider_name == "tiingo":
        fb = _build_macro_panel_yfinance(start, end)
        return fb if not fb.empty else None
    if not fred_key:
        return None
    try:
        fb = download_macro_fred(start, end, fred_key)
    except ImportError:
        return None
    return fb if not fb.empty else None


def _derive_macro_features(mdf: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro features from raw vix / yield columns."""
    mdf = mdf.copy()
    mdf["yield_curve_spread"] = mdf["tnx_yield"] - mdf["irx_yield"]
    mdf["vix_ret_5d"] = mdf["vix"].pct_change(5)
    mdf["vix_percentile"] = mdf["vix"].rolling(252, min_periods=60).apply(
        lambda w: (w[-1] >= w[:-1]).mean(), raw=True,
    )
    return mdf[
        ["date", "vix", "vix_ret_5d", "tnx_yield", "yield_curve_spread", "vix_percentile"]
    ].copy()


def build_feature_panel(
    labeled: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    start: str,
    end: str | None,
    skip_earnings: bool,
    earnings_workers: int,
    provider: object | None = None,
    provider_name: str = "yfinance",
    macro_merge: bool = True,
    fred_api_key: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer all features and return the panel with the feature column list.

    Parameters
    ----------
    provider : DataProvider | None
        When supplied, equity/macro downloads go through this provider.
        ``None`` falls back to the legacy inline yfinance code.
    provider_name : str
        ``yfinance`` or ``tiingo`` — selects which *alternate* macro API to call
        when ``macro_merge`` is True (Yahoo vs FRED cross-fill).
    macro_merge : bool
        If True, merge the provider macro panel with the complementary source
        (FRED when using Yahoo macro, Yahoo when using Tiingo/FRED macro) so
        missing dates or columns are filled where possible.
    fred_api_key : str | None
        FRED API key for Yahoo→FRED gap-fill; ``None`` uses ``FRED_API_KEY`` env
        after optional ``python-dotenv`` load.
    """
    _wiki_headers = {"User-Agent": "stock-predictor/0.1 (Python/pandas; educational research)"}
    req = urllib.request.Request(SP500_WIKI_URL, headers=_wiki_headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    wiki_constituents = pd.read_html(io.StringIO(html))[0]

    features = add_price_features(labeled.copy())
    vol_long = volume.stack(future_stack=True).rename("volume").reset_index()
    vol_long.columns = ["date", "ticker", "volume"]
    features = features.merge(vol_long, on=["date", "ticker"], how="left")
    features = add_volume_features(features)
    features = add_regime_features(features)
    features = add_cross_sectional_ranks(features)

    sector_map = (
        wiki_constituents[["Symbol", "GICS Sector"]]
        .rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
        .assign(ticker=lambda df: df["ticker"].str.replace(".", "-", regex=False))
    )
    features = features.merge(sector_map, on="ticker", how="left")
    pre = (features["date"] < GICS_2018_CUTOFF) & features["ticker"].isin(_GICS_2018_PRE_SECTOR)
    features.loc[pre, "sector"] = features.loc[pre, "ticker"].map(_GICS_2018_PRE_SECTOR)
    features = add_sector_relative_features(features)
    features["sector"] = features["sector"].astype("category")

    try:
        if provider is not None:
            macro_raw = provider.download_macro(start, end)
        else:
            macro_raw = _build_macro_panel_yfinance(start, end)
        if macro_merge:
            _load_dotenv()
            fk = fred_api_key if fred_api_key is not None else os.environ.get("FRED_API_KEY", "")
            alt = _macro_fallback_panel(provider_name, start, end, fk)
            if alt is not None:
                macro_raw = merge_macro_panels(macro_raw, alt)
                msg = (
                    "Yahoo fills FRED gaps"
                    if provider_name == "tiingo"
                    else "FRED fills Yahoo gaps"
                )
                print(f"  Macro merge ({msg}): {len(macro_raw)} dates")
        macro_panel = _derive_macro_features(macro_raw)
    except Exception as exc:
        print("Macro download failed:", exc)
        macro_panel = pd.DataFrame(
            columns=["date", "vix", "vix_ret_5d", "tnx_yield", "yield_curve_spread", "vix_percentile"]
        )

    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    if not macro_panel.empty:
        macro_panel["date"] = pd.to_datetime(macro_panel["date"]).dt.normalize()
    else:
        # Empty DataFrame: add NaN macro columns directly to avoid dtype mismatch on merge
        for col in ["vix", "vix_ret_5d", "tnx_yield", "yield_curve_spread", "vix_percentile"]:
            features[col] = np.nan
        macro_panel = None
    if macro_panel is not None:
        features = features.merge(macro_panel, on="date", how="left")
    features = add_calendar_features(features)

    earn_cols: list[str] = []
    if skip_earnings:
        print("Skipping earnings feature (--skip-earnings).")
    else:
        print("Fetching earnings dates (Yahoo)…")
        earn_map = build_earnings_map(
            sorted(features["ticker"].unique()), max_workers=earnings_workers,
        )
        features = add_days_since_last_earnings(features, earn_map)
        earn_cols = list(EARNINGS_FEATURE_COLS)
        print(f"  Earnings NaN rate: {features['days_since_last_earnings'].isna().mean():.2%}")

    feature_cols = (
        PRICE_FEATURE_COLS
        + VOLUME_FEATURE_COLS
        + REGIME_FEATURE_COLS
        + RANK_FEATURE_COLS
        + ["ret_5d_vs_sector", "ret_21d_vs_sector", "vol_vs_sector", "sector"]
        + MACRO_FEATURE_COLS
        + CALENDAR_FEATURE_COLS
        + earn_cols
    )
    return features, feature_cols


def run_optuna_search(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    ts_cv_splits: int,
    n_trials: int,
    seed: int,
) -> dict:
    """Run Optuna TPE search over LightGBM hyperparameters."""
    train_sorted = train.sort_values(["date", "ticker"]).reset_index(drop=True)
    X_cv = train_sorted[feature_cols]
    y_cv = train_sorted["target_5pct"].to_numpy()
    neg_cv, pos_cv = int((y_cv == 0).sum()), int((y_cv == 1).sum())
    spw_cv = neg_cv / pos_cv
    tsc = TimeSeriesSplit(n_splits=ts_cv_splits)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        make_objective(X_cv, y_cv, spw_cv, tsc, seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"Optuna best CV PR-AUC: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return dict(study.best_params)


def train_final_model(
    train: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    seed: int,
) -> tuple[lgb.LGBMClassifier, int]:
    """Find n_trees via early stopping on a val split, then retrain on full train."""
    train_inner, val_inner = _inner_train_val_split(train, "date", val_frac=0.15)
    X_tr_inner = train_inner[feature_cols]
    y_tr_inner = train_inner["target_5pct"]
    X_val = val_inner[feature_cols]
    y_val = val_inner["target_5pct"]

    neg_inner, pos_inner = int((y_tr_inner == 0).sum()), int((y_tr_inner == 1).sum())
    spw_inner = neg_inner / pos_inner
    es_model = lgb.LGBMClassifier(
        **params,
        scale_pos_weight=spw_inner,
        metric="average_precision",
        eval_metric="average_precision",
        random_state=seed,
        n_jobs=-1,
    )
    es_model.fit(
        X_tr_inner, y_tr_inner,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False, first_metric_only=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iter = es_model.best_iteration_
    n_trees = int(best_iter) + 1 if best_iter is not None else params.get("n_estimators", 500)
    print(f"Best iteration (from val split): {n_trees}")

    y_train = train["target_5pct"]
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    print(f"scale_pos_weight: {spw:.1f}")
    final_params = {k: v for k, v in params.items() if k != "n_estimators"}
    model = lgb.LGBMClassifier(
        **final_params,
        n_estimators=n_trees,
        scale_pos_weight=spw,
        metric="average_precision",
        eval_metric="average_precision",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], y_train)
    return model, n_trees


def evaluate_test_set(
    model: lgb.LGBMClassifier,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, float, pd.Series]:
    """Score the test set and return PR-AUC, ROC-AUC, and weekly precision@10."""
    y_test = test["target_5pct"]
    y_prob = model.predict_proba(test[feature_cols])[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"PR-AUC:  {pr_auc:.4f} (baseline {y_test.mean():.4f})")
    print(f"ROC-AUC: {roc_auc:.4f}")

    weekly_precision = (
        test.assign(prob=y_prob)
        .assign(week=lambda df: pd.to_datetime(df["date"]).dt.to_period("W"))
        .groupby("week", observed=True)
        .apply(
            lambda g: precision_at_k(g["target_5pct"], g["prob"].values, k=10),
            include_groups=False,
        )
    )
    print(f"Mean weekly Precision@10: {weekly_precision.mean():.4f}")
    return pr_auc, roc_auc, weekly_precision


def save_eval_plots(
    plots_dir: Path,
    weekly_precision: pd.Series,
    y_test: pd.Series,
    feature_cols: list[str],
    model: lgb.LGBMClassifier,
) -> None:
    """Save weekly precision@10 and feature importance PNGs."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 3))
    weekly_precision.plot(ax=ax, title="Weekly Precision@10 (test)")
    ax.axhline(y_test.mean(), color="red", linestyle="--", label="baseline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "weekly_precision_at_10.png", dpi=120)
    plt.close(fig)

    importance = pd.Series(
        model.feature_importances_, index=feature_cols,
    ).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    importance.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_title("LightGBM feature importance")
    plt.tight_layout()
    fig2.savefig(plots_dir / "feature_importance.png", dpi=120)
    plt.close(fig2)
    print(f"Saved plots under {plots_dir}")


def save_model_artifacts(
    output_path: Path,
    model: lgb.LGBMClassifier,
    meta: dict,
    optuna_best: dict,
) -> None:
    """Pickle the model and write JSON metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in meta.items() if k != "optuna_best"},
            f, indent=2, default=str,
        )
    if optuna_best:
        with open(output_path.with_suffix(".optuna.json"), "w", encoding="utf-8") as f:
            json.dump(optuna_best, f, indent=2)
    print(f"Saved model to {output_path} (metadata: {meta_path})")

