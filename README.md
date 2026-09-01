# stock-predictor

LightGBM model that ranks S&P 500 stocks by expected 10-day performance. Two training objectives are supported: a **binary classifier** (probability of gaining ≥ 5% over the next 10 sessions) and a **lambdarank ranker** (`--rank-objective`) trained on per-date forward-return quintile grades — a cross-sectional target that is market-neutral by construction. Built for educational research into quantitative equity screening—not production trading.

> ## ⚠️ Performance figures in this file are obsolete
>
> A clean, verified rebuild on 2026-08-28 supersedes every performance number
> below. See **[BASELINE.md](BASELINE.md)** for what replaces them and what
> verified it. Two findings change how the rest of this document should be read:
>
> - **One engine does show alpha, and it is not the one being traded.** The
>   dollar-neutral long-short book returns **16.92% ± 1.19%** with alpha
>   **+8.90% (HAC t = +2.76)** on beta **+0.21** and a **−11.5%** drawdown,
>   with all four runs above |t| = 2. It is now reachable as
>   `backtest-sp500 --mode long-short`. Read it against the multiplicity
>   caveat below: this configuration was selected from a large search.
> - **The long-only engines show none.** Against SPY on excess returns, the cohort engine
>   — the one the live path simulates — shows **+2.36%/yr (HAC t = +0.37)** on
>   beta **+1.190**; rank-hold shows **+6.80% (t = +0.84)** on beta **+1.431**.
>   Neither is distinguishable from zero, and both carry well over one unit of
>   market risk, so most of what they beat the index by is leverage.
> - **A run can now be reproduced exactly.** `train-sp500 --replay-snapshot`
>   rebuilds from a run's verified snapshot; two replays produce byte-identical
>   panels and reproduce the original backtest to four decimals. Use it to
>   compare code changes. *Fresh* rebuilds still draw new vendor noise:
> - **The headline number is not stable across fresh rebuilds.** Four rebuilds from one commit, one
>   pinned data window and one seed produced cohort CAGRs spanning 17.20% to
>   23.12% — **20.24% ± 2.45%**. SPY returned 17.60% over the same window. Vendor float noise flips LightGBM splits, and a different fifteen
>   names get held. Every figure quoted below to two decimals implies a
>   precision that does not exist, and most historical comparisons between
>   configurations are smaller than this noise floor.
>
> The findings *about defects* below remain accurate and are worth reading; the
> performance figures attached to them are not.

The evaluation pipeline is deliberately conservative: purged walk-forward splits (no forward-return label ever straddles a train/test boundary), cash-ledger backtest NAV (realized P&L, slippage, and commissions all compound through cash), benchmark comparison with CAPM alpha/beta, information ratio, and alpha t-statistics, and sweep tooling for out-of-sample sub-window checks.

## Setup

Requires **Python ≥ 3.12**. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
# optional: pytest
uv sync --extra dev
```

The package installs in editable mode from `src/` as **`stock_predictor`**. Jupyter and notebook dependencies are included in the default install.

**Console commands** (after `uv sync`): `train-sp500`, `backtest-sp500`, and `predict-sp500` (see `[project.scripts]` in `pyproject.toml`).

Without uv:

```bash
pip install -e ".[dev]"
```

## Production workflow setup

This project is still research-focused, but you can run a stable "production-like" workflow for repeatable training and daily signal generation.

### 1) One-time setup

1. Install dependencies:

```bash
uv sync --extra dev --extra tiingo
```

2. Configure API keys in your shell or `.env`:
   - `FRED_API_KEY` (recommended for macro merge quality)
   - `TIINGO_API_KEY` (required only when using `--provider tiingo`)
3. Ensure `artifacts/` is writable (model, plots, and walk-forward scores are written there by default).

### 2) Train workflow (GitHub Actions)

Use the built-in workflow at `.github/workflows/train-sp500.yml`:

1. In GitHub, open **Settings -> Secrets and variables -> Actions**.
2. Add:
   - `FRED_API_KEY`
   - `TIINGO_API_KEY` (optional unless you select `tiingo`)
3. Run **Actions -> train-sp500 -> Run workflow**.
4. Choose:
   - `provider`: `yfinance` (default) or `tiingo`
   - `sample_n`: e.g. `10000` for near-full universe
   - `extra_train_args`: extra flags appended to `train-sp500` (e.g. `--rank-objective --start 2015-01-01 --test-start 2019-01-01`)
   - `sweep_run_id` + `sweep_args`: **sweep-only mode** — skip training and run `scripts/backtest_sweep.py` against a previous run's `wf_scored.parquet` artifact (e.g. `sweep_args: --grid hold --until-date 2022-12-31`); finishes in under a minute
5. Download workflow artifacts:
   - `model-and-scores-*` (`model.pkl`, `wf_scored.parquet`)
   - `plots-*`
   - `run-snapshots-*` (if generated)

### 3) Daily predict workflow

Choose one operational mode:

- Local/VM cron + `scripts/run_pipeline.sh predict` (simplest for persistent state).
- GitHub Actions only if you also persist `portfolio_state.json` between runs (artifact restore, external store, or self-hosted runner disk).

Recommended rollout:

1. Run dry mode first:

```bash
./scripts/run_pipeline.sh predict
```

2. Switch to state-updating mode only after validation:

```bash
./scripts/run_pipeline.sh predict --confirm
```

3. Add monitoring/alerts for failed runs and kill-switch events before fully automating.

**Strategy settings are defined once and applied to both sides.**
`run_pipeline.sh` builds one list of selection flags and passes it to
`backtest-sp500` *and* `predict-sp500`, so the configuration you measure is the
configuration you trade. It previously passed them to `predict` only —
`backtest` fell back to its own defaults, which happened to agree, so setting
`TOP_N=25` would have traded twenty-five names against a simulation of fifteen
with nothing looking wrong.

| variable | default | applies to |
|---|---|---|
| `HORIZON` | 63 | training; **`HOLDING_DAYS` derives from it** |
| `TOP_N`, `HOLDING_DAYS`, `MAX_COHORTS` | 15, `$HORIZON`, 2 | both |
| `WEIGHTING`, `SLIPPAGE_BPS` | equal, 5 | both |
| `EXIT_RANK`, `RANK_OFFSET` | 40, 0 | both |
| `MIN_PROB`, `MIN_CROSS_SECTION` | unset (omitted) | both |
| `COMMISSION_PER_SHARE`, `COMMISSION_PER_ORDER` | 0, 0 | both |
| `REBALANCE_DAY` | `Friday` (`any` to trade every session) | both |
| `HOLD_MODE` | `fixed` (`rank` for rank-decay exits) | both |
| `MAX_DD` | 0.15 | live only — the kill switch has no simulation twin |

`HOLDING_DAYS` **derives from `HORIZON`** rather than being a second number to
keep in step. Trading a 63-day signal on a 10-day exit is the same class of
mismatch as measuring one configuration and trading another, and it is not
something to rely on remembering. Override it explicitly if you want them to
differ; it still moves both sides together.

Training defaults reproduce the deployed model: `--rank-objective`,
`--horizon 63`, `--skip-earnings`, `--no-optuna`, `--provider hybrid`, trained
2010→2024. A monthly retrain therefore rebuilds the model that is deployed
instead of silently reverting to an older configuration. `TRAIN_PROVIDER` is
`hybrid` (an unbiased training panel needs the delisted names) while
`PREDICT_PROVIDER` is `yfinance` (a live run only ever trades current index
members, and should not spend Tiingo quota on names it could not buy).

`REBALANCE_DAY` is the one that changed behaviour: `backtest-sp500` defaults to
Friday and `predict-sp500` to *any* day, so a daily cron was opening cohorts on
a schedule the backtest never simulated. The pipeline now pins both to Friday.

Check what a run will do without running it:

```bash
DRY_RUN=1 TOP_N=25 MIN_PROB=0.4 ./scripts/run_pipeline.sh predict
DRY_RUN=1 TOP_N=25 MIN_PROB=0.4 ./scripts/run_pipeline.sh backtest
```

`tests/test_pipeline_flags.py` drives the real script and fails if the two
commands ever disagree on a selection flag.

> `train-sp500 --run-backtest` is a **sanity check at library defaults** — that
> command has no strategy flags — and it now prints the config it used. For the
> configuration you trade, use `run_pipeline.sh backtest`.

## Usage

### Training CLI

```bash
uv run train-sp500 --help

# Fast smoke run (small universe, no Optuna, no earnings fetch)
uv run train-sp500 --sample-n 100 --no-optuna --skip-earnings

# Lambdarank objective (cross-sectional quintile grades; NDCG-tuned Optuna;
# ranker saved as the model artifact and used in the walk-forward)
uv run train-sp500 --rank-objective --output-model artifacts/model.pkl

# Equivalent launcher at repo root
uv run python train_sp500.py --sample-n 100 --no-optuna --skip-earnings

# Model + plots + reproducibility snapshots (default: artifacts/runs/<run_id>/)
uv run train-sp500 --output-model artifacts/model.pkl --plots-dir artifacts/plots

# Emit walk-forward scores for the backtest CLI, then backtest
uv run train-sp500 --wf-scores-path artifacts/wf_scored.parquet --plots-dir artifacts/plots
uv run backtest-sp500 artifacts/wf_scored.parquet --plots-dir artifacts/plots

# Or run the portfolio simulation in the same train process (needs walk-forward)
uv run train-sp500 --run-backtest --plots-dir artifacts/plots

# Custom snapshot root or disable snapshots entirely
uv run train-sp500 --snapshot-dir ./my_run_data
uv run train-sp500 --no-snapshot
```

`--run-backtest` and `--wf-scores-path` require walk-forward (do **not** pass `--skip-walk-forward`).

### Backtest CLI

**Input:** a Parquet or CSV panel from walk-forward scoring (`--wf-scores-path` or the notebook). Required columns: **`date`**, **`ticker`**, **`adj_close`**, and either **`prob`** or **`probability`**. Optional **`vix_percentile`** enables **`--vix-filter`** and the `vix_scale_exposure` config option — and those options now **raise** when the column is absent rather than silently doing nothing.

Two portfolio engines share the same cash-ledger NAV discipline:

- **`--mode cohort`** (default): weekly top-N baskets held exactly `--holding-days` sessions, up to `--max-cohorts` overlapping, funded from the compounding cash pool.
- **`--mode rank-hold`**: one continuously managed portfolio — buy the top-N, sell a holding only when its cross-sectional rank decays beyond `--exit-rank` (or it leaves the scored universe). Turnover is driven by signal decay instead of the calendar, which cuts trading costs substantially at wide exit thresholds.

```bash
uv run backtest-sp500 --help

uv run backtest-sp500 path/to/wf_scored.parquet --plots-dir artifacts/plots
uv run python backtest.py path/to/wf_scored.parquet --plots-dir artifacts/plots

# Rank-based holding instead of fixed 10-day baskets
uv run backtest-sp500 scores.parquet --mode rank-hold --exit-rank 40

# Offline (no benchmark download); optional rebalance / benchmark ticker
uv run backtest-sp500 scores.parquet --no-benchmark --rebalance-day last
uv run backtest-sp500 scores.parquet --benchmark-ticker RSP --plots-dir artifacts/plots
```

Benchmark prices are **reindexed to the strategy’s trading days** (forward-filled) so strategy vs buy-and-hold metrics use the **same calendar**. When a benchmark is present, the report also prints an **ACTIVE vs benchmark** section: annualized active return, tracking error, information ratio, CAPM beta/alpha with its **t-statistic**, up/down capture, and the equity of a dollar-neutral long-strategy/short-benchmark overlay.

The alpha t-statistic is **Newey–West (HAC)**, with the lag window set to at least the holding period. A cohort strategy holding positions for `--holding-days` sessions produces strongly autocorrelated daily returns, and an i.i.d. standard error is not valid on that series. The naive statistic is printed underneath as `vs naive i.i.d. t-stat` — the correction moves the t-stat in **either** direction depending on the sign of the residual autocorrelation, so it is a correction, not a haircut. Comparing against **`--benchmark-ticker RSP`** (equal-weight S&P 500) isolates stock-selection skill from the equal-weight-vs-cap-weight effect.

#### Backtest flags

| Flag | Default | Description |
|------|---------|-------------|
| `scored_path` | — | Parquet (`.parquet`) or CSV with a `date` column |
| `--mode` | cohort | `cohort` (fixed `--holding-days` baskets) or `rank-hold` (sell on rank decay) |
| `--top-n` | 15 | Number of tickers per cohort / portfolio slots (ranked by score) |
| `--holding-days` | 10 | Cohort mode: holding period in **trading** days |
| `--exit-rank` | 40 | Rank-hold mode: sell held names ranked worse than this (must be ≥ top-n) |
| `--rebalance-day` | Friday | `Monday`–`Friday` or `last` (last session in each ISO week) |
| `--weighting` | equal | `equal` or `probability` (normalize scores to weights). `probability` **rejects negative scores** — use `equal` with `--rank-objective` models |
| `--slippage-bps` | 5 | Round-trip modeled as **per-side** basis points |
| `--capital` | 100000 | Starting notional |
| `--max-cohorts` | 2 | Cohort mode: overlapping cohorts (cash / free slots per entry) |
| `--vix-filter` | none | Skip rebalance (cohort) / block buys (rank-hold) when `vix_percentile` exceeds this. **Errors** if the panel has no `vix_percentile` column |
| `--min-prob` | none | Score floor: never buy a name scoring below this. Baskets shrink and weights renormalize; a date with no eligible name does not trade |
| `--min-cross-section` | `rank_offset + top_n` | Fewest scored names a date must carry before it may **open** positions. Exits are never gated, so a narrowing cross-section cannot strand a holding. Stops a ragged panel edge being traded as if it were a ranking |
| `--min-recent-coverage` | 0.8 | *(live only)* Fraction of the last 20 sessions a name needs before it may be **ranked**. Coverage and rankability are different questions — a name whose last three weeks are missing counts as downloaded, but its momentum features span the hole. `0` disables |
| `--max-model-age-years` | 2.0 | *(live only)* Refuse to trade on a model whose training data ends more than this long ago. `0` disables |
| `--max-data-age-sessions` | 3 | *(live only)* Refuse to trade on a price panel this many **exchange sessions** behind. `0` disables |
| `--allow-stale` | off | Downgrade a staleness block to a warning — deliberately and visibly |
| `--rf-rate` | inferred | Annualized risk-free rate for Sharpe/Sortino. **Funding costs are on by default**: the panel's realized `irx_yield` (13-week T-bill) is charged per date when present, else a 4.5% cash proxy. Pass `0` to switch it off. The rate applied is printed in the report header |
| `--benchmark-ticker` | SPY | yfinance symbol for buy-and-hold column |
| `--no-benchmark` | off | Skip benchmark download (table shows N/A for benchmark) |
| `--plots-dir` | none | Writes `equity_curve.png`, `drawdown.png`, `monthly_returns.png` |
| `--compare-with` | none | Second scored Parquet/CSV; same rules as primary (shared flags) |
| `--compare-label-a` | stem of primary file | Legend label for primary strategy |
| `--compare-label-b` | stem of compare file | Legend label for `--compare-with` |
| `--commission-per-share` | 0 | Dollars per share on each buy and sell leg (0 = off) |
| `--commission-per-order` | 0 | Flat dollars per ticker on each buy and sell order (0 = off) |

**Config-only option:** `BacktestConfig(vix_scale_exposure=True)` (Python API and the sweep's `--grid vix`) scales new-entry capital by the VIX regime instead of — or on top of — the hard skip: full size while `vix_percentile ≤ 0.5`, linearly down to zero at 1.0, with the remainder held in cash. Produces a lower-beta, lower-drawdown variant.

#### Execution assumptions (backtest vs predict)

- **Calendar:** Historical sessions are the **union of session dates present in the data** (ground truth for the past). *Future* sessions — the entry and expiry dates a live run must place beyond the end of the price history — come from the **NYSE exchange calendar** via `exchange_calendars`, not from business days. Business-day projection was wrong 3.6% of the time (156 of 4,337 days in this panel, ~9.1 a year): an entry could land on a closed market, and a 63-session expiry drifted ~2 sessions early. Early closes are not modelled — a shortened session still has a close, and every fill here is a close.
- **Backtest:** Signal on date *T* → **entry** on the next session in that calendar → **exit** after `--holding-days` trading sessions (cohort mode) or on rank decay (rank-hold mode). Cohort returns are **fractional** weights; slippage is applied to historical **adj. close** bars.
- **Funding:** Sharpe and Sortino are excess-return statistics, charged at the panel's realized 13-week T-bill rate (or 4.5% when a panel predates that column). Total return, CAGR and drawdown stay on raw returns.
- **NAV accounting:** the daily NAV is a **cash ledger** — capital is debited at entry and credited back at exit as `capital × (1 + net_return)`, so realized P&L, exit slippage, and commissions all compound through cash. Total return, Sharpe, and drawdown reflect costs.
- **predict-sp500:** entry is the first session **strictly after** *today* (same next-day convention as the backtest); fixed mode expiries use the same trading-day count as the backtest, rank mode positions carry an open-ended expiry sentinel and close on rank decay. Orders use **integer shares** (lot sizes differ from a fractional simulation).
- **Position sizing:** both paths fund a new cohort with `free_cash / free_slots`. (Live previously divided by `max_cohorts`, so a portfolio with one of two slots open deployed half its free cash and the remainder never got invested.)
- **Duplicate holdings:** both paths allow a persistently top-ranked name in two overlapping cohorts at double weight. Pass `--one-lot-per-ticker` to cap live at one lot, accepting that it will then under-weight names the simulation kept buying.
- **Parity:** Use the same `--weighting`, `--slippage-bps`, `--holding-days` / `--exit-rank`, and commission flags in both CLIs when comparing research simulation to generated orders. Do **not** switch `--hold-mode` on an existing state file.

#### Comparing two strategies

Use **`--compare-with`** to run the **same** portfolio rules (`--top-n`, `--holding-days`, slippage, etc.) on **two different scored panels**—for example two models, two feature sets, or two training snapshots. The CLI prints an **overlap-window** metrics table (NAV normalized to 1.0 on the first shared day) and, with **`--plots-dir`**, saves **`equity_compare.png`** (both strategies plus benchmark from the primary run, if enabled).

```bash
uv run backtest-sp500 model_a.parquet \
  --compare-with model_b.parquet \
  --compare-label-a "with earnings" --compare-label-b "no earnings" \
  --plots-dir artifacts/plots
```

In Python: `print_strategy_comparison`, `plot_strategy_comparison` in `stock_predictor.backtest`.

**Fair comparison tips:** use the **same date range and universe** in both panels when possible; keep **`--capital`** identical so the overlay is in comparable dollars.

#### Strategy sweeps (grids of variants)

[`scripts/backtest_sweep.py`](scripts/backtest_sweep.py) runs a grid of `BacktestConfig` variants over one scored panel, downloads the benchmark **once**, and prints two tables: absolute metrics per variant and the relative-return framing vs the benchmark (active return, IR, beta, alpha + t-stat, capture, overlay).

```bash
# General grid: top-N ladder, weighting, VIX filter, holding periods
uv run python scripts/backtest_sweep.py artifacts/wf_scored.parquet

# Low-beta / VIX-regime grid (skip thresholds, continuous exposure scaling)
uv run python scripts/backtest_sweep.py artifacts/wf_scored.parquet --grid vix

# Cohort vs rank-hold engines (exit-rank ladder)
uv run python scripts/backtest_sweep.py artifacts/wf_scored.parquet --grid hold

# Out-of-sample sub-window + alternate benchmark
uv run python scripts/backtest_sweep.py artifacts/wf_scored.parquet \
  --grid hold --until-date 2022-12-31 --benchmark-ticker RSP
```

The same sweeps run on GitHub Actions in ~30 s via the `train-sp500` workflow's `sweep_run_id` + `sweep_args` inputs (reuses a previous run's `wf_scored.parquet` artifact — no retraining). **Discipline note:** a grid's best cell is a hypothesis, not a result — re-test it on a sub-window it has never seen (`--from-date` / `--until-date`) and check the alpha t-stat before believing it.

#### Signal diagnostics (before you tune anything)

A backtest conflates ranking skill with market beta, costs, and sizing. [`scripts/signal_depth.py`](scripts/signal_depth.py) asks the narrower question — sort each date's universe by score, and see what the top *k* names actually returned:

```bash
uv run python scripts/signal_depth.py artifacts/wf_scored.parquet

# fair benchmark for an equal-weighted strategy, on an unseen sub-window
uv run python scripts/signal_depth.py artifacts/wf_scored.parquet \
  --benchmark-ticker RSP --until-date 2025-06-30
```

It prints forward return by selection depth, the rank IC, and CAPM alpha across a `--top-n` ladder, all with HAC t-statistics. **Read the shape, not the level.** A ranker with usable skill puts its best forward returns at the tightest bucket and *gains* alpha as you concentrate; the script fails a shape check and says so when a deeper bucket wins, which means the edge is not where a top-N strategy trades.

#### Strategy ideas to try (research / educational)

| Idea | What to vary | How to approximate here |
|------|----------------|---------------------------|
| **Model A vs model B** | Different WF score files | Train twice (`--output-model` / different seeds or features), export two `--wf-scores-path` panels, compare |
| **Equal vs probability weights** | Position sizing | Same `wf_scored.parquet`, run backtest twice with `--weighting equal` vs `probability`. Classifier panels only — `probability` rejects the signed scores a `--rank-objective` model emits |
| **Conviction floor** | `min_prob` | Same scores, compare `--min-prob` off vs a floor; fewer, higher-scoring trades at the cost of idle cash |
| **Concentration** | `top_n` | Same scores, compare `--top-n 5` vs `--top-n 20` (two CLI runs or two configs in a notebook) |
| **Holding horizon** | `holding_days` vs label horizon | Align `holding_days` to your forward window or try shorter/longer holds on the same scores |
| **Rebalance rhythm** | `--rebalance-day Friday` vs `last` vs Monday | Same scores, different execution calendar |
| **Risk-off filter** | VIX / regime | Build scores with `vix_percentile` in the panel; compare `--vix-filter` off vs on, or `vix_scale_exposure` (see `--grid vix`) |
| **Holding rule** | Fixed expiry vs rank decay | Same scores, `--mode cohort` vs `--mode rank-hold` with an `--exit-rank` ladder (see `--grid hold`) |
| **Overlapping slots** | `--max-cohorts` | More slots ≈ more concurrent exposure (capital split across cohorts) |
| **Simple baseline** | Cross-sectional rank / random | Export a panel with `prob` = past 20d momentum rank or shuffle (sanity check vs your model) |

These are **not** recommendations—only sensible axes to explore when you stress-test an ML ranker.

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | 2018-01-01 | Price download start |
| `--end` | today | Price download end |
| `--train-end` | 2022-12-31 | Last date in training set |
| `--test-start` | 2023-01-01 | First date in test / walk-forward region |
| `--sample-n` | 500 | Cap the universe at N tickers, drawn as a **seeded random sample** (not an alphabetical prefix). Use a large value (e.g. `10000`) for the full universe |
| `--min-coverage` | 0.98 | Fail the run if the price download returns less than this fraction of the **current** index members (`0` = warn only). Departed members are reported as a survivorship gap, never gated |
| `--batch-size` | 100 | Symbols per yfinance request. Lower it if Yahoo throttles a large universe; no effect with `--provider tiingo` |
| `--horizon` | 10 | Forward return horizon (sessions); also the **purge window** at every split boundary |
| `--threshold` | 0.05 | Binary label threshold (e.g. 5%) |
| `--objective` | rank | `rank`: LGBMRanker (lambdarank, grouped by date) on per-date forward-return quintile grades. `binary`: LGBMClassifier on `fwd_ret >= --threshold`. Applies to Optuna (NDCG@15 vs PR-AUC), the walk-forward, and the saved model |
| `--label-target` | raw | With `--objective rank`, what the ranker ranks: `raw` forward return, `vol_adj` (per unit trailing volatility), or `excess_vol_adj` (peer-relative per unit risk). A plain `excess` option was removed — grades are ranked within a date, so subtracting that date's median cannot change the ordering |
| `--rank-objective` | — | Deprecated; `rank` is now the default. Accepted for compatibility, conflicts with `--objective binary` |
| `--no-optuna` | off | Skip Optuna search; use defaults + any prior best |
| `--optuna-trials` | 40 | Optuna trials |
| `--ts-cv-splits` | 5 | Purged, **date-grouped** expanding CV folds for tuning (no trading day straddles a fold; last `horizon` dates before each validation block are excluded) |
| `--seed` | 42 | RNG seed |
| `--strict-dropna` | off | Drop rows with ANY NaN feature (legacy); default keeps them — LightGBM handles missing values natively |
| `--skip-earnings` | off | Omit Yahoo earnings feature (faster) |
| `--earnings-workers` | 8 | Parallelism for earnings fetch |
| `--skip-walk-forward` | off | Skip monthly walk-forward |
| `--wf-min-train-rows` | 5000 | Minimum training rows per WF month |
| `--wf-top-k` | 10 | Top-K for per-date precision / WF ranking |
| `--wf-scores-path` | none | Write scored WF panel Parquet (for `backtest-sp500`) |
| `--run-backtest` | off | Run portfolio backtest after WF (needs WF scores) |
| `--output-model` | none | Save `.pkl` + `.meta.json` (+ `.optuna.json` if tuned) |
| `--plots-dir` | none | Save evaluation and WF PNGs |
| `--snapshot-dir` | `artifacts/runs/<id>/` | Parquet snapshots + `manifest.json` |
| `--no-snapshot` | off | No Parquet dumps and **no** `manifest.json` |
| `--provider` | yfinance | `yfinance` or `tiingo` (Tiingo equities + FRED macro; needs `[tiingo]` extra + API keys) |
| `--no-macro-merge` | off | Disable **Yahoo ↔ FRED** macro cross-fill (default: merge to fill gaps) |

**Macro cross-fill:** With the default provider (`yfinance`), if `FRED_API_KEY` is set (and `fredapi` is installed via `uv sync --extra tiingo`), training merges Yahoo `^VIX` / `^TNX` / `^IRX` with FRED `VIXCLS` / `DGS10` / `DTB3` so missing dates or columns are filled where possible. With `--provider tiingo`, the primary macro is FRED and Yahoo is used as fallback. Levels can differ slightly between vendors; derived features (`vix_ret_5d`, spreads, percentiles) are computed **after** the merge.

### End-to-end walkthrough

This is the usual **research pipeline**: setup -> **full** train (Optuna, earnings, walk-forward, snapshots, macro merge) -> optional **backtest** on saved scores -> **predict** for daily signals. All of this is educational, not production trading advice.

For one-time environment and secret configuration, follow [Production workflow setup](#production-workflow-setup) first.

#### 1. Install and environment

```bash
cd /path/to/stock-predictor
uv sync --extra dev --extra tiingo
```

This step assumes your keys are already configured as described in [Production workflow setup](#production-workflow-setup).

#### 2. Full training (no shortcuts)

Omit these flags so nothing is disabled: **`--no-optuna`**, **`--skip-earnings`**, **`--skip-walk-forward`**, **`--no-snapshot`**, **`--no-macro-merge`**.

Default **`--sample-n` is 500** (caps the universe). The cap draws a **seeded random sample**, so a capped run is still representative of the index; use a **large** value (e.g. `10000`) to include essentially all tickers overlapping your date window.

Equity downloads are **batched** (`--batch-size`, default 100 symbols per request). One request for the whole universe reliably trips Yahoo's rate limiter, which replies with a *partial frame* rather than an error; each batch is retried with backoff, and symbols still missing get a second pass in smaller chunks.

Coverage is then checked in two tiers, because a flat percentage conflates two opposite situations:

| Cohort | Expectation | On a gap |
|--------|-------------|----------|
| **Current** index members | Vendors serve these reliably | **Fails the run** below `--min-coverage` (default 98%) — a broken or throttled download |
| **Departed** members | Yahoo drops most acquired/renamed/delisted symbols | **Warns only** — this is the documented survivorship bias, not a fault |

A real full-universe run looks like this:

```
  Union tickers overlapping window: 691
  equity download: 594/691 tickers (86.0% overall), 100.0% of current index members
  Survivorship gap: 97/188 departed index members are unavailable from this vendor
  and are absent from the panel, which flatters results. Missing: ABC, ABMD, ADS, AET, AGN (+92 more)
```

86% overall would fail a flat 90% gate even though the download was perfect. A Tiingo key recovers most of the departed names.

**Yahoo equities (default), merged macro when FRED is available:**

```bash
uv run train-sp500 \
  --start 2018-01-01 \
  --train-end 2022-12-31 \
  --test-start 2023-01-01 \
  --sample-n 10000 \
  --optuna-trials 40 \
  --ts-cv-splits 5 \
  --earnings-workers 8 \
  --plots-dir artifacts/plots \
  --output-model artifacts/model.pkl \
  --wf-scores-path artifacts/wf_scored.parquet \
  --run-backtest
```

(`--end` is omitted so prices run **through today**.)

**Tiingo equities + FRED macro** (same idea; requires API keys):

```bash
uv run train-sp500 \
  --provider tiingo \
  --start 2018-01-01 \
  --train-end 2022-12-31 \
  --test-start 2023-01-01 \
  --sample-n 10000 \
  --optuna-trials 40 \
  --plots-dir artifacts/plots \
  --output-model artifacts/model.pkl \
  --wf-scores-path artifacts/wf_scored.parquet \
  --run-backtest
```

- **`--wf-scores-path`** writes the walk-forward panel for later **`backtest-sp500`**.
- **`--run-backtest`** runs the portfolio sim in the same process after walk-forward (requires walk-forward; do not pass `--skip-walk-forward`).
- **Snapshots** go to **`artifacts/runs/<run_id>/`** unless you set **`--snapshot-dir`** or **`--no-snapshot`**.

A run like this can take a long time and may hit API rate limits (especially Tiingo on a free tier).

#### 3. Backtest (optional if you skipped `--run-backtest`)

```bash
uv run backtest-sp500 artifacts/wf_scored.parquet --plots-dir artifacts/plots
```

Use the same execution knobs you care about (`--top-n`, `--holding-days`, `--weighting`, slippage, commissions) as in the [Backtest flags](#backtest-flags) table.

#### 4. Predict (daily): init → dry run → confirm

Match training where features must align: **same `--provider`**, omit **`--skip-earnings`** if the model was trained with earnings, omit **`--no-macro-merge`** if you relied on merged macro, and use a **`--sample-n`** at least as large as in training.

**Create portfolio state once:**

```bash
uv run predict-sp500 \
  --model artifacts/model.pkl \
  --state portfolio_state.json \
  --init \
  --initial-capital 100000
```

**Dry run (prints orders; does not update state):**

```bash
uv run predict-sp500 \
  --model artifacts/model.pkl \
  --state portfolio_state.json \
  --sample-n 10000 \
  --provider yfinance \
  --top-n 15 \
  --max-cohorts 2 \
  --holding-days 10 \
  --slippage-bps 5 \
  --weighting equal \
  --max-drawdown 0.15
```

Add **`--commission-per-share`** / **`--commission-per-order`** to mirror **`backtest-sp500`**. Use **`--provider tiingo`** if you trained with Tiingo.

**Persist the new positions:**

```bash
uv run predict-sp500 \
  --model artifacts/model.pkl \
  --state portfolio_state.json \
  --sample-n 10000 \
  --confirm
```

(Reuse the same flags as the dry run.)

| Training choice | Predict |
|-----------------|--------|
| No `--skip-earnings` | Omit `--skip-earnings` |
| `--provider yfinance` / `tiingo` | Same `--provider` |
| Macro merge on (default) | Omit `--no-macro-merge`; keep `FRED_API_KEY` (+ `fredapi`) when you used Yahoo↔FRED merge |
| Large universe (`--sample-n 10000`) | Same or larger `--sample-n` |

See also [Execution assumptions (backtest vs predict)](#execution-assumptions-backtest-vs-predict) and [Kill-switch](#kill-switch).

### Automation

Use this section for runtime orchestration patterns. For one-time environment and GitHub workflow setup, see [Production workflow setup](#production-workflow-setup).

#### 1. Repo script (local or cron)

[`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) wraps the same commands as the walkthrough. Run from the **repo root** (or any cwd—the script `cd`s to the repo).

```bash
chmod +x scripts/run_pipeline.sh

# Full train + WF parquet + in-process backtest (long run)
./scripts/run_pipeline.sh train-full

# Backtest only (expects artifacts/wf_scored.parquet)
./scripts/run_pipeline.sh backtest

# Daily signal, dry run (no state update)
./scripts/run_pipeline.sh predict

# Daily signal and persist portfolio_state.json
./scripts/run_pipeline.sh predict --confirm
```

Override paths and knobs with **environment variables**, for example:

```bash
MODEL=artifacts/model.pkl STATE=~/state/portfolio.json SAMPLE_N=10000 \
  ./scripts/run_pipeline.sh predict

TRAIN_PROVIDER=tiingo PREDICT_PROVIDER=tiingo ./scripts/run_pipeline.sh train-full
```

Ensure **`.env`** is present (or export keys) so Tiingo/FRED and macro merge work when you use them. For **non-interactive** runs, rely on `uv` and a fixed working directory.

#### 2. Cron (macOS / Linux)

Example: **weekday** dry run after US equity close (times are machine-local; adjust).

```cron
# m h dom mon dow command
15 16 * * 1-5 cd /path/to/stock-predictor && /usr/bin/env MODEL=artifacts/model.pkl ./scripts/run_pipeline.sh predict >> artifacts/logs/predict.log 2>&1
```

Use **`predict --confirm`** only on a machine and account where you accept automated state mutation. Log rotation is your responsibility (`logrotate`, etc.).

#### 3. GitHub Actions (or other CI)

Typical split once setup is done:

| Job | Trigger | Notes |
|-----|---------|--------|
| **Train** | `workflow_dispatch` or weekly cron | Use the built-in `.github/workflows/train-sp500.yml`; keep long timeout and upload `model.pkl` + `wf_scored.parquet` as artifacts. |
| **Predict** | Daily cron | Needs the **model file** on the runner (download from **Release** / **S3** / cache artifact from train workflow). `portfolio_state.json` must be **persisted** between runs (workflow artifact, external DB, or self-hosted runner with a disk)—GitHub-hosted runners start clean each job unless you restore state. |

GitHub-hosted runners are a poor fit for **mutating** `portfolio_state.json` unless you check it in (usually not desired) or use an external store. **Self-hosted** runners or **local cron** are simpler for predict + confirm.

If this folder is not the git repository root on GitHub, copy or symlink `.github/workflows` into the repo root that GitHub builds.

#### 4. What not to automate blindly

- **Full train** is expensive and rate-limit sensitive; schedule it rarely and monitor failures.
- **`predict --confirm`** changes real (or paper) state; pair with alerts and kill-switch awareness.
- **Never commit** `.env` or `.pkl` models if they are private or huge; use secrets + artifacts.

### Predict CLI (daily inference)

Generate tomorrow's picks from a trained model, manage portfolio state, and enforce risk limits.

```bash
uv run predict-sp500 --help

# Initialize a portfolio
uv run predict-sp500 --model models/latest.pkl --init --initial-capital 100000

# Daily signal (dry run — shows orders without executing)
uv run predict-sp500 --model models/latest.pkl --skip-earnings

# Rank-hold mode: sell only when a holding's rank decays (fresh state file!)
uv run predict-sp500 --model models/latest.pkl --hold-mode rank --exit-rank 40

# Execute (updates portfolio_state.json with new positions)
uv run predict-sp500 --model models/latest.pkl --skip-earnings --confirm
```

#### Predict flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to trained `.pkl` model artifact |
| `--state` | `portfolio_state.json` | Portfolio state JSON file |
| `--init` | off | Create a new portfolio state file |
| `--initial-capital` | 100000 | Starting capital (with `--init`) |
| `--top-n` | 15 | Stocks per cohort / portfolio slots |
| `--max-cohorts` | 2 | Fixed mode: max overlapping cohorts |
| `--holding-days` | 10 | Fixed mode: trading days until cohort expiry |
| `--hold-mode` | fixed | `fixed` (expiry after `--holding-days`) or `rank` (sell on rank decay; parity with `backtest-sp500 --mode rank-hold`). Don't switch modes on an existing state file |
| `--exit-rank` | 40 | Rank mode: sell held names ranked worse than this (≥ top-n) |
| `--max-drawdown` | 0.15 | Kill-switch: halt if drawdown exceeds this |
| `--slippage-bps` | 5 | Slippage per side (basis points) |
| `--skip-earnings` | off | Skip earnings feature (must match training) |
| `--sample-n` | 500 | Cap the universe at N tickers, drawn as a **seeded random sample**. Use the same value as training |
| `--seed` | from model meta | Seed for the universe sample; defaults to the seed recorded at training time so the live universe matches |
| `--min-coverage` | 0.98 | Fail if the price download returns less than this fraction of **current** index members (`0` = warn only) |
| `--batch-size` | 100 | Symbols per yfinance request; lower it if Yahoo throttles |
| `--weighting` | equal | `equal` or `probability` (same as backtest; `probability` requires non-negative scores) |
| `--commission-per-share` | 0 | Per-share fee per buy/sell leg (match `backtest-sp500`) |
| `--commission-per-order` | 0 | Flat fee per ticker per buy/sell order (match `backtest-sp500`) |
| `--no-macro-merge` | off | Same as training: skip Yahoo↔FRED macro merge |
| `--one-lot-per-ticker` | off | Fixed mode: cap each ticker at one lot. **Default is cohort parity** — a persistently top-ranked name may sit in overlapping cohorts at double weight, exactly as `backtest-sp500 --mode cohort` models it |
| `--confirm` | off | Update portfolio state (without: dry run) |

#### Kill-switch

The predict CLI tracks a high watermark in the portfolio state. If the current NAV drops more than `--max-drawdown` (default 15%) below the watermark, all new entries are blocked: order generation runs with **no buy leg**, so **`--confirm` never persists new positions** while halted—only expiries and cash from sells update state. Reset by manually editing `portfolio_state.json` or re-initializing with `--init`.

### Python API

After install, import from the package (notebooks can also use root shims `sp500_pit` / `calendar_features`):

```python
from stock_predictor.pit import load_sp500_stints, tickers_overlapping_window
from stock_predictor.calendar_features import add_calendar_features, FOMC_STATEMENT_DATES
from stock_predictor.backtest import BacktestConfig, run_backtest
from stock_predictor.backtest_reporting import print_report, plot_backtest
from stock_predictor.execution_calendar import trading_dates_from_index
from stock_predictor.portfolio import PortfolioState, generate_orders, check_kill_switch
from stock_predictor.predict import load_model, score_universe, build_inference_panel

# generate_orders(..., trading_dates=trading_dates_from_index(adj_close.index), ...)
```

Training lives in `stock_predictor.training` and `stock_predictor.cli`; reproducibility helpers in `stock_predictor.repro`.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/exploration.ipynb` | **ARCHIVED** — invalid methodology, outputs cleared |
| `notebooks/sp500_predictor.ipynb` | **ARCHIVED** — invalid methodology, outputs cleared |

Ensure the repo root or `src` is on `PYTHONPATH`, or run notebooks from an environment where `uv sync` has been applied. Root shims re-export `stock_predictor.pit` and `stock_predictor.calendar_features` for older import paths.

### Tests

```bash
uv run pytest

# include the Tiingo/FRED provider tests (otherwise 6 skip on a missing fredapi)
uv sync --extra dev --extra tiingo && uv run pytest
```

**CI** — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to `main` and every pull request: pytest on Python 3.12 and 3.13, `ruff check`, and an offline smoke job that exercises every console script and both backtest engines against a synthetic panel. It is deliberately network-free, so a Yahoo outage never reds the build. The long training and sweep workflows stay on `workflow_dispatch` / schedule.

Covers PIT membership, calendar features, reproducibility, training utilities (purged splits, rank labels, both Optuna objectives), both backtest engines (NAV compounding, rank-decay exits, VIX scaling), relative-return metrics, portfolio management (fixed and rank-hold order generation), execution parity, macro merge, data providers, and the inference pipeline (`tests/`, 200+ tests).

## Reproducibility (snapshots + manifest)

When snapshots are enabled (default), each run writes under `--snapshot-dir` or `artifacts/runs/<utc_run_id>/`:

| File | Contents |
|------|----------|
| `manifest.json` | `run_id`, UTC time, `argv`, git commit / dirty (if available), env hints, **SHA-256** + row/column counts per snapshot |
| `stints.parquet` | PIT membership table |
| `equity_prices_long.parquet` | Long-format adjusted close + volume |
| `labeled.parquet` | Panel after forward return, **before** the PIT filter (the filter now runs mid-way through feature engineering) |
| `features_clean.parquet` | Feature matrix after NaN-tolerant row selection (label + minimal price history; `--strict-dropna` restores full-row `dropna`) |

With `--output-model`, the manifest also records the pickle path and checksum; the manifest is rewritten at the end of the run.

With **`--no-snapshot`**, no Parquet files and no `manifest.json` are written (the `run_id` may still appear in saved model metadata).

`artifacts/` is gitignored by default—copy snapshots elsewhere if you need to keep them in version control or upload them to object storage.

## Pipeline

1. **Universe**: Point-in-time S&P 500 membership via [fja05680/sp500](https://github.com/fja05680/sp500); capped by `--sample-n` as a seeded random draw and coverage-checked after download
2. **Prices**: Daily adjusted close + volume from Yahoo Finance (`yfinance`) or Tiingo
3. **Labels**: Per-date forward-return quintile grades (default, market-neutral by construction), or `--objective binary` for `fwd_ret ≥ --threshold`.

   **Why rank is the default.** The binary `+5% in 10 sessions` label is satisfied *mechanically* by volatility: a name needs a wide return distribution to clear the threshold, regardless of its expected return. A model trained on it ranks risk, not return — measured on this pipeline, cross-sectional IC of score vs `vol_21d` was **+0.75** while score vs `fwd_ret` was **+0.008**, and mean annualized volatility ran 17% in the bottom score decile to 49% in the top. The rank objective halves that coupling (vol IC +0.43) and flips the top-5 selection bucket from **−0.81%** to **+0.91%** against the universe. See `scripts/signal_depth.py`.
4. **Features**: Price/volume (15), sector-relative (4), macro (5), regime (2), cross-sectional ranks (3), calendar (6), optional earnings (1)

   **Stage order is load-bearing** (`build_feature_panel`):

   1. *Time-series* features (price, volume) on each ticker's **full contiguous history**
   2. *Point-in-time membership filter*
   3. *Cross-sectional* features (regime medians, ranks, sector spreads) on the surviving in-index cross-section
   4. Date-level joins (macro, calendar) and the optional earnings feature

   Filtering before step 1 lets rolling windows span index-membership gaps — a symbol that left and rejoined the index gets a multi-year move reported as its next `ret_1d`. Running step 3 before the filter pollutes ranks and "market" medians with names that were not in the index that day.
5. **Tuning**: Optuna over purged, date-grouped expanding CV folds — PR-AUC for the classifier, NDCG@15 for the ranker (optional)
6. **Training**: LightGBM (classifier or lambdarank ranker) with a purged inner validation split for effective tree count
7. **Evaluation**: per-date Precision@K, rank IC and top-K excess return, aggregated across signal dates; pooled PR-AUC / ROC-AUC reported alongside as secondary (same binary target for both objectives, so they compare directly)
8. **Walk-forward**: Monthly expanding window with a `horizon`-day purge before each test month (optional)
9. **Backtest**: Cash-ledger simulation vs configurable benchmark — cohort or rank-hold engine, with relative-return framing (IR, beta, alpha + t-stat) and sweep grids
10. **Inference**: Daily scoring of the live universe (either model family), portfolio state management, fixed-expiry or rank-hold order generation with kill-switch risk control

## Project layout

```
src/stock_predictor/
  __init__.py            Version + re-exports
  pit.py                 PIT S&P 500 membership
  calendar_features.py   Calendar / FOMC features
  training.py            Features, labels (binary + rank grades), purged CV,
                         Optuna (PR-AUC / NDCG), walk-forward, train/eval, model IO
  cli.py                 train-sp500 entry point
  execution.py           ONE selection/sizing core: backtest, paper and live
                         all call it, so a rule tuned in simulation reaches the
                         account. Only `whole_shares` differs between them
  backtest.py            Backtest engines (cohort + rank-hold) + CLI
  backtest_reporting.py  Reports, relative-return metrics (IR/beta/alpha/t), plots
  portfolio.py           Portfolio state, kill-switch, and order generation
                         (selection delegated to execution.py)
  predict.py             Daily inference CLI (predict-sp500)
  execution_calendar.py  Trading-day calendars. Past sessions from the data,
                         future sessions from the real NYSE calendar
  macro_merge.py         Yahoo ↔ FRED macro cross-fill
  data_provider.py       Provider protocol + factory
  providers/             yfinance + Tiingo/FRED implementations
  universe.py            Seeded ticker sampling + download coverage guard
  signal_depth.py        Selection-depth diagnostics (does the top of the ranking work?)
  long_short.py          Dollar-neutral long-short engine (borrow + turnover costs)
  borrow.py              Per-name short borrow: real rates, stylised proxy, or flat
  fundamentals.py        Point-in-time SEC EDGAR fundamentals (joined on filing date)
                         Ticker->CIK is layered: SEC's published map, then EDGAR's
                         company browser for what it misses (~10% of an S&P panel,
                         AEP/EA/DFS included), cached with negative results in
                         <edgar-cache>/cik_fallback.json. Gross profit and total
                         liabilities are derived by accounting identity when the
                         filer tags the components but not the total.
  stats.py               HAC / Newey-West helpers
  repro.py               Run manifests, hashing, Parquet snapshots
scripts/
  run_pipeline.sh        train-full / backtest / predict orchestration
  backtest_sweep.py      Variant grids (default / vix / hold) + relative tables
  grid_search_sharpe.py  Config grid ranked by Sharpe, with alpha t-stats
  insider_signal.py      SEC Form 3/4/5 insider buying vs a scored panel
                         (tested and rejected -- see Limitations)
  signal_depth.py        Selection-depth / rank-IC / alpha-ladder diagnostics
train_sp500.py           Thin launcher → cli
backtest.py              Thin launcher → backtest
sp500_pit.py             Notebook shim → pit
calendar_features.py     Notebook shim → calendar_features
tests/                   pytest suite (200+ tests)
notebooks/               Exploration + full pipeline
```

## Data sources

| Source | What | Notes |
|--------|------|-------|
| [fja05680/sp500](https://github.com/fja05680/sp500) | PIT membership stints | Community-maintained |
| Yahoo Finance (`yfinance`) | OHLCV, macro, earnings | Rate-limited; data quality varies |
| Wikipedia | GICS sector mapping | Snapshot + hard-coded 2018 override—not a full historical sector time series |

## Limitations

- **Stale artifacts.** Any `wf_scored.parquet` or report produced before the universe fix was built on an **alphabetically truncated** universe (`--sample-n` sliced a sorted ticker list, so a 500-cap run covered roughly A–POOL) with features computed *after* the PIT filter. Those panels are not comparable to current ones — on the same rules, the truncated panel showed +76.6% and Sharpe 1.05 where the corrected panel shows +22.2% and Sharpe 0.16. Regenerate before drawing any conclusion.

  **The notebooks are archived, not current.** Both implement the methodology
  this section calls invalid — alphabetical truncation (`tickers[:SAMPLE_N]`),
  the point-in-time filter applied *before* time-series features, unlabelled
  rows dropped, and a row-based unpurged `TimeSeriesSplit`. They were committed
  with executed outputs, so numbers produced by all four defects were on display
  as findings. Outputs are now cleared and the banner names each defect; the
  code is kept as a record of how the project started. `tests/test_notebooks_archived.py`
  fails if outputs are ever committed again.

  Panels produced before the **row-role fix** are stale in a second, quieter way: they are missing the final `--horizon` sessions entirely. Check with
  `panel.groupby("date").size().tail(63)` — if the tail collapses to a handful of
  names, the panel predates the fix and its most recent quarter is fiction.
- **Not investment advice.** Past backtests and metrics do not guarantee future results.

- **You cannot rank what you cannot price, or what you did not pass.** Two
  faults in the live path, both in how the cross-section reached selection.

  `score_universe` dropped a row only when **all** its ticker-level features
  were NaN. Calendar features such as `days_to_fomc` are date-level constants
  but are not in `MACRO_FEATURE_COLS`, so they counted as ticker-level and kept
  an **unpriced row alive**. It was then ranked, entered `latest_prices` as
  `NaN`, and inflated the cross-section width `min_cross_section` measures. A
  finite positive price is now required before ranking; zero and negative do not
  count either.

  Fixed-hold passed only `scored.head(top_n * 2)` to the shared selection —
  a leftover from before the execution core did the filtering. Every rule that
  reads beyond the head broke:

  ```
  top_n=5, rank_offset=10
    full cross-section      -> T10, T11, T12, T13, T14
    head(top_n * 2) = 10 rows -> nothing
  ```

  Ten rows cannot survive an offset of ten, and ten sits below the cross-section
  floor of `rank_offset + top_n`. A score floor applied to a pre-truncated list
  is likewise a different basket. Rank-hold already passed the full ranking;
  fixed-hold now does too.

- **Which historical companies train must not depend on who joined later.** The
  universe was formed over the whole *download* window and then sampled, so with
  a capped `--sample-n` names admitted to the index **after** the training
  period competed for slots with the historical ones — future membership decided
  which past companies the model ever saw. That is transductive leakage.

  ```
  download-window universe (start -> today)      : 845
  training-window universe (start -> 2024-12-31) : 813
  joined only after train_end                    :  32

  sample_n=500   historical names drawn: download-pool 478 | training-pool 500
                                         differ by 352
  sample_n=10000 differ by 0
  ```

  **At `--sample-n 500`, 352 of the historical names differed.** Uncapped it is
  inert, which is why it went unnoticed — the deployed configuration draws
  everything.

  Three universes now, three purposes:

  | universe | what it is |
  |---|---|
  | **download** | everything fetched — deliberately wider, since recent cross-sections and execution prices both need names outside the training window |
  | **fitted** | drawn from the population that existed *during* training, so the draw depends only on information available then |
  | **scoring** | fitted names the index still holds — what a live run may trade |

  The cap applies to the **fitted** draw; later entrants are added to the
  download afterwards rather than competing for sample slots. Verified on the
  real stints: the fitted draw is now identical whether or not 19 further months
  of index membership are visible.

  Model metadata records the **fitted** universe, so a live run cannot trade a
  name the final model never trained on. A `refit` moves `train_end` forward and
  brings recent entrants into the fit, which is what the monthly schedule is
  for.

- **A missing bar is not a delisting, and a 2-session return is not a label.**
  Terminal-label construction treated *any* ticker whose last quote preceded the
  panel end as delisted, and replaced every missing forward return before it
  with the return to that final quote. Measured on the real panel:

  ```
  rows given a terminal label: 9,453 (0.340%)
  effective horizon: min 1 | median 33 | max 2059   (all used as 63-session targets)
    146 labels measured a single session
  tickers labelled delisted: 155
    ...still current index members: 3   AVB, EA, EQR
  ```

  **AVB, EA and EQR are current S&P 500 members** — the same three the recent-
  coverage guard flags for vendor gaps — being labelled as delisted. And the
  2059-session maximum is not a terminal case at all: it is an *interior* gap,
  where a row years earlier is handed a return to the final quote.
  `specs.md:587` is explicit that missing terminal vendor data is not
  automatically a delisting.

  Terminal labels now require **evidence** — the same `(ticker, date, proceeds)`
  channel the delisting policy uses — and the filled window is bounded to the
  last `horizon` sessions before the event, so a label measures what it claims
  to. Without evidence the tail stays **censored**.

  `terminal_fill="assume_delisted"` restores the old heuristic for reproducing
  older panels. It is unsound and named so.

  **This reintroduces survivorship bias in the labels** until corporate-action
  evidence is supplied: departed members' terminal declines and buyout premiums
  are no longer in the training targets. On the real panel it removes 9,140
  labels (98.51% → 98.18% of rows). That bias is now *stated* rather than
  papered over with a guess about which of 155 tickers actually died.

- **Evaluation and production refit are separate stages.** Three faults, all
  from the same change:

  - Training writes `model_candidate.pkl`; the scheduled workflow uploaded
    `artifacts/model.pkl` with `if-no-files-found: error`. **On a clean runner
    the monthly job failed** — the artifact it wanted was never produced.
  - `TRAIN_END` defaulted to a **fixed** `2024-12-31`, so every scheduled retrain
    refit the same window. The cron ran monthly and learned nothing.
  - Purging removes another `horizon` sessions before `test_start`, so the
    model's last real training signal was **2024-10-01** while its metadata
    reported 2024-12-31 — and the freshness gate read the metadata, understating
    staleness by a whole quarter.

  ```bash
  ./scripts/run_pipeline.sh evaluate   # holds a window back; the walk-forward measures it
  ./scripts/run_pipeline.sh refit      # trains through the newest labellable date
  ./scripts/run_pipeline.sh deploy     # validated promotion
  ```

  They want different windows and cannot be the same run: one holds data back to
  measure, the other uses everything it can label. `--train-through-latest`
  ignores `--train-end` and fits through `HORIZON` sessions behind the last
  session, so a refit's window **moves**. On the current panel that is
  2026-05-18 against a fixed 2024-12-31 — **407 extra training sessions.**
  The walk-forward in a refit run measures nothing, by construction; use
  `evaluate` for that.

  Metadata now records **`fitted_through`** as well as `train_end`, and the
  freshness gate prefers it. Scheduled runs set `REFIT=1`, and the workflow
  uploads the whole bundle — candidate model, its metadata, the scores, and the
  execution panel — so an artifact can be verified rather than just downloaded.

- **Scores and execution prices come from the same run, and it is checked.**
  `run_pipeline.sh` defaulted `EXECUTION_PRICES` to a parquet path and used it
  *if the file existed* — but **nothing in the package, the scripts, or CI ever
  wrote that file.** Where it existed it was a leftover from an ad-hoc fetch,
  and the pipeline paired it with fresh scores without comment:

  ```
  scores            409 sessions, 2025-01-02 -> 2026-08-20
  execution prices 4181 sessions, 2010-01-04 -> 2026-08-18
  ```

  Two scored sessions had no execution row. Absence degraded just as quietly:
  the backtest fell back to forward-filled prices, which on the rank-hold engine
  is the difference between **+17.28% and +22.95%** — a gap that reads as a
  strategy result and is really a missing file.

  - `train-sp500 --execution-prices-path` writes the unfiltered `adj_close`
    panel from **this run's own download**, and records it in the run manifest
    with a hash alongside the other snapshots.
  - `run_pipeline.sh train-full` writes it; `backtest` consumes the same path
    and **fails loudly** if it is absent instead of quietly proceeding.
  - `backtest-sp500` validates date coverage, ticker coverage and schema before
    running, and refuses on mismatch unless `--allow-price-mismatch`.
  - `train-sp500 --run-backtest` uses the panel it just downloaded rather than
    none at all.

  The execution panel is expected to be *wider and longer* than the scored
  panel — it is the full download, the scored panel is point-in-time filtered —
  so only the reverse, a scored row with no execution row, is a fault.

- **A position that cannot be sold has to go somewhere.** Rejecting unpriceable
  fills was correct, but it left capital locked in holdings that could never
  exit — on the current panel, **1068 deferred exits**. `specs.md` sets the
  rules: delisting treatment must use *explicit evidence* or an *explicit
  conservative fallback* (`:181`), a data gap is never itself proof (`:587`),
  and the policy must be documented, configurable, and in the diagnostics
  (`:249`).

  Three outcomes, in order of preference:

  1. **Evidence** — `--delisting-proceeds` takes `(ticker, date, proceeds)`: a
     cash acquisition price, or zero for a bankruptcy. Applied point-in-time, so
     a deal settling next month cannot pay today.
  2. **A named fallback after a grace period** — default `write_off` at zero
     once a holding has been unpriceable for `--delisting-grace-sessions` (63,
     about a quarter). Shorter gaps are halts or vendor outages, not delistings.
     Zero is the conservative choice: a stock you cannot sell is not worth its
     last quote.
  3. **`--delisting-fallback hold`** — keep it indefinitely and leave the
     capital visibly stuck.

  Inventing a price is never an option. Forward-filled quotes may *value* a
  position; they may not dispose of one.

  | policy | deferred | written off | CAGR | Sharpe | maxDD |
  |---|---|---|---|---|---|
  | hold forever | 1068 | 0 | +16.19% | 0.49 | −50.7% |
  | write_off after 63 sessions | 0 | 6 | +17.28% | 0.50 | −52.3% |
  | **write_off + execution prices** | **0** | **0** | **+22.95%** | **0.62** | −52.3% |

  The third row is the one that matters. **With the full unfiltered download
  nothing gets stuck or written off at all**, because almost nothing in this
  universe is genuinely unpriceable — it was only unpriceable in the *PIT
  filtered* panel. Rank-hold was being clogged: rejected exits kept dead names
  in the book, consuming slots that should have rotated into fresh picks. Run
  the backtest with `--execution-prices` and the delisting policy barely fires.

  Reported per run: `exits_deferred`, `disposals_by_evidence`,
  `disposals_written_off`, `disposal_proceeds`.

- **A fill needs a quote on the session it executes.** The price panel is
  forward-filled so open positions can be *marked* between quotes. Execution
  read straight from it, so a leg with no quote on its entry or exit session
  filled at an earlier price. Counting those afterwards — which an earlier
  change did — makes the problem visible without preventing it, and `specs.md`
  is explicit: the scored panel must not be forward-filled to create execution
  prices, and every requested fill must record Filled or Rejected.

  Three gaps the counting missed entirely: rank-hold **buys** were never
  counted; the rank-hold denominator was `2 × len(closed)`, ignoring every entry
  for a position still open; and the **long-short engine ignored execution
  prices and the diagnostics together**.

  | engine / mode | requested | rejected | rate | CAGR | Sharpe |
  |---|---|---|---|---|---|
  | cohort, fill-stale (old) | 1680 | 0 | 0.00% | +17.27% | 0.52 |
  | cohort, reject | 1680 | 20 | 1.19% | +17.73% | 0.53 |
  | cohort, reject + exec prices | 1680 | 4 | 0.24% | +17.46% | 0.52 |
  | **L-S, fill-stale (old)** | 4918 | 0 | 0.00% | +16.64% | 0.91 |
  | **L-S, reject** | 5401 | **538** | **9.96%** | +16.54% | 0.91 |
  | L-S, reject + exec prices | 5071 | 170 | 3.35% | +16.66% | 0.91 |

  **Nearly 10% of long-short fills were against carried-forward prices** — the
  engine behind every headline number here, and the only one with no
  diagnostics. The P&L effect is small because errors wash out across ~500
  names, but a tenth of the trades were not real.

  Fills are now rejected rather than faked, in all three engines.
  `--execution-prices` supplies the full unfiltered download and removes most
  rejections; `--allow-stale-fills` restores the old behaviour explicitly.
  Every run reports `fills_requested`, `fills_filled`, `fills_rejected` and
  `fill_reject_rate`, and rank-hold reports `exits_deferred`.

  A rank-hold exit that cannot be priced is **deferred and the position
  retained**, rather than exiting flat at the entry price. That ties up capital
  in a position that cannot be sold, which is honest but not free — **delisting
  proceeds remain unmodelled**, and that gap is now visible in the diagnostics
  instead of hidden behind a forward fill.

- **A missing quote is not a fill.** Both live order generators priced exits
  with `prices.get(ticker, entry_price)`. When a holding had no quote the exit
  executed **at the entry price** — not even at the `last_price` the position
  already carries — the position was removed, and the cash was credited.

  A probe: 10 shares, entry $100, last observed $40, no current quote produced
  `SELL 10 @ $100.00`, crediting **$1,000** for something last seen worth $400.

  It was reachable because the live download covers the model's universe
  intersected with current index membership, so a holding that *left* the index
  is not downloaded and therefore has no quote. That intersection was introduced
  to fix universe reproduction; it narrowed the download and widened this hole.

  Three changes:

  - **Held tickers are downloaded regardless of the entry universe.** A holding
    is a position, not a candidate.
  - **Quotes come from the raw download, not the scored panel.** The panel is
    PIT-filtered, so a departed holding has no row in it and would read as "no
    quote" even when downloaded successfully.
  - **Without a valid quote the exit is deferred and the position retained.**
    Zero and NaN do not count as quotes. `mark_price` still falls back for
    *marking*, which is an estimate; a fill is not.

  A permanently unquotable holding therefore stays in the book and is reported
  every run, rather than being converted into cash that does not exist. That is
  the correct answer — you still own it — and it needs an operator decision, not
  a synthetic sale.

- **Selection and valuation are different questions.** The backtest priced fills
  from the *scored* panel, which is point-in-time filtered, and forward-filled it
  unconditionally. A holding that left the index — or was delisted, halted, or
  had a data gap — stopped having rows, and its last in-index price was carried
  forward indefinitely. Exits executed against that stale quote; rank-hold fell
  back to the entry price outright.

  Measured on `artifacts/final/wf_control.parquet`: **40 of 840 cohort legs had
  no row at their exit date, affecting 11 of 56 cohorts.** DELL exited at a
  forward-filled 237.64 against a true 456.79 — 92% on one leg.

  | panel | execution source | stale legs | rate | CAGR | Sharpe |
  |---|---|---|---|---|---|
  | pre-row-role | scored panel | 41 | 2.44% | +12.21% | 0.39 |
  | pre-row-role | real prices | 1 | 0.06% | **+13.58%** | 0.43 |
  | current | scored panel | 10 | 0.60% | +17.27% | 0.52 |
  | current | real prices | 2 | 0.12% | +17.35% | 0.52 |

  **It cost 1.37pp of CAGR on the old panel and 0.08pp on the current one** —
  the row-role fix had already removed most of it, since restoring the tail gave
  most holdings a real row again. The mechanism remained, and per-leg error is
  unbounded.

  `--execution-prices` takes the full unfiltered download and prices fills from
  it; `run_pipeline.sh backtest` passes `artifacts/hybrid_adj_close.parquet`
  automatically when present. Every run reports `stale_fills` and
  `stale_fill_rate`, and the report warns when any fill was a guess.

  **Two legs remain unpriceable even with full history** — genuinely delisted
  with no recoverable quote. Those need explicit delisting proceeds, which are
  still not modelled.

- **The live universe is the model's own draw, not a reseed.** Training samples
  the union of tickers overlapping the whole training window; inference used to
  resample a 400-day window with the same seed. **The same seed drawing from a
  different population is a different draw.** Measured on the real stints at
  `--sample-n 500`:

  | | training draw | live draw | overlap |
  |---|---|---|---|
  | tickers | 500 | 500 | **307** |
  | current index members | 305 | 473 | 289 |

  Every cross-sectional feature is a rank *within the universe*, so a different
  universe means different feature values for the same stock. `sample_n=10000`
  draws everything and both sides end up with the same 503 current members,
  which is why this stayed latent — the deployed configuration is uncapped.

  Model metadata now records `universe` (the tickers actually drawn) and
  `universe_hash`. A live run intersects that with current index membership
  instead of reseeding. Members added *since* training are excluded rather than
  silently joining the cross-section — the model has never ranked them — and the
  run reports how many, so the monthly retrain has a visible purpose.

  Models trained before this say so and fall back to reseeding, with a warning
  when `--sample-n` is capped. `sample_mismatch_warning` compares sample
  *sizes*; it would have stayed silent at 500 versus 500, which is why the draw
  itself is now recorded.

- **The execution panel was a gap-filler, not the authority.** When both
  sources had a price, `_prepare_scored` kept the *scored* panel's and used the
  execution panel only to patch holes:

  ```
  scored price:             100
  execution price:          200
  fill price actually used: 100
  ```

  `specs.md:233` is explicit that a scored panel's `adj_close` "MAY be included
  for diagnostics, but it is not authoritative execution data" — so the
  authoritative source was reduced to filling gaps in the non-authoritative
  one, which is the wrong way round twice.

  An execution panel now *replaces* the price matrix instead of patching it.
  Read strictly: a scored ticker with no execution row has no price, so its
  fills are rejected rather than invented from the signal table.
  `validate_execution_panel` already reports that coverage gap.

  **On the current bundle this changes nothing** — 1 of 909,171 scored prices
  differs from the execution panel, and coverage is complete, so CAGR stays
  13.15%. The bug was real and latent: it only bites when the two sources are
  adjusted differently, which is exactly the case nobody notices. That
  disagreement is now measured and printed, which is the one job the scored
  panel's `adj_close` is still allowed to do.

  Separately, `backtest_sweep.py`, `grid_search_sharpe.py` and
  `signal_depth.py` never passed execution prices at all, so every number they
  produced came from the point-in-time panel — the configuration measured just
  above at **4.44% CAGR against 13.15%**. All three now take
  `--execution-prices` and warn loudly when it is absent.

- **The cohort engine decided entries using the exit quote.** `_build_cohort`
  priced the entry *and* the exit at construction time — on the signal date —
  and dropped any name whose exit price was missing or stale. The exit is
  `holding_days` sessions in the future, so a position that was perfectly
  enterable vanished because of something that had not happened yet:

  ```
  complete data: cohort entered A on 2024-01-03
  missing exit:  the entire cohort disappeared
  ```

  Look-ahead and survivorship in one line, and it flatters twice over: the
  names it removed are disproportionately the ones that stopped being quoted,
  which is to say the failures. The reject-stale-fills work made it stricter,
  and therefore more biased, by also requiring the exit quote to be *real*.

  Entry now asks one question — is there a real, positive price to buy at
  today? Exits are resolved afterwards through the same machinery rank-hold
  already used: fill if there is a quote, otherwise defer to the next session
  that prints one, and dispose by explicit evidence or the configured fallback
  once the grace period lapses. A leg that settles late becomes its own cohort,
  so the capital stays tied up for as long as it really was.

  Measured on the control panel at top-15 / horizon 63:

  | config | CAGR before | CAGR after | Sharpe | max DD | cohorts |
  |---|---|---|---|---|---|
  | with execution panel | 13.68% | **13.15%** | 0.49 → 0.47 | −38.2% | 56 → 57 |
  | no execution panel | 11.00% | **4.44%** | 0.42 → 0.25 | −37.1% → **−55.2%** | 54 → **72** |

  With a full execution panel the bias is half a point, because the panel
  prices nearly everything. Without one — which is how this repo measured
  itself for most of its history, and what `backtest-sp500` still does if you
  omit `--execution-prices` — **6.6 points of CAGR were look-ahead**, along
  with 18 percentage points of drawdown. The 18 extra cohorts are exactly the
  ones that had been quietly deleted, and they lose money. Treat any
  cohort-engine number in this README's history that predates the execution
  panel as inflated.

- **Promotion was not atomic, and the comment admitted it.** The deployed
  model and its metadata are two files. Promotion staged both and issued two
  consecutive `os.replace` calls — each atomic alone, but a failure on the
  second left the pair mixed, and the handler cleaned up temporaries without
  undoing the first:

  ```
  PromotionError {'model': 'new', 'meta': 'old'}
  ```

  `specs.md:511` is unambiguous: *"Promotion MUST be atomic. A crash MUST NOT
  leave the model and metadata from different versions."* The code shipped with
  a comment calling itself *"as close to atomic as two paths get without a
  symlinked release dir"* — a known gap written down instead of closed.
  Narrowing a window is not closing it.

  A release is now an immutable directory holding both files, and the deployed
  paths resolve through a single pointer:

  ```
  model.pkl        -> .current_release/model.pkl
  model.meta.json  -> .current_release/model.meta.json
  .current_release -> releases/20260825T204001Z-<run_id>/
  ```

  Promotion renames one symlink, so both paths move together. The regression
  test injects a failure at *every* filesystem call in turn — six per
  promotion — and asserts the on-disk pair is internally consistent each time.

  Superseded releases stay on disk, so rollback is a pointer swap:
  `deploy_model.py --list` and `--rollback-to <release>`, validated the same
  way a promotion is. **Not `mv`** — a symlink pointing at a directory swallows
  the replacement *into* the directory instead of replacing the link, leaving
  the old version live and writing into a release that is meant to be
  immutable. That trap has its own test, because the obvious shell one-liner
  hits it silently.

- **The scheduled refit would have failed on its first firing.**
  `--train-through-latest` exists so the monthly cron stops refitting a
  hard-coded window. It set `train_end` to the newest labelable session — then
  carried on doing everything an *evaluation* run does, including purging a
  full horizon away from a test period that by construction held nothing:

  ```
  panel end:             2025-11-28
  newest labelable:      2025-09-02
  actual fitted through: 2025-06-05   <- 63 sessions discarded
  test rows:             0
  ```

  The empty frame then reached `evaluate_test_set`, which cannot score zero
  rows. `train-sp500.yml` sets `REFIT=1` on schedule, so every monthly run
  would fail *after* paying for the fit. It had never fired — the cron was
  added on 2026-08-21 and the first run was due 2026-09-01 — so this was caught
  one week before it would have run.

  Purging protects a test period from training rows that saw its prices. With
  no test period there is nothing to protect, and the discarded horizon is the
  *most recent* data the model has, which is the entire point of refitting.
  `split_train_test` now branches: evaluation holds out and purges, a refit
  trains on every labelable row and returns `None` for the test set rather than
  an empty frame, so a caller that forgets to branch fails loudly. Evaluation
  and walk-forward are skipped in refit mode, and a refit's metadata records
  the window it actually used — it was reading `args.train_end`, so a refit
  would have reported the default months after it stopped using it.

  Use `evaluate` (`REFIT=0`) to measure and `refit` (`REFIT=1`) to build.

- **One price dictionary answered three different questions.** The live path
  built `latest_prices` from `adj_close.ffill().iloc[-1]` and handed the same
  dict to the kill switch, the staleness warning, and order generation. Forward
  fill is *correct* for the first (`specs.md:248` — a holding that stops being
  quoted must still be markable, or it falls back to its entry price and can
  never show a loss the kill switch could see) and forbidden for the third
  (`specs.md:405`).

  So the defect hid inside a line that looked right. A holding last printed at
  $41 three sessions ago reached order generation as a clean $41:

  ```
  latest raw quote      : nan
  quote passed to orders: 41.0
  valid_quote() sees    : 41.0
  stale_positions()     : ()        # dict is populated, so nothing to report
  ```

  `valid_quote()` was added earlier precisely to refuse missing quotes, and it
  never saw one — the forward fill had already replaced it upstream. The
  earlier fix hardened the *consumer* and left the *producer* filling. Both
  halves were needed.

  [`quotes.py`](src/stock_predictor/quotes.py) now separates the roles at the
  source: `execution_quotes` returns the final session only and omits anything
  that did not print, `valuation_marks` carries prices forward for NAV and the
  kill switch, and `quote_ages` / `last_quote_dates` make the gap a number with
  a date on it rather than something inferred from a dict's shape.

- **Evaluation pooled observations at the wrong unit.** PR-AUC and ROC-AUC
  ranked every ticker-date in a month against every other, and "weekly
  Precision@10" picked ten rows out of a *whole week* — all ten could come from
  one session. Neither is a book anyone could have held. The strategy chooses a
  cross-section **per signal date**, and a LambdaRank score is only calibrated
  within its group, so pooling grades a comparison the model was never asked to
  make.

  Primary evaluation is now per signal date, then aggregated: `precision_at_k`,
  `rank_ic` (Spearman of score against forward return), and `top_k_excess`.
  Pooled AUCs are still reported, renamed `pr_auc_pooled` / `roc_auc_pooled` so
  the unit is visible at the call site.

  On the 2019–2026 control panel (1,854 signal dates, base rate 44.07%):

  | k | per-date P@k | weekly-pooled P@k | rank IC | top-k excess |
  |---|---|---|---|---|
  | 5 | 0.4717 | 0.4518 | +0.0383 | +2.55% |
  | 10 | 0.4661 | 0.4718 | +0.0383 | +2.36% |
  | 20 | 0.4619 | 0.4782 | +0.0383 | +2.14% |

  Note the pooled column is not biased in a fixed direction — it understates at
  k=5 and overstates at k=10 and k=20. It was not a constant offset that could
  be reasoned around; it was the wrong measurement. The honest edge over the
  base rate is 2–3 points, and precision decays with k as a real signal should.

- **Dollar-neutral is not market-neutral.** Equalising notional equalises
  dollars, not exposure. This model ranks volatility *positively*, so the long
  leg holds beta-1.27 names and the short leg beta-0.66 ones, and the book keeps
  a market position that a notional description hides.

  Measured on the 2019–2026 panel at horizon 63 (single schedule offset — read
  the rows against each other, not against the 21-offset medians quoted
  elsewhere):

  | config | beta | (t) | alpha/yr | (t) | *raw-spec* alpha | (t) | CAGR | Sharpe | maxDD |
  |---|---|---|---|---|---|---|---|---|---|
  | unhedged | **+0.251** | +4.10 | +8.18% | +2.12 | +11.55% | +2.90 | 16.6% | 0.97 | −16.4% |
  | `hedge_beta=0.25` | +0.022 | +0.40 | +7.74% | +2.05 | +12.14% | +3.14 | 12.7% | 0.73 | −17.5% |
  | `hedge_beta=0.20` | +0.068 | +1.19 | +7.84% | +2.07 | +12.03% | +3.09 | 13.5% | 0.79 | −17.3% |
  | `hedge_beta=0.40` | −0.114 | −2.19 | +7.44% | +2.00 | +12.45% | +3.29 | 10.2% | 0.52 | −18.9% |

  The `alpha/yr` column is CAPM on **excess** returns, `(r_s − r_f) = α + β(r_b
  − r_f)`. The `raw-spec` column is the same series regressed raw on raw, which
  is what this table used to report — kept only to show the size of the error.

  Three things follow, and they do not all point the same way.

  **About 3.4 points of the old headline were the cash rate, not skill.** A raw
  regression leaves the intercept absorbing `r_f · (1 − β)`. A dollar-neutral
  book is mostly cash, so β is near zero and almost the entire risk-free rate
  landed in "alpha". At `hedge_beta=0.40` the bias is +5.0 points.

  **This inverted the hedging conclusion.** An earlier revision of this section
  read *"the alpha is real and survives hedging — +10.95% becomes +11.67%"*.
  That was an artifact: hedging cuts β, which *raises* `r_f · (1 − β)`, which
  inflates raw alpha. Specified correctly, hedging **lowers** alpha, +8.18% →
  +7.74%. Every raw-spec row is monotone in the hedge for the same reason, and
  none of it was selection skill.

  **The alpha survives the correction, but with less room.** t falls from +2.90
  to +2.12 unhedged and +2.05 hedged: still past the conventional bar, no longer
  comfortably. Single sample, single schedule offset, one asset class — and
  selected from a large search, so see
  [Descriptive, not confirmatory](#descriptive-not-confirmatory). These are not
  confirmatory p-values.

  **Hedging still made this sample worse**: Sharpe 0.97 → 0.73, and max drawdown
  slightly *deeper* at −17.5%. Over 2019–2026 the +0.25 beta was a tailwind, and
  the overlay costs slippage and borrow. Hedging is not a way to improve a
  backtest run over a bull market; it removes a risk you did not choose and are
  not paid for skill on, and the sign of that trade flips in a falling market.

  `hedge_beta` shorts the benchmark as an overlay — added *after* selection, so
  it never consumes a decile slot, and priced through the same engine so it pays
  the same slippage, borrow and financing as any other short. Hedging through
  the index rather than by scaling the short book (which would need **1.93×**
  short notional) leaves the selection untouched and keeps beta-estimation error
  out of position sizing. It is **off by default**: an unhedged book is a
  defensible choice, an unstated one is not.

  Every long-short run now reports `beta`, `beta_t`, `alpha_ann` and `alpha_t`
  whenever a benchmark is configured.

- **Training writes a candidate; promotion is a separate step.** `train-full`
  used to write straight to the model the live path loads, so a retrain replaced
  the model being traded the instant it finished — with no check that the new one
  was loadable, fresh, or the right horizon. (A stray `train-full` in this repo's
  history came within a download phase of doing exactly that.)

  ```bash
  ./scripts/run_pipeline.sh train-full   # -> artifacts/model_candidate.pkl
  ./scripts/run_pipeline.sh deploy       # validates, archives, promotes
  ```

  Promotion refuses a candidate that does not load, lacks the metadata the live
  path requires, fails the freshness policy, or whose **horizon does not match
  the holding rule** — the exact defect this repo shipped, a horizon-10 model
  traded on a 63-day exit. A refused promotion changes nothing, and the outgoing
  model is archived so a bad one is reversible. `--force` overrides and still
  reports and still archives.

  Note the scheduled GitHub Actions retrain uploads an **artifact**; it cannot
  deploy to the machine running `predict`. Promotion there is deliberate.

- **The daily report names its own numbers.** It printed `P(+5%)=29.000` for a
  lambdarank model — an unbounded ranking score labelled as a probability, on the
  line an operator reads every morning. The label is now derived the same way the
  score is: `predict_proba` means a probability, otherwise a score.

- **Stale inputs block the live run.** A live run consumes a fitted model, a
  price panel and a state file, all of which rot at different rates, and nothing
  used to check any of them.

  That was not hypothetical. **The model deployed here until 2026-08-21 had
  `train_end` of 2022-12-31 — 3.64 years before it was used to pick trades — at
  a 10-day horizon while the strategy traded 63.** It was caught by reading the
  metadata by hand. `predict-sp500` printed the feature count and the horizon
  and went ahead.

  `predict-sp500` now refuses to trade when the model is older than
  `--max-model-age-years` (default 2.0) or the panel is more than
  `--max-data-age-sessions` behind (default 3). Data age is counted in
  **exchange sessions**, so a long weekend does not read as a vendor outage.
  An unknown age counts as stale — metadata without a `train_end` is not the
  same as a fresh model. `--allow-stale` downgrades the block to a warning,
  deliberately and visibly.

  The gate runs **before any order is generated**, so a blocked run prints no
  signal to be tempted by and does not touch the state file.

- **A decision names itself the same way twice.** Cohort IDs were
  `uuid.uuid4().hex[:8]`, so re-running an unchanged signal produced different
  identifiers and nothing downstream could tell whether two runs had reached the
  same conclusion. They are now a SHA-256 digest of the signal date, the sorted
  basket and the holding rule — stable across processes, unlike `hash()`, whose
  string seed is randomised per run.

  Idempotency itself was already sound (`last_signal_date` blocks a second
  cohort for the same `as_of`, and `save_state` writes atomically via
  `os.replace`). What was missing was the ability to *verify* it. This is also
  what a broker's client-order-ID has to be: the guarantee that a retry after a
  timeout is recognised as the same order rather than a second one.

- **Downloaded is not the same as rankable.** `check_download_coverage` asks
  whether a ticker came back from the vendor at all. It does not ask whether the
  *recent* sessions its cross-sectional features are built from are present — and
  a momentum feature computed across a three-week hole is not a slightly wrong
  number, it is a different one.

  Measured on this panel, **AVB and EQR — both continuous S&P 500 members — were
  present on 1 and 2 of the 20 most recent sessions while counting as fully
  covered**. Those holes move around: a later snapshot cleared AVB and flagged
  EA instead, which is why this is a guard rather than an exclusion list.

  `predict-sp500` now drops names below `--min-recent-coverage` (default 0.8 of
  the last 20 sessions) before scoring. The check is scoped to names that survive
  the **PIT filter**: applied to the raw download it flags every delisted symbol
  at 0% — 192 of 845 on this universe — burying the two that matter. Against
  current members it flags 2 of 503.

  It is live-only by design. In a backtest the hole is historical and already
  reflected in the prices; live, it means ranking on stale features for a
  decision about to be acted on. Applying it retroactively would change measured
  results, which is a separate decision from protecting the next trade.

- **Insider transactions were tested and rejected.** SEC Form 3/4/5 open-market
  buying (the [insider transactions data sets](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets))
  is a well-documented factor and the data is *cleaner* than the fundamentals
  already in use — `ISSUERTRADINGSYMBOL` gives the ticker directly, and both
  `TRANS_DATE` and `FILING_DATE` are present with a median Form 4 lag of two
  days, so the point-in-time join is exact. It still shows nothing here.

  Measured over 2019-01-02 → 2026-08-19 (909,208 labelled rows, 632 tickers,
  23,595 open-market purchases), joined on `FILING_DATE`, horizon 63:

  | measure | value | HAC t |
  |---|---|---|
  | standalone rank IC, all names | +0.0045 | +0.64 |
  | excess fwd return, any insider buy (~74 names/date) | +0.15% | +0.54 |
  | excess fwd return, cluster buy ≥2 buyers (~25/date) | **−0.70%** | −1.68 |
  | excess fwd return, ≥3 buyers (~12/date) | −0.38% | −0.69 |
  | rank IC of buy *size* among buyers | −0.0240 | −1.49 |

  For comparison the price-only model scores **+0.0374 (t +1.64)** on the same
  panel. Nothing here clears |t| = 2 in either direction, so the honest reading
  is *no detectable effect* — not that insider buying is bearish. The cluster
  variant, which the literature reports as the strongest, is the most negative
  cell in the table.

  Two limits bound that conclusion. The signal is live on only **15.1%** of rows
  (5.2% for clusters), and the documented effect is concentrated in small caps —
  the S&P 500 was always the least favourable universe for it. And these excess
  returns are **raw, not sector- or style-neutral**: insider clusters form after
  drawdowns, so a negative reading across a mega-cap-led tape may be reporting a
  value tilt rather than an insider effect.

  One hard data constraint if you revisit this: the **10b5-1 flag (`AFF10B5ONE`)
  does not exist in the datasets before 2023q1** — 17 of the 32 quarters here
  lack the column entirely — so the discretionary-versus-scheduled split can
  only be built on recent history.

  Re-derive with [`scripts/insider_signal.py`](scripts/insider_signal.py); like
  the rest of this section it is meant to be re-run, not trusted.

- **One execution core.** The backtest, paper trading and live orders share
  [`execution.py`](src/stock_predictor/execution.py): the same selection rules,
  the same weights, the same fill prices, the same fee model. Each path used to
  carry its own copy, kept in step by comments reading "mirrors the backtest",
  and they drifted — `--min-prob`, `--rank-offset` and `--min-cross-section`
  reached the simulation and never the live path, so a configuration could be
  measured and then quietly not traded. `_compute_weights` and
  `_long_only_weights` were the same algorithm in two files with two different
  error messages.

  Exactly one thing legitimately differs, and it is a single argument
  (`whole_shares`): an account buys integer lots, a simulation need not. The
  loop and where state lives differ too, and those belong to the callers.

  `tests/test_execution_parity.py` asserts the two ends agree on a shared
  scored cross-section across six configurations, and that the live CLI exposes
  every selection rule the backtest can express — a rule you can measure but
  not trade is the failure this prevents.

  The refactor is behaviour-preserving for the simulation: on the 639-ticker
  panel, all six configurations reproduce their pre-refactor metrics bit for
  bit, except one `total_costs` differing by 1e-10 on a $30,956 figure from
  summation order.

- **Row roles are separate, and used to be conflated.** `build_labeled_panel`
  answered three different questions with one `dropna`, so a row missing a
  *future* label was deleted as though its *past* features were invalid. The
  rows deleted were the newest ones — precisely what a live model ranks. The
  panel now carries two flags and drops neither role:

  | flag | meaning | consumer |
  |---|---|---|
  | `is_tradable` | a positive price exists, so the row can be ranked and traded | scoring, backtest |
  | `has_label` | a forward return exists, so the row can supervise training | training, IC/depth metrics |

  `target_5pct` is `NaN`, not `0`, for unlabelled rows: an unknown future is not
  a negative outcome. `select_training_rows` requires a label;
  `select_scoring_rows` is the superset that does not. Pass
  `drop_unlabeled=True` for the old behaviour.

  This was a **backtest/live divergence**:
  [`build_inference_panel`](src/stock_predictor/predict.py) is explicitly "no
  forward return, no label", so the live path always scored the newest sessions
  while the backtest deleted them. On the pre-fix panel the final 63 sessions
  carried a median of **2 names against a universe median of 491**, and 61 dates
  held fewer than 15 names.

  **The fix restores data; it does not create alpha.** Measured on one panel,
  full versus its `has_label` subset: return IC is bit-identical (unlabelled
  rows are excluded from IC by construction), CAGR moves +12.2% → +12.4% and
  Sharpe 0.68 → 0.69. Do not compare panels across a re-download to size this
  effect — ticker set and price vintage move too, and that confound is larger
  than the fix.

  Both long-only engines also gained **`min_cross_section`** (default
  `rank_offset + top_n`), which gates *entries* on a date having enough scored
  names to rank at all. Exits are never gated, so a narrowing cross-section
  cannot strand a position.
- **Horizon is the biggest lever found.** The disclosure further down was
  measured at a 10-session horizon, which is the wrong instrument for
  cross-sectional equity signal. Figures below are the current control panel:
  **639 tickers, 1,918 sessions, 2019-01-02 → 2026-08-19, `--horizon 63`**,
  all fixes applied including row roles.

  Return IC is **+0.0374 (HAC t +1.64)** and the long-short decile spread is
  **+17.5%/yr gross**. On strictly non-overlapping samples — the only honest
  test at this horizon, since consecutive daily observations share 62 of 63
  days — that is **t = +2.31 across 30 non-overlapping periods**.

  Those periods are non-overlapping *in time*, which is what makes the standard
  error honest about autocorrelation. It is not what "independent" usually
  implies, and an earlier revision used that word. See
  [Descriptive, not confirmatory](#descriptive-not-confirmatory) below before
  reading any t-statistic here as evidence.

  Traded at 1.0x gross through [`long_short.py`](src/stock_predictor/long_short.py)
  with 5bps slippage, per-name borrow and financing charged, the median across
  21 rebalance-schedule offsets is **CAGR +12.4%, Sharpe 0.69, max drawdown
  −16.1%**, every offset positive. Borrow tolerance is better than previously
  stated: at a flat 10%/yr it still returns +11.5% CAGR (Sharpe 0.56), and
  even at 20%/yr it is +6.0% (Sharpe 0.18) — the earlier "decays to zero by
  ~10%" claim was measured on a different panel and does not hold here. That
  curve is a single schedule, not the offset median, so read it as a shape.

  <a id="descriptive-not-confirmatory"></a>
  ### Every t-statistic here is descriptive, not confirmatory

  HAC standard errors correct for **autocorrelation**. Nothing in this repo
  corrects for **multiplicity**, and the search was not small. Counted from the
  code and this README, all run against substantially the same 2019–2026 panel:

  | searched | how many |
  |---|---|
  | `grid_search_sharpe.py` default grid (top-N × holding × rebalance day × score floor × VIX filter) | 12, and all five axes take comma-separated lists |
  | `backtest_sweep.py` named grids (`default`, `vix`, `hold`) | 8 + 8 + 7 |
  | rebalance-schedule offsets | 21 |
  | `hedge_beta` levels | 4 |
  | depth profile top-N levels | 4 |
  | borrow assumptions (flat, per-name, 10%, 20%) | 4 |
  | horizons (10 vs 63 — "the biggest lever found") | 2 |
  | objectives (`rank`, `binary`) × tuning metrics (NDCG, PR-AUC, top-N excess) | 6 |
  | Optuna trials per run, over the full hyperparameter space | configurable |

  Reporting the best of a search as though it were a single pre-registered test
  is how a backtest flatters itself, and the fact that each individual number
  above was measured honestly does not fix it. A t of +2.31 or +2.12 is roughly
  what the *maximum* of a few dozen correlated noise draws looks like.

  So: **read every t-statistic in this README as descriptive of this panel, not
  as evidence the effect will persist.** Nothing here has been through a locked
  holdout, a nested walk-forward where selection happens inside each fold, or a
  multiple-testing correction. Until one of those exists, the honest summary is
  that the sign is stable across many reasonable choices — which is worth
  something, and is a much weaker claim than significance.

  Three things keep this short of a strategy.

  The **long-only** book — the one the cohort engines simulate — remains the
  weaker expression: the significant measure and the implementable one are
  different measures.

  **Magnitude is schedule-sensitive**: Sharpe ranges 0.44–0.91 across the 21
  offsets and CAGR +8.9%–+16.6%. An early run once quoted the maximum, so
  always sweep offsets before quoting a number.

  **The regime split no longer holds magnitude.** 2019–2022 (incl. COVID) is
  +11.7%/yr at t **+1.01** (n=16) against 2023+ at +24.4%/yr, t **+2.78**
  (n=14). The sign holds in both halves; the size does not, and the early half
  is not significant on its own. An earlier version of this README claimed the
  result held "sign and magnitude across two regimes" — on the current panel
  only the sign does.

  Borrow is the caveat that **turned out to be backwards**: the assumption was
  that the short book skewed to high-volatility, hard-to-borrow names, making a
  flat rate optimistic. Measured with `--per-name-borrow`, the short book is
  *cheaper* than the universe (0.83% vs 1.22%, a **0.68x** concentration)
  because the model ranks volatility **positively** — the volatile,
  expensive-to-borrow names sit in the *long* book, where no borrow is paid.
  Per-name borrow moves median Sharpe from 0.71 to 0.69. The assumption was
  plausible and wrong, and it was in this README until it was measured.

  Depth profile at horizon 63 (excess vs universe, HAC t):
  **top 5 +2.18% (t +1.40)**, **top 15 +2.96% (t +2.35)**,
  **top 25 +2.74% (t +2.51)**, **top 50 +2.34% (t +2.58)**. The traded top-15
  band clears |t| = 2 here, unlike at horizon 10 — but note the t-statistic
  *rises* with depth while the excess falls, so significance is coming from
  lower variance, not stronger signal.

- **Fundamentals still trail the price-only control, but the gap was mostly a
  coverage defect.** SEC EDGAR point-in-time features (`--fundamentals`) take
  **31–46% of total model gain**, all 11 used.

  An earlier version of this README called the variant *unstable*, on the
  evidence that a re-download moved its Sharpe 0.68 → 0.49 while the control
  held at 0.69. That reading was wrong. Most of the swing was missing data,
  not instability. On an **identical panel** — same 639 tickers, same 941,381
  rows, same sessions, only the fundamental values changed — repairing
  coverage moves the variant:

  | metric | old coverage | fixed coverage | control |
  |---|---|---|---|
  | return IC | +0.0232 (t +1.43) | **+0.0272 (t +1.81)** | +0.0374 (t +1.64) |
  | top-15 excess | +2.12% (t +2.37) | **+2.39% (t +2.99)** | +2.96% (t +2.35) |
  | L-S spread | +9.3%/yr (t +1.75) | **+12.4%/yr (t +2.54)** | +17.5%/yr (t +2.31) |
  | Sharpe | 0.49 | **0.69** | 0.69 |

  Sharpe recovers to exactly the control's. The control still wins on IC and
  on spread magnitude, so fundamentals stay out of the default feature set —
  but on *significance* the fundamentals variant is now the stronger of the
  two (depth t +2.99 vs +2.35, long-short t +2.54 vs +2.31), earning smaller
  excess returns with lower variance.

  The lesson is methodological: a feature set that is silently 22% complete
  will read as a weak or unstable signal, and the honest first move on a
  disappointing feature block is to measure its coverage before concluding
  anything about its content.

- **Honest results disclosure.** On a corrected 2023–2026 walk-forward panel (544 tickers spanning the full alphabet, 100% of current index members, features staged before the PIT filter, purged splits, cost-inclusive NAV, HAC alpha t-stats, `--rf-rate 0.045`), **this model has no demonstrable stock-selection skill at the top of its ranking — the only part the strategy trades.**

  Baseline `--top-n 15`: total return **+22.2%** against **+105.2%** for SPY and **+61.5%** for equal-weight RSP over the same sessions; Sharpe **0.16**; max drawdown **−27.7%**. CAPM alpha is **−12.6% (HAC t = −1.64)** vs SPY and **−7.5% (t = −0.97)** vs RSP. Against the equal-weight benchmark — the fair comparison for an equal-weighted strategy, especially across a mega-cap-dominated tape — that is statistically indistinguishable from zero. It is *no skill*, not proven harm.

  The diagnostic that matters is how alpha responds to conviction. A ranker with skill earns more alpha as you concentrate. This one earns less:

  | `--top-n` | CAPM alpha vs SPY | HAC t |
  |---|---|---|
  | 5 | −23.6% | −2.11 |
  | 10 | −17.9% | −2.09 |
  | 15 | −12.6% | −1.64 |
  | 25 | −10.0% | −1.51 |
  | 50 | −6.1% | −1.12 |

  Mean 10-day forward return by selection depth (equal-weighted, before costs) shows why — the extreme top of the ranking is the worst part of it:

  | Bucket | Mean 10d fwd return | vs universe | HAC t |
  |---|---|---|---|
  | top 5 | −0.2393% | −0.81% | −1.57 |
  | top 10 | +0.2931% | −0.28% | −0.65 |
  | top 15 | +0.4423% | −0.13% | −0.34 |
  | top 25 | +0.5703% | −0.00% | −0.01 |
  | top 50 | +0.7176% | +0.14% | +0.61 |
  | top 100 | +0.6892% | +0.12% | +0.67 |
  | bottom 100 | +0.4072% | −0.17% | −1.25 |
  | universe | +0.5730% | — | — |

  No single bucket clears |t| = 2 on its own, so read the *ordering*, not any one row: forward return rises monotonically from the top of the list down to the top-50 band, then falls away again. Rank IC is **+0.0082 (HAC t = +0.49)** — indistinguishable from noise. The earlier claim that this model showed "real cross-sectional ranking skill, roughly enough to pay its own trading costs" did not survive the universe and feature-staging fixes.

  Reproduce all of the above on any panel with [`scripts/signal_depth.py`](scripts/signal_depth.py) — this disclosure is meant to be re-derived, not trusted.

  **The label mattered; tuning did not — under any metric.** Switching to the
  rank objective is the one change that improved the traded end of the ranking
  and held its sign across a 2023-24 / 2025+ split. Tuning did not add to it,
  including after the tuning objective was rewritten to measure the traded
  rule itself (`--optuna-metric topn_excess` / `topn_ir`):

  | config | top-5 excess | HAC t | return IC | alpha vs RSP | t |
  |---|---|---|---|---|---|
  | rank, untuned | **+0.91%** | +1.90 | **+0.0144** | −2.7% | −0.35 |
  | rank, tuned NDCG@15 | +0.41% | +0.83 | +0.0023 | +1.0% | +0.13 |
  | rank, tuned `topn_excess` | +0.10% | +0.20 | +0.0049 | +1.6% | +0.20 |
  | rank, tuned `topn_ir` | +0.65% | +1.42 | +0.0084 | +2.8% | +0.35 |

  Note the shape of that failure: `topn_excess` *maximizes* top-N excess
  forward return, and the model it produced has the **worst** top-5 excess of
  the four. Optuna scored 0.0090 on that metric across pre-2023 folds and none
  of it carried to 2023+. Every tuned variant also has lower return IC than
  the untuned model. The tuning folds sit entirely before the evaluation
  window, so this is a generalization failure, not leakage.

  Full-window alpha turns mildly positive for all three tuned variants, which
  looks like progress until the split:

  | config | alpha 2023-24 | alpha 2025+ |
  |---|---|---|
  | rank, untuned | −8.3% (t −1.09) | +3.5% (t +0.24) |
  | rank, `topn_excess` | −5.1% (t −0.52) | +7.1% (t +0.55) |
  | rank, `topn_ir` | −11.6% (t −1.39) | **+21.1%** (t +1.45) |

  `topn_ir`'s +21.1% out-of-sample alpha is the most eye-catching number this
  repo can produce, and it arrives attached to −11.6% in-sample. That is
  regime dependence, not edge. Nothing anywhere clears |t| = 2.

  **The tree counts are the clearest signal.** Now that early stopping uses
  the metric being maximized, validation performance peaks at 2 trees
  (`topn_excess`), 4 (`topn_ir`) and 10 (NDCG@15). A ranker that cannot
  justify more than ten trees has found almost nothing learnable in these
  features — and a 2-tree `model.pkl` is unusable for `predict-sp500`, so
  check `Best iteration` in the log before shipping a tuned model.

  The earlier NDCG-specific finding, for reference:

  | config | top-5 excess (2023-24) | (2025+) | alpha vs RSP (2023-24) | (2025+) |
  |---|---|---|---|---|
  | binary +5% | −1.35% (t −2.42) | −0.13% | −14.8% | +1.2% |
  | rank, untuned | **+0.71%** (t +1.24) | **+1.17%** (t +1.43) | −8.3% | +3.5% |
  | rank, Optuna-tuned | −0.24% (t −0.41) | +0.97% (t +1.03) | −7.2% | −1.5% |

  Tuning flattened the top of the list — the tuned depth profile is nearly
  uniform (0.86% / 0.82% / 0.85% / 0.81% across top 5 / 15 / 50 / 100) and its
  in-sample top-5 excess turns slightly negative. NDCG@15 over ~600 names
  scored on quintile grades rewards broad ordering, so maximizing it is not
  the same as maximizing the top-15 basket's return. That diagnosis was
  correct and `--optuna-metric` fixes it, but fixing it did not help: see the
  table above. **`--no-optuna` remains the recommended setting for the rank
  objective.**

  Configurations that look good still get there through beta. `--mode rank-hold --exit-rank 100` returns **+97%** (Sharpe 0.65) with **beta 1.40** and alpha **−6.7%** — leverage, not selection. Treat any better-looking result you produce here with the same suspicion: read the HAC alpha t-stat and the beta column, compare against RSP as well as SPY, and re-test on a sub-window the configuration has never seen.
- **Sector labels** are a pragmatic blend of current Wikipedia GICS and a fixed override; they are not a perfect point-in-time sector history for every ticker-date.
- **Earnings** come from Yahoo as-of download time; the feature is not a fully audited point-in-time fundamental database.
- **Delisted tickers**: Yahoo no longer serves many departed S&P members (~90–140 depending on the window), so even with PIT membership the price panel has survivorship bias that flatters results. A Tiingo key recovers most of them.
- **FOMC / calendar** helpers depend on maintained date lists—verify critical dates for your own research.
- **Survivorship and data snooping**: even with PIT index membership, corporate actions, delistings, and feature lookahead need careful review for any live use.
