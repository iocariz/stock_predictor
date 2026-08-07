# stock-predictor

LightGBM classifier that predicts which S&P 500 stocks will gain ≥ 5% over the next 10 trading days. Built for educational research into quantitative equity screening—not production trading.

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

## Usage

### Training CLI

```bash
uv run train-sp500 --help

# Fast smoke run (small universe, no Optuna, no earnings fetch)
uv run train-sp500 --sample-n 100 --no-optuna --skip-earnings

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

**Input:** a Parquet or CSV panel from walk-forward scoring (`--wf-scores-path` or the notebook). Required columns: **`date`**, **`ticker`**, **`adj_close`**, and either **`prob`** or **`probability`**. Optional **`vix_percentile`** enables **`--vix-filter`**.

```bash
uv run backtest-sp500 --help

uv run backtest-sp500 path/to/wf_scored.parquet --plots-dir artifacts/plots
uv run python backtest.py path/to/wf_scored.parquet --plots-dir artifacts/plots

# Offline (no benchmark download); optional rebalance / benchmark ticker
uv run backtest-sp500 scores.parquet --no-benchmark --rebalance-day last
uv run backtest-sp500 scores.parquet --benchmark-ticker VOO --plots-dir artifacts/plots
```

Benchmark prices are **reindexed to the strategy’s trading days** (forward-filled) so strategy vs buy-and-hold metrics use the **same calendar**.

#### Backtest flags

| Flag | Default | Description |
|------|---------|-------------|
| `scored_path` | — | Parquet (`.parquet`) or CSV with a `date` column |
| `--top-n` | 15 | Number of tickers per cohort (ranked by score) |
| `--holding-days` | 10 | Holding period in **trading** days |
| `--rebalance-day` | Friday | `Monday`–`Friday` or `last` (last session in each ISO week) |
| `--weighting` | equal | `equal` or `probability` (normalize scores to weights) |
| `--slippage-bps` | 5 | Round-trip modeled as **per-side** basis points |
| `--capital` | 100000 | Starting notional |
| `--max-cohorts` | 2 | Overlapping cohorts (capital / max_cohorts per slot) |
| `--vix-filter` | none | Skip rebalance when `vix_percentile` exceeds this (requires column in data) |
| `--benchmark-ticker` | SPY | yfinance symbol for buy-and-hold column |
| `--no-benchmark` | off | Skip benchmark download (table shows N/A for benchmark) |
| `--plots-dir` | none | Writes `equity_curve.png`, `drawdown.png`, `monthly_returns.png` |
| `--compare-with` | none | Second scored Parquet/CSV; same rules as primary (shared flags) |
| `--compare-label-a` | stem of primary file | Legend label for primary strategy |
| `--compare-label-b` | stem of compare file | Legend label for `--compare-with` |
| `--commission-per-share` | 0 | Dollars per share on each buy and sell leg (0 = off) |
| `--commission-per-order` | 0 | Flat dollars per ticker on each buy and sell order (0 = off) |

#### Execution assumptions (backtest vs predict)

- **Calendar:** Both paths use the **union of session dates present in the data** (scored panel for the backtest; downloaded OHLC index for `predict-sp500`), not a full exchange holiday calendar.
- **Backtest:** Signal on date *T* → **entry** on the next session in that calendar → **exit** after `--holding-days` **trading sessions** (same offset as `exit` in `Cohort`). Cohort returns are **fractional** weights; slippage is applied to historical **adj. close** bars.
- **predict-sp500:** `--as-of` is **calendar** *today*; **entry** is the first session in the OHLC index on or after that day; **expiry** uses the same **trading-day count** as the backtest. Orders use **integer shares** (lot sizes differ from a fractional simulation).
- **Commissions:** When non-zero, fees reduce **cohort `net_return` and `total_costs`** in the backtest. The **daily NAV time series** still marks positions using prices only (it does not apply the same cash drag as the cohort-level cost line), so risk metrics from NAV can be **slightly optimistic** when commissions are large—use `total_costs` and cohort stats as a cross-check.
- **Parity:** Use the same `--weighting`, `--slippage-bps`, `--holding-days`, and commission flags in both CLIs when comparing research simulation to generated orders.

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

#### Strategy ideas to try (research / educational)

| Idea | What to vary | How to approximate here |
|------|----------------|---------------------------|
| **Model A vs model B** | Different WF score files | Train twice (`--output-model` / different seeds or features), export two `--wf-scores-path` panels, compare |
| **Equal vs probability weights** | Position sizing | Same `wf_scored.parquet`, run backtest twice with `--weighting equal` vs `probability` (no code change to scores) |
| **Concentration** | `top_n` | Same scores, compare `--top-n 5` vs `--top-n 20` (two CLI runs or two configs in a notebook) |
| **Holding horizon** | `holding_days` vs label horizon | Align `holding_days` to your forward window or try shorter/longer holds on the same scores |
| **Rebalance rhythm** | `--rebalance-day Friday` vs `last` vs Monday | Same scores, different execution calendar |
| **Risk-off filter** | VIX / regime | Build scores with `vix_percentile` in the panel; compare `--vix-filter` off vs on (or different cutoffs) |
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
| `--sample-n` | 500 | Max tickers to download |
| `--horizon` | 10 | Forward return horizon (sessions) |
| `--threshold` | 0.05 | Binary label threshold (e.g. 5%) |
| `--no-optuna` | off | Skip Optuna search; use defaults + any prior best |
| `--optuna-trials` | 40 | Optuna trials |
| `--ts-cv-splits` | 5 | `TimeSeriesSplit` folds for tuning |
| `--seed` | 42 | RNG seed |
| `--skip-earnings` | off | Omit Yahoo earnings feature (faster) |
| `--earnings-workers` | 8 | Parallelism for earnings fetch |
| `--skip-walk-forward` | off | Skip monthly walk-forward |
| `--wf-min-train-rows` | 5000 | Minimum training rows per WF month |
| `--wf-top-k` | 10 | Top-K for weekly precision / WF ranking |
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

Default **`--sample-n` is 500** (caps the universe). Use a **large** value (e.g. `10000`) to include essentially all tickers overlapping your date window.

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
| `--top-n` | 15 | Stocks per cohort |
| `--max-cohorts` | 2 | Max overlapping cohorts |
| `--holding-days` | 10 | Trading days until cohort expiry |
| `--max-drawdown` | 0.15 | Kill-switch: halt if drawdown exceeds this |
| `--slippage-bps` | 5 | Slippage per side (basis points) |
| `--skip-earnings` | off | Skip earnings feature (must match training) |
| `--sample-n` | 500 | Max tickers to download |
| `--weighting` | equal | `equal` or `probability` (same as backtest) |
| `--commission-per-share` | 0 | Per-share fee per buy/sell leg (match `backtest-sp500`) |
| `--commission-per-order` | 0 | Flat fee per ticker per buy/sell order (match `backtest-sp500`) |
| `--no-macro-merge` | off | Same as training: skip Yahoo↔FRED macro merge |
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
| `notebooks/exploration.ipynb` | EDA |
| `notebooks/sp500_predictor.ipynb` | Full predictor pipeline |

Ensure the repo root or `src` is on `PYTHONPATH`, or run notebooks from an environment where `uv sync` has been applied. Root shims re-export `stock_predictor.pit` and `stock_predictor.calendar_features` for older import paths.

### Tests

```bash
uv run pytest
```

Covers PIT membership, calendar features, reproducibility, training utilities, backtest engine, portfolio management, execution parity, and inference pipeline (`tests/`).

## Reproducibility (snapshots + manifest)

When snapshots are enabled (default), each run writes under `--snapshot-dir` or `artifacts/runs/<utc_run_id>/`:

| File | Contents |
|------|----------|
| `manifest.json` | `run_id`, UTC time, `argv`, git commit / dirty (if available), env hints, **SHA-256** + row/column counts per snapshot |
| `stints.parquet` | PIT membership table |
| `equity_prices_long.parquet` | Long-format adjusted close + volume |
| `labeled_pit.parquet` | Panel after forward return + PIT filter |
| `features_clean.parquet` | Feature matrix after `dropna` on model inputs |

With `--output-model`, the manifest also records the pickle path and checksum; the manifest is rewritten at the end of the run.

With **`--no-snapshot`**, no Parquet files and no `manifest.json` are written (the `run_id` may still appear in saved model metadata).

`artifacts/` is gitignored by default—copy snapshots elsewhere if you need to keep them in version control or upload them to object storage.

## Pipeline

1. **Universe**: Point-in-time S&P 500 membership via [fja05680/sp500](https://github.com/fja05680/sp500)
2. **Prices**: Daily adjusted close + volume from Yahoo Finance (`yfinance`)
3. **Labels**: Binary — forward return ≥ `--threshold` over `--horizon` sessions
4. **Features**: Price/volume (15), sector-relative (4), macro (5), regime (2), cross-sectional ranks (3), calendar (6), optional earnings (1)
5. **Tuning**: Optuna + `TimeSeriesSplit` (optional)
6. **Training**: LightGBM with inner validation for effective tree count
7. **Evaluation**: PR-AUC, ROC-AUC, weekly Precision@K
8. **Walk-forward**: Monthly expanding window (optional)
9. **Backtest**: Weekly-rebalance long-only simulation vs configurable benchmark
10. **Inference**: Daily scoring of the live universe, portfolio state management, order generation with kill-switch risk control

## Project layout

```
src/stock_predictor/
  __init__.py            Version + re-exports
  pit.py                 PIT S&P 500 membership
  calendar_features.py   Calendar / FOMC features
  training.py            Features, Optuna, train/eval, model IO
  cli.py                 train-sp500 entry point
  backtest.py            Backtest engine + CLI
  backtest_reporting.py  Reporting + plots for backtest results
  portfolio.py           Portfolio state, orders, kill-switch
  predict.py             Daily inference CLI (predict-sp500)
  repro.py               Run manifests, hashing, Parquet snapshots
train_sp500.py           Thin launcher → cli
backtest.py              Thin launcher → backtest
sp500_pit.py             Notebook shim → pit
calendar_features.py     Notebook shim → calendar_features
tests/                   pytest suite (51 tests)
notebooks/               Exploration + full pipeline
```

## Data sources

| Source | What | Notes |
|--------|------|-------|
| [fja05680/sp500](https://github.com/fja05680/sp500) | PIT membership stints | Community-maintained |
| Yahoo Finance (`yfinance`) | OHLCV, macro, earnings | Rate-limited; data quality varies |
| Wikipedia | GICS sector mapping | Snapshot + hard-coded 2018 override—not a full historical sector time series |

## Limitations

- **Not investment advice.** Past backtests and metrics do not guarantee future results.
- **Sector labels** are a pragmatic blend of current Wikipedia GICS and a fixed override; they are not a perfect point-in-time sector history for every ticker-date.
- **Earnings** come from Yahoo as-of download time; the feature is not a fully audited point-in-time fundamental database.
- **FOMC / calendar** helpers depend on maintained date lists—verify critical dates for your own research.
- **Survivorship and data snooping**: even with PIT index membership, corporate actions, delistings, and feature lookahead need careful review for any live use.
