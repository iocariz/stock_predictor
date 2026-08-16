# stock-predictor

LightGBM model that ranks S&P 500 stocks by expected 10-day performance. Two training objectives are supported: a **binary classifier** (probability of gaining ≥ 5% over the next 10 sessions) and a **lambdarank ranker** (`--rank-objective`) trained on per-date forward-return quintile grades — a cross-sectional target that is market-neutral by construction. Built for educational research into quantitative equity screening—not production trading.

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

- **Calendar:** Both paths use the **union of session dates present in the data** (scored panel for the backtest; downloaded OHLC index for `predict-sp500`, extended with future business days for entry/expiry placement), not a full exchange holiday calendar.
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
| `--rank-objective` | off | Train LGBMRanker (lambdarank, grouped by date) on per-date forward-return quintile grades instead of the binary classifier; applies to Optuna (NDCG@15), the walk-forward, and the saved model |
| `--no-optuna` | off | Skip Optuna search; use defaults + any prior best |
| `--optuna-trials` | 40 | Optuna trials |
| `--ts-cv-splits` | 5 | Purged, **date-grouped** expanding CV folds for tuning (no trading day straddles a fold; last `horizon` dates before each validation block are excluded) |
| `--seed` | 42 | RNG seed |
| `--strict-dropna` | off | Drop rows with ANY NaN feature (legacy); default keeps them — LightGBM handles missing values natively |
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
| `notebooks/exploration.ipynb` | EDA |
| `notebooks/sp500_predictor.ipynb` | Full predictor pipeline |

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
3. **Labels**: Binary (forward return ≥ `--threshold` over `--horizon` sessions) or, with `--rank-objective`, per-date forward-return quintile grades (market-neutral by construction)
4. **Features**: Price/volume (15), sector-relative (4), macro (5), regime (2), cross-sectional ranks (3), calendar (6), optional earnings (1)

   **Stage order is load-bearing** (`build_feature_panel`):

   1. *Time-series* features (price, volume) on each ticker's **full contiguous history**
   2. *Point-in-time membership filter*
   3. *Cross-sectional* features (regime medians, ranks, sector spreads) on the surviving in-index cross-section
   4. Date-level joins (macro, calendar) and the optional earnings feature

   Filtering before step 1 lets rolling windows span index-membership gaps — a symbol that left and rejoined the index gets a multi-year move reported as its next `ret_1d`. Running step 3 before the filter pollutes ranks and "market" medians with names that were not in the index that day.
5. **Tuning**: Optuna over purged, date-grouped expanding CV folds — PR-AUC for the classifier, NDCG@15 for the ranker (optional)
6. **Training**: LightGBM (classifier or lambdarank ranker) with a purged inner validation split for effective tree count
7. **Evaluation**: PR-AUC, ROC-AUC, weekly Precision@K (same binary target for both objectives, so they compare directly)
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
  backtest.py            Backtest engines (cohort + rank-hold) + CLI
  backtest_reporting.py  Reports, relative-return metrics (IR/beta/alpha/t), plots
  portfolio.py           Portfolio state, fixed & rank-hold orders, kill-switch
  predict.py             Daily inference CLI (predict-sp500)
  execution_calendar.py  Trading-day calendar helpers (shared backtest/live)
  macro_merge.py         Yahoo ↔ FRED macro cross-fill
  data_provider.py       Provider protocol + factory
  providers/             yfinance + Tiingo/FRED implementations
  universe.py            Seeded ticker sampling + download coverage guard
  signal_depth.py        Selection-depth diagnostics (does the top of the ranking work?)
  stats.py               HAC / Newey-West helpers
  repro.py               Run manifests, hashing, Parquet snapshots
scripts/
  run_pipeline.sh        train-full / backtest / predict orchestration
  backtest_sweep.py      Variant grids (default / vix / hold) + relative tables
  grid_search_sharpe.py  Config grid ranked by Sharpe, with alpha t-stats
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
- **Not investment advice.** Past backtests and metrics do not guarantee future results.
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

  Configurations that look good still get there through beta. `--mode rank-hold --exit-rank 100` returns **+97%** (Sharpe 0.65) with **beta 1.40** and alpha **−6.7%** — leverage, not selection. Treat any better-looking result you produce here with the same suspicion: read the HAC alpha t-stat and the beta column, compare against RSP as well as SPY, and re-test on a sub-window the configuration has never seen.
- **Sector labels** are a pragmatic blend of current Wikipedia GICS and a fixed override; they are not a perfect point-in-time sector history for every ticker-date.
- **Earnings** come from Yahoo as-of download time; the feature is not a fully audited point-in-time fundamental database.
- **Delisted tickers**: Yahoo no longer serves many departed S&P members (~90–140 depending on the window), so even with PIT membership the price panel has survivorship bias that flatters results. A Tiingo key recovers most of them.
- **FOMC / calendar** helpers depend on maintained date lists—verify critical dates for your own research.
- **Survivorship and data snooping**: even with PIT index membership, corporate actions, delistings, and feature lookahead need careful review for any live use.
