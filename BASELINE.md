# Baseline

Every performance figure this project published before 2026-08-28 was measured
on artifacts whose provenance was not recorded, and several were wrong for
reasons unrelated to the strategy — a look-ahead in cohort construction, a
scored panel pricing its own fills, an alpha that was mostly the risk-free
rate. **Those numbers are obsolete.** This file records what replaces them,
what verified it, and how much of it is trustworthy.

The short version: after rebuilding cleanly and measuring properly, the
long-only strategy shows **no statistically significant alpha**, and its
headline return is **not stable enough to quote to two decimals**.

---

## Provenance

| | |
|---|---|
| commit | `df296a6ebc68` (clean tree) |
| run id | `20260828T053022Z_94f664aa` |
| built by | `./scripts/rebuild_baseline.sh` |
| verified by | `uv run python scripts/verify_baseline.py artifacts/baseline` |

Configuration, pinned and recorded in `artifacts/baseline/config.json`:

| | |
|---|---|
| provider | `hybrid` (yfinance + Tiingo for delisted names) |
| data window | 2010-01-01 → 2026-08-28 (`--end` pinned) |
| train / test split | train ≤ 2018-12-31, out-of-sample from 2019-01-01 |
| universe | full point-in-time membership, no cap |
| horizon | 63 sessions | 
| objective | LambdaRank, no Optuna, seed 42 |
| strategy | top-15, 63-day hold, 2 overlapping cohorts, 5bps slippage, Friday |

Input snapshot hashes (`artifacts/baseline/snapshot/manifest.json`):

| snapshot | sha256 (16) | rows |
|---|---|---|
| `equity_prices_long` | `0b39d7e0016999b4` | 3,538,860 |
| `execution_prices` | `0b39d7e0016999b4` | 3,538,860 |
| `features_clean` | `191ccd6fd96f2345` | 1,906,663 |
| `labeled` | `0e6334b2101dd6f4` | 2,774,557 |
| `stints` | `2f2953aa2fa5207d` | 1,259 |

Scored panel: 940,720 rows, 1,924 sessions, 640 tickers.
Execution panel: 4,188 sessions × 845 tickers.

---

## Verification

All eight gates pass. Each is a hard failure, not a warning.

| gate | result |
|---|---|
| point-in-time integrity | 0 of 940,720 scored rows outside index membership; labels stop exactly at the last labelable session; execution covers every scored row; scored and execution prices agree on 940,720/940,720 cells |
| survivorship | 313 of 358 departed names carry prices — **at the measured vendor ceiling**, see below |
| accounting (cohort) | NAV reconciles with the trade ledger to `0.00e+00` |
| accounting (rank-hold) | reconciles to `1.14e-16`, with 15 open positions and +45,910.79 unrealized |
| fills (both engines) | zero stale fills; every refusal disposed under a stated policy |
| deterministic backtest (both) | identical NAV hashes on repeat |

### What the survivorship gate tolerates

45 of the 358 companies that left the index during the window carry no usable
history, listed by name in `artifacts/baseline/survivorship_gap.json`. Tiingo
returns 0–12 rows for each and returns the same on a refetch, so the gate's bar
is the **measured ceiling (87.4%)**, not a threshold chosen to pass. A name
missing because a quota ran out still fails.

**At least 12 of those 45 are not data gaps at all but ticker renames** — ABC→COR,
ANTM→ELV, CBS→PARA, CDAY→DAY, COG→CTRA, PEAK→DOC, PKI→RVTY, RE→EG, WLTW→WTW,
CTL→LUMN, TMK→GL, BLL→BALL — whose successors are already in the panel with full
history. Mapping them would raise the ceiling to roughly 90% at no cost. Four
more (BK, MMC, ATGE, IGT) are *currently listed* companies with 0–5 rows, which
is a download fault rather than a vendor gap. Neither is fixed here.

---

## Results

Out-of-sample 2019-01-02 → 2026-08-28, four independent rebuilds from the same
commit and the same pinned window.

| engine | CAGR | Sharpe | max drawdown |
|---|---|---|---|
| cohort (top-15, 63d) | **17.76% ± 3.58%** | 0.61 ± 0.10 | −37.8% ± 2.9% |
| rank-hold (exit rank 40) | **22.48% ± 2.07%** | 0.66 ± 0.05 | −55.0% ± 1.5% |
| SPY, same window | 17.60% | — | — |

Against SPY, with CAPM on excess returns and Newey–West standard errors:

| engine | beta | alpha/yr | HAC t | information ratio |
|---|---|---|---|---|
| cohort | **+1.138** | +1.00% | **+0.20** | 0.15 |
| rank-hold | **+1.392** | +4.86% | **+0.62** | 0.38 |

**Neither alpha is distinguishable from zero.** The book beats the index on raw
return because it carries beta 1.14–1.39 — it is leveraged market exposure, not
selection skill. A cohort CAGR of 17.76% against SPY's 17.60%, with a 3.58%
run-to-run standard deviation, is not evidence of an edge.

Signal quality, walk-forward, per signal date: precision@15 ≈ 0.47, rank IC
≈ +0.05, top-15 excess ≈ +3.1% per 63-session horizon. The ranking carries some
information; it does not survive the cost of trading it.

---

## The reproducibility limit

The four runs above used **one commit, one pinned data window, one seed**. Their
execution panels agree to `2e-6` relative — float noise in the vendor's
adjustment arithmetic, not revised data, with identical coverage. They produced
cohort CAGRs of 13.85%, 16.58%, 18.21% and 22.40%.

LightGBM splits flip on near-ties, the ranking changes, and a different fifteen
names get held. **All four passed every gate.**

Three consequences, and they are not small:

1. **Two-decimal precision is fiction.** Any figure from this pipeline carries
   roughly ±3.6 points of CAGR at one sigma. Every number this project has ever
   quoted implied a precision that does not exist.
2. **Most historical comparisons were noise.** Configuration A beating
   configuration B by two or three points of CAGR says nothing. The search
   described under *"multiplicity"* in the README compared options whose true
   differences sit well below this floor.
3. **Bit-reproducibility needs snapshot replay.** The snapshots are hashed and
   written; the pipeline cannot yet consume one as input. Until it can, "rerun
   the baseline" means "draw again from the same distribution", not "reproduce".

---

## What is retired

Every performance number published before this file, including:

- all cohort-engine CAGR/Sharpe/drawdown figures measured without an execution
  panel — those overstate CAGR by roughly 6.6 points and understate drawdown by
  18 (see the cohort look-ahead entry in the README);
- every result from `backtest_sweep.py`, `grid_search_sharpe.py` and
  `signal_depth.py` prior to those scripts accepting `--execution-prices`;
- any alpha computed on raw rather than excess returns;
- any figure quoted to two decimals without a run-to-run spread.

The long-short book is **not** covered here. It is not wired into any CLI, so it
has no baseline, and its previously quoted figures remain unreproduced.

## Reproducing this

```bash
./scripts/rebuild_baseline.sh                                   # ~30 min, warm cache
uv run python scripts/verify_baseline.py artifacts/baseline
```

If the survivorship gate reports recoverable names, fill them first — Tiingo's
free tier allows ~50–76 new tickers per window, so this resumes:

```bash
uv run python scripts/recover_delisted.py --passes 8 --wait 3700
```

Expect the headline numbers to land within the spread above, not on them.
