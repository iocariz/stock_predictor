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
| commit | `4a9e226c4008` (clean tree) |
| run id | `20260830T204011Z_f814aa3d` |
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
| `equity_prices_long` | `fc1f0abcdcdbb765` | 3,492,792 |
| `execution_prices` | `fc1f0abcdcdbb765` | 3,492,792 |
| `features_clean` | `2c6f6c6bbe4b41b3` | 1,939,552 |
| `labeled` | `398d274e3197ee86` | 2,790,875 |
| `macro` | `3592ae0a96d169d0` | 4,227 |
| `sector_map` | `408da130fa1f0985` | 503 |
| `stints` | `27dbf956b96a35ae` | 1,247 |

Scored panel: 952,329 rows, 1,924 sessions, 643 tickers.
Execution panel: 4,188 sessions × 834 tickers.

---

## Verification

All ten gates pass. Each is a hard failure, not a warning. Verification is
**read-only and self-contained**: it reads membership from
`snapshot/stints.parquet` and the vendor-absent set from `vendor_absent.json`
recorded at build time, never from live state, and writes nothing into the
baseline. Pass `--report` to emit the survivorship residual elsewhere.

| gate | result |
|---|---|
| snapshot integrity | all five artifacts recomputed and matching their recorded sha256 |
| ticker renames | all 15 checked against this panel's own prices: each successor carries the predecessor's membership, and none trade concurrently after the effective date |
| point-in-time integrity | 0 of 940,720 scored rows outside index membership; labels stop exactly at the last labelable session; execution covers every scored row; scored and execution prices agree on 940,720/940,720 cells |
| survivorship | 317 of 347 departed names carry prices (**91.4%**, the measured vendor ceiling); departed names are scored on **95.9%** of the sessions they were members (118,725/123,859 name-sessions) |
| accounting (cohort) | NAV reconciles with the trade ledger to `0.00e+00` |
| accounting (rank-hold) | reconciles to `1.14e-16`, with 15 open positions and +45,910.79 unrealized |
| fills (both engines) | zero stale fills; every refusal disposed under a stated policy |
| deterministic backtest (both) | identical NAV hashes on repeat |

### Ticker renames

A renamed company looked exactly like a delisted one. Stints name companies by
the symbol they used at the time; prices are served under the symbol they use
now. So Anthem's 2002–2022 membership pointed at `ANTM`, which nothing prices,
and the company was absent from the cross-section for the twenty years it was
actually a member.

Fifteen such renames are now resolved (`src/stock_predictor/renames.py`), and
the two halves of each membership rejoined into one stint. Survivorship
coverage went **87.4% → 91.3%**, and the scored panel gained 11,609 rows. The
predecessor symbol is recorded in an `alias` column rather than discarded
(`specs.md:157`), which is also what lets the map be checked from the baseline
afterwards.

**What the validation does and does not establish.** Coverage shows the
successor prices the predecessor's membership: necessary, not sufficient — a
successor that simply has long history satisfies it whatever its identity.
Each entry carries an effective date, which supplies the one real falsifier
available from prices: after the symbol changed only one of the two can trade,
so any session where both print refutes the claim. Neither test establishes
*issuer identity*. That needs a permanent identifier (CUSIP/CIK/FIGI) this
project does not carry, and each entry's recorded note remains the warrant.

Every entry is validated against prices before it is trusted, because the
plausible ones are not all real. **CBS→PARA, RX→IQV, ESV→VAL and MDP→IAC were
rejected** — a merger, a re-IPO, a post-bankruptcy relisting and a break-up —
each scoring *zero* coverage across the predecessor's stint. Trusting the
plausible list would have attached Paramount's returns to CBS's membership.

### What the survivorship gate still tolerates

30 of the 347 companies that left the index carry no usable history, listed in
`artifacts/baseline/survivorship_gap.json`. Both vendors return 0–12 rows for
each and return the same on a refetch, so the bar is the **measured ceiling**,
not a threshold chosen to pass; a name missing because a quota ran out still
fails.

Four of the 30 — **BK, MMC, ATGE, IGT** — are *currently listed* index members
that neither yfinance nor Tiingo will serve. MMC and BK are large and liquid,
so this is a vendor-side gap rather than anything this code does, but it means
two current members are absent from every panel here.

---

## Results

Out-of-sample 2019-01-02 → 2026-08-28, four independent rebuilds from the same
commit and the same pinned window.

| engine | CAGR (n=4) | this artifact | Sharpe | max drawdown |
|---|---|---|---|---|
| cohort (top-15, 63d) | **20.24% ± 2.45%** | 15.43% | 0.67 ± 0.07 | −45.2% ± 2.6% |
| rank-hold (exit rank 40) | **26.41% ± 1.46%** | 21.11% | 0.73 ± 0.03 | −49.8% ± 3.6% |
| SPY, same window | 17.60% | 17.60% | — | — |

The `n=4` column is the spread measured across four rebuilds; `this artifact`
is the single run recorded above. **Both fall below the four-run band** —
rank-hold by more than three of its standard deviations. Four runs is too few
to have pinned the spread, and the honest reading is that it is wider than
±1.5 points, not that this run is anomalous. Treat the range, not either
column, as the estimate.

Against SPY, with CAPM on excess returns and Newey–West standard errors, mean
of the same four runs:

| engine | beta | alpha/yr | HAC t | information ratio |
|---|---|---|---|---|
| cohort | **+1.190** | +2.36% ± 2.13% | **+0.37** | +0.24 |
| rank-hold | **+1.431** | +6.80% ± 1.22% | **+0.84** | +0.46 |

**Neither engine shows evidence of skill.** Cohort alpha is +2.36% at t = +0.37,
ranging from −0.05 to +0.77 across the four runs; rank-hold is +6.80% at
t = +0.84 and never exceeds t = 1.03. Both carry beta well above 1 (1.19 and
1.43), so most of what they beat the index by is leverage, not selection.

Two corrections landed on these figures and are recorded because the direction
matters. Resolving ticker renames moved cohort alpha from +1.00% to −2.47%.
Fixing the cohort expiry boundary — a cohort selling on a signal session was
counted as still holding its slot, and its cash withheld from that session's
entry, on 13 of 57 cohorts — moved it back to +2.36% and CAGR from 15.28% to
20.24%. The sign of the cohort engine's alpha was an artifact of an off-by-one
in both directions; its *insignificance* was not.

Signal quality, walk-forward, per signal date: precision@15 ≈ 0.47, rank IC
≈ +0.05, top-15 excess ≈ +3.1% per 63-session horizon. The ranking carries some
information; it does not survive the cost of trading it.

---

## Reproducibility — solved

`train-sp500 --replay-snapshot artifacts/baseline` verifies the recorded hashes
and then rebuilds from them: prices, macro, membership and the sector map all
come from the snapshot. Feature engineering, labelling, training and the
walk-forward re-run as normal, so this replaces the *inputs*, not the pipeline
— a code change still shows, the data no longer moves underneath it.

Two of the five external inputs were not being recorded at all. The sector map
and the macro series were fetched mid-build, so a run could not be reproduced
from what it saved even in principle. Both are captured now.

Measured on this baseline: two independent replays produced **byte-identical**
`wf_scored.parquet` and `execution_prices.parquet`, and both reproduce the
original run's backtest to four decimals.

| | cohort CAGR | max DD | rank-hold CAGR | max DD |
|---|---|---|---|---|
| original run | 18.5206% | −44.6400% | 22.2642% | −57.9037% |
| replay A | 18.5206% | −44.6400% | 22.2642% | −57.9037% |
| replay B | 18.5206% | −44.6400% | 22.2642% | −57.9037% |

Replay reproduces the *run*, not merely itself. A comparison between two
measurements from one snapshot is now a comparison of the change, not of two
draws — which is what makes any of the numbers below worth arguing about.

**Fresh rebuilds still differ**, and that is a separate thing: re-downloading
draws new vendor float noise and lands somewhere in the spread below. Use
replay to compare code changes; use fresh rebuilds only to refresh the data.

## The historical spread (fresh rebuilds)

The four runs above used **one commit, one pinned data window, one seed**. Their
execution panels agree to `2e-6` relative — float noise in the vendor's
adjustment arithmetic, not revised data, with identical coverage. They produced
cohort CAGRs spanning 17.20% to 23.12%.

LightGBM splits flip on near-ties, the ranking changes, and a different fifteen
names get held. **All four passed every gate.**

Three consequences, and they are not small:

1. **Two-decimal precision is fiction.** Any figure from this pipeline carries
   roughly ±2.5 points of CAGR at one sigma. Every number this project has ever
   quoted implied a precision that does not exist.
2. **Most historical comparisons were noise.** Configuration A beating
   configuration B by two or three points of CAGR says nothing. The search
   described under *"multiplicity"* in the README compared options whose true
   differences sit well below this floor.
3. **Bit-reproducibility is available.** `--replay-snapshot` reproduces a run
   exactly from its recorded inputs; see *Reproducibility — solved* above. The
   spread here applies to *fresh* rebuilds, which draw new vendor noise.

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
# Reproduce this exact baseline from its snapshot (no network):
uv run train-sp500 --replay-snapshot artifacts/baseline \
  --provider hybrid --start 2010-01-01 --end 2026-08-28 \
  --train-end 2018-12-31 --test-start 2019-01-01 --sample-n 10000 \
  --horizon 63 --wf-top-k 15 --seed 42 --no-optuna --skip-earnings \
  --output-model artifacts/replay/model.pkl \
  --wf-scores-path artifacts/replay/wf_scored.parquet \
  --execution-prices-path artifacts/replay/execution_prices.parquet \
  --snapshot-dir artifacts/replay/snapshot

# Or build a new one from fresh data (~30 min, warm cache):
./scripts/rebuild_baseline.sh
uv run python scripts/verify_baseline.py artifacts/baseline
```

Verification does not modify the baseline, and its verdict is a property of the
recorded artifacts rather than of the machine it runs on. A baseline built
before `vendor_absent.json` existed falls back to the live cache and says so.

**Known gap surfaced by the per-session coverage check:** 12 priced departed
names — APC, FB, HCP, INFO, LB, NFX, SBNY, SCG among them — reach the scored
panel on *none* of their member sessions. Several look like further renames
(FB→META, HCP→PEAK, LB→BBWI) that the rename map does not yet cover.

If the survivorship gate reports recoverable names, fill them first — Tiingo's
free tier allows ~50–76 new tickers per window, so this resumes:

```bash
uv run python scripts/recover_delisted.py --passes 8 --wait 3700
```

Expect the headline numbers to land within the spread above, not on them.
