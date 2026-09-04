# Baseline

Every performance figure this project published before 2026-08-28 was measured
on artifacts whose provenance was not recorded, and several were wrong for
reasons unrelated to the strategy — a look-ahead in cohort construction, a
scored panel pricing its own fills, an alpha that was mostly the risk-free
rate. **Those numbers are obsolete.** This file records what replaces them,
what verified it, and how much of it is trustworthy.

The short version, after four independent rebuilds of one commit and one
pinned window:

* **Neither long-only engine shows alpha distinguishable from zero**, and
  neither is stable enough to quote from a single run. Cohort alpha changes
  sign between draws; rank-hold's CAGR carries a ±10.7-point band.
* **The long-short book is the one result that survives its own noise** — all
  four draws clear |t| = 2 — but it still fails the locked holdout, so it is a
  measured tilt rather than an established edge.
* **Any comparison must clear the spread to mean anything**: ±3.3 points of
  CAGR on long-short, ±7.5 on cohort, ±10.7 on rank-hold. Use
  `--replay-snapshot` for code comparisons; a fresh rebuild has a price.

---

## Provenance

| | |
|---|---|
| commit | `b8ab695c173d` (clean tree) |
| run id | `20260904T051107Z_1a6a5664` |
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
| `equity_prices_long` | `325b0a4c27b5ec79` | 3,492,792 |
| `execution_prices` | `325b0a4c27b5ec79` | 3,492,792 |
| `features_clean` | `66ea526ef5bf3147` | 1,940,969 |
| `labeled` | `f1e2281fd9dcbb13` | 2,770,323 |
| `macro` | `9e16de179cc5ad22` | 4,228 |
| `sector_map` | `408da130fa1f0985` | 503 |
| `stints` | `27dbf956b96a35ae` | 1,247 |
| `benchmark` | `09e5960f468c526b` | 4,190 |

Output hashes (`manifest["outputs"]`, checked on every verification, both
**recorded at write time** by the run that produced them):

| output | sha256 (16) |
|---|---|
| `wf_scored` | `2494cf089ec17042` |
| `execution_prices` | `f593a935eb2dc413` |

Scored panel: 952,329 rows, 1,924 sessions, 643 tickers.
Execution panel: 4,188 sessions × 834 tickers.

---

## Verification

All seventeen gates pass. Each is a hard failure, not a warning. Verification is
**read-only and self-contained**: it reads membership from
`snapshot/stints.parquet`, the benchmark from `snapshot/benchmark.parquet`, and
the vendor-absent set from `vendor_absent.json` recorded at build time — never
from live state — and writes nothing into the baseline. Pass `--report` to emit
the survivorship residual elsewhere.

| gate | result |
|---|---|
| snapshot integrity | all eight input artifacts recomputed and matching their recorded sha256, all written by this run |
| output integrity | `wf_scored.parquet` and `execution_prices.parquet` recomputed against recorded hashes |
| execution derivation | the wide panel reproduced exactly from the hashed long snapshot: 4,188 sessions × 764 priced tickers |
| ticker renames | 15 checked for successor coverage; **0 of 15 testable for concurrent trading** — canonicalisation removes the predecessor's own symbol from the panel, so the one real falsifier cannot run. Coverage shows a successor prices the predecessor's membership; it cannot show they are the same issuer, and each entry's recorded note remains the warrant |
| point-in-time integrity | 0 of 952,329 scored rows outside index membership, on the half-open `[start_date, end_date)` convention production filters with; labels stop exactly at the last labelable session; execution covers every scored row; scored and execution prices agree on 952,329/952,329 cells |
| survivorship | 272 of 347 departed names carry prices **during their membership** (**78.4%**, the measured ceiling); departed names are scored on **99.9%** of the sessions they were members (118,721/118,896) |
| recorded benchmark | SPY, 1,924 sessions, from the snapshot — so beta, alpha and the HAC t are checkable offline |
| pinned metrics | every published CAGR, Sharpe, drawdown, beta, alpha and HAC t recomputed against `expected_metrics.json` |
| accounting (cohort) | NAV reconciles with the trade ledger to `3.37e-16`; cash + holdings = NAV on all 1,924 sessions (residual `0.00e+00`) |
| accounting (rank-hold) | reconciles to `7.83e-16`, with 15 open positions and −27,937.51 unrealized; per-session residual `0.00e+00` |
| accounting (long-short) | cash + holdings = NAV on all 1,924 sessions (residual `0.00e+00`). No cohort ledger to close a terminal identity against, but `specs.md:414`'s per-session identity applies and is checked |
| fills (all three engines) | zero stale fills; every refusal disposed under a stated policy |
| deterministic backtest (all three) | identical NAV hashes on repeat |

Nothing here is sealed or backfilled. The previous baseline carried both
output hashes as `sealed-after-the-fact` and its benchmark as
`recorded-after-the-fact`, because the pipeline did not yet capture them; this
one records all three during the run that produced them. The seal and the
backfill scripts remain for older artifacts and are no longer used here.

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

### Reused ticker symbols

A ticker is not a company. When one is acquired or renamed its symbol is
retired and the exchange may reassign it, so the panel picks up a *different
issuer's* prices under a departed member's name:

| ticker | company, when it left | prices in the raw panel |
|---|---|---|
| APC | Anadarko, acquired 2019-08 | from 2026-02-12 |
| FB | Facebook, renamed META 2022-06 | from 2025-06-26 |
| Q | Qwest, left the index 2011-04 | from 2025-10-27 (14.6 years later) |
| SNDK | SanDisk, acquired 2016 | from 2025-02-13 (a re-IPO, new entity) |

**50 of 347 departed names are like this** in this panel — priced only outside
the window they were ever members. They were being counted as survivorship
recoveries, which is why coverage was once reported as 91.4% when the truth is
**78.4%**. The count moves between downloads (46, 49 and 50 across rebuilds of
the same window) because symbols keep being reassigned; it is recorded per run
in `manifest["recycled_symbols"]` rather than assumed constant.

The point-in-time filter kept them out of the scored panel, which is why none
reached a trade and why they surfaced as zero-coverage rather than as bad
fills. That filter was doing all the work. They still sat in the execution
panel, where `_resolve_leg_exit` walks forward for the next real quote when an
exit cannot fill — the write-off grace period ends that walk first under the
default policy, but `fallback="hold"` does not, and the walk would sell a 2019
holding at an unrelated 2026 company's price.

`drop_recycled_prices` blanks a departed ticker's prices that resume after a
long dead period, and the build records which tickers it altered. A company
*demoted* out of the index keeps trading with continuous prices across the
boundary — those are the same company and are exactly what prices an exit
after a name drops out, so they are untouched. A reused symbol has years of
nothing first.

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

Out-of-sample 2019-01-02 → 2026-08-27, measured on the artifacts in
`artifacts/baseline` and pinned to them in `expected_metrics.json`. Every
figure below is recomputed by `verify_baseline.py` on each run and the
verification fails if any of them moves.

<!-- pinned-metrics:start -->
| engine | CAGR | Sharpe | max drawdown | beta | alpha/yr | HAC t |
|---|---|---|---|---|---|---|
| long-short | 18.04% | 1.12 | -13.03% | +0.24 | +9.53% | +2.56 |
| cohort | 24.07% | 0.79 | -45.94% | +1.21 | +5.00% | +0.89 |
| rank-hold | 35.94% | 0.95 | -51.90% | +1.40 | +13.92% | +1.78 |
<!-- pinned-metrics:end -->

SPY over the same window: **17.55%** CAGR, beta 1.00 by construction.

Generated by `scripts/pin_baseline_metrics.py`, not typed, and compared against
`expected_metrics.json` by a test.

**Read none of those numbers on their own.** They are one draw. The table below
is the one to quote from.

### The spread, measured

Four fresh rebuilds — **one commit, one pinned window, one seed**, differing
only in when the download ran. The panels agree to vendor float noise; LightGBM
splits flip on near-ties, a different fifteen names get held, and the curve
lands somewhere else.

| engine | CAGR (n=4) | Sharpe | alpha/yr | HAC t | range of t |
|---|---|---|---|---|---|
| **long-short** | **17.22% ± 1.66%** | **1.07 ± 0.12** | **+8.99% ± 1.47%** | **+2.61 ± 0.22** | **+2.44 … +2.94** |
| cohort | 19.72% ± 3.74% | 0.66 ± 0.11 | +1.36% ± 3.15% | +0.23 ± 0.55 | −0.29 … +0.89 |
| rank-hold | 32.95% ± 5.35% | 0.88 ± 0.11 | +11.82% ± 3.99% | +1.41 ± 0.43 | +0.80 … +1.78 |

`uv run python scripts/baseline_spread.py artifacts/baseline artifacts/baseline_v3 artifacts/baseline_v4 artifacts/baseline_v5`

**What a difference has to clear before it means anything (2 sd on CAGR):**

| engine | band |
|---|---|
| long-short | ±3.32% |
| cohort | **±7.48%** |
| rank-hold | **±10.69%** |

Most improvements this project has ever quoted sit inside one of those bands.
Use `--replay-snapshot` to compare code changes; a fresh rebuild now has a
measured price.

### Only one engine's result survives its own noise

**Long-short.** Every one of the four draws clears |t| = 2 — +2.44, +2.56,
+2.61, +2.94 — on a beta that barely moves (0.223 ± 0.011) and the tightest
CAGR band of the three. Four independent downloads agree on the statistic the
conclusion rests on. That does *not* make it an edge; see the locked holdout
below, which it still fails.

**Cohort alpha changes sign between draws**: +5.00%, +2.93%, −1.68%, −0.83%.
The distribution is centred near zero (+1.36% ± 3.15%, t = +0.23). Any single
figure — including the +4.72% and the +2.36% this document has previously
published — is a draw, not a measurement.

**Rank-hold cannot be quoted at all.** Its alpha spans +5.87% to +14.33% across
draws; the entire reported effect fits inside its own noise. Both the 26.41%
CAGR this document once carried and the 18.83% published as its *correction*
are draws from a distribution with a ±10.69% band.

*Correction.* The previous revision reported rank-hold at 18.83% CAGR and
+0.45% alpha, and treated the fall from 26.41% as the effect of removing
contaminated prices. Four draws show 25.04%, 34.37%, 35.94% and 36.44%: the
18.83% was an outlier, and the confident story told about *why* it fell was
built on a single draw. The direction of the recycled-symbol correction may
still be right; the magnitude was never measurable this way.

### The locked holdout

`scripts/locked_holdout.py` answers the multiplicity objection directly: search
the grid on an early window only, commit to the single winner, evaluate it
**once** on the later window. Selection by Sharpe, deliberately not by the alpha
t-statistic that gets reported.

**These figures were measured on the previous artifact** (`20260831T085207Z`,
now `artifacts/baseline_v1`) and have not been re-run on this one. They are
kept because what they establish is about the *procedure* — whether selecting
on one window generalises to the next — which the spread does not overturn. But
they are a single draw like everything else, and the long-short CAGR band of
±1.66% applies to them too.

The holdout is measured as a **continuation**: the engine runs once over the
whole panel and the holdout window is read out of the running NAV, so the book
arrives at the split holding what it held, on the calendar it was already on.
Re-running the engine over a truncated panel — which is what this did at first —
restarts the rebalance schedule and begins flat, which measures a fresh
strategy launched on the holdout's first session rather than the one under test.

| split | committed configuration | holdout alpha | HAC t | rank on holdout |
|---|---|---|---|---|
| 2023-01-01 | decile 0.20, 1.0x, 63d | +6.04% | **+1.95** | 9 / 18 |
| 2022-01-01 | decile 0.20, 1.0x, 63d | +5.20% | **+1.87** | 5 / 18 |

**The full-period t = +2.60 does not survive.** A pre-committed configuration
reaches +1.95 and +1.87 — neither split clears |t| = 2, and both fall well
short of the figure the full-period search produced.

*Correction.* An earlier version of this table reported **+1.78 / +2.26**, from
a holdout that restarted the strategy instead of continuing it. The 2022 split
was the one that moved: a book relaunched flat on 2022-01-04 skipped the
drawdown a continuing book carried across the boundary, which is what lifted it
above +2. The corrected pair is more consistent than the pair it replaces, and
the conclusion is the same one, stated more firmly — it no longer depends on
which split you pick.

**The search carried no information.** The committed configuration lands
mid-pack on the holdout both times — 9th and 5th of 18. Choosing on the first
window told you nothing about which configuration would do well on the second,
which is what fitting noise looks like from the outside. Note also that the
honest procedure picks decile **0.20**, not the 0.10 this project has always
quoted; the historically quoted configuration reaches t = +2.53 on the
2023 holdout, but we only know that because we went and looked.

**Two things do survive**, and they are worth more than the headline was:

* The grid is **structured, not random**, and this is the finding that holds up
  best. Sorted by holdout alpha t it separates almost perfectly by rebalance
  frequency, and the same way under both splits:

  | rebalance | t at the 2023 split | t at the 2022 split |
  |---|---|---|
  | 21d | +0.31 … +0.96 | −0.15 … +0.55 |
  | 63d | +1.89 … +2.74 | +1.81 … +2.15 |
  | 126d | +1.59 … +2.94 | +0.90 … +1.05 |

  **21-day rebalancing destroys the effect under both splits; 63-day carries it
  under both.** 126-day is strong at one split and weak at the other, so only
  63-day is consistent. Tighter deciles (0.05, 0.10) beat 0.20 in both, though
  the margin at the 2022 split is small. A pattern that repeats across an
  independent cut is harder to explain as noise than any single maximum.
* The tilt is **mostly positive but not uniformly so**. All 18 configurations
  have positive holdout alpha at the 2023 split (median t +1.92, 8 of 18 above
  +2); at the 2022 split only 14 of 18 are positive and the median is +1.02.

  *Correction.* The earlier version of this section claimed all 18 were positive
  under **both** splits. That was true only of the restarted holdout; measured
  as a continuation, four configurations — all of them 21-day — go negative at
  the 2022 split.

So the defensible claim is a **positive but statistically unestablished tilt
that lives at longer holding periods and tighter deciles** — not an edge of
+7.98% at t = +2.60.

**This does not make it a strategy.** Three things stand against reading
t = +2.60 as settled, and the holdout above confirms the first of them:

1. **Multiplicity — demonstrated, not merely suspected.** The locked holdout
   above shows a pre-committed configuration reaching only +1.95/+1.87, and
   the selection ranking mid-pack. The +2.60 is the best of a correlated
   search, quoted as though it were a test.
2. **31 rebalances.** Seven and a half years at a 63-day horizon is a small
   number of independent decisions.
3. **It is not wired to anything live.** The live path trades the cohort
   engine, which is the one with no measurable alpha.

Against SPY, with CAPM on excess returns and Newey–West standard errors. These
are the pinned figures from [the results table](#results), measured on the
recorded benchmark rather than a fresh download:

| engine | beta | alpha/yr (this draw) | HAC t | alpha across 4 draws |
|---|---|---|---|---|
| cohort | **+1.21** | +5.00% | **+0.89** | +1.36% ± 3.15%, sign unstable |
| rank-hold | **+1.40** | +13.92% | **+1.78** | +11.82% ± 3.99%, t +0.80…+1.78 |

**Neither engine shows evidence of skill**, and neither can be quoted from a
single draw. Cohort alpha is centred near zero and changes sign between
rebuilds; rank-hold's spans +5.87% to +14.33% and never clears t = 2. Both
carry beta well above 1 (1.21 and 1.40), so most of what they beat the index by
is leverage, not selection.

Three corrections were previously recorded here as having moved these figures:
ticker renames taking cohort alpha from +1.00% to −2.47%, the cohort expiry
boundary taking it back to +2.36%, and recycled-symbol removal taking it to
+4.72% while cutting rank-hold from +6.80% to +0.45%.

**Those attributions do not survive the spread.** Each was a single draw
compared against another single draw, and cohort alpha varies by ±3.15% across
draws of one commit — wider than two of the three "effects". The *fixes* are
right on their own terms, each with a test and a mechanism; what cannot be
supported is the claim that any of them moved alpha by a specific amount. The
one durable finding is the one that never depended on a difference: neither
long-only engine's alpha is distinguishable from zero, in any draw.

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

Two independent replays produced **byte-identical** `wf_scored.parquet` and
`execution_prices.parquet`, and both reproduced the original run's backtest to
four decimals.

| | cohort CAGR | max DD | rank-hold CAGR | max DD |
|---|---|---|---|---|
| original run | 18.5206% | −44.6400% | 22.2642% | −57.9037% |
| replay A | 18.5206% | −44.6400% | 22.2642% | −57.9037% |
| replay B | 18.5206% | −44.6400% | 22.2642% | −57.9037% |

**These figures are from run `20260830T204011Z_f814aa3d`, not the baseline on
disk.** They are kept because what they demonstrate — that a replay reproduces
the run it came from, exactly — is a property of the mechanism and does not
depend on which artifact it was shown on. They are *not* this baseline's
performance; those are [in the results table](#results), which is pinned and
verified. Replay has not been re-run against the current artifact.

Replay reproduces the *run*, not merely itself. A comparison between two
measurements from one snapshot is now a comparison of the change, not of two
draws — which is what makes any of the numbers below worth arguing about.

**Fresh rebuilds still differ**, and that is a separate thing: re-downloading
draws new vendor float noise and lands somewhere in the spread below. Use
replay to compare code changes; use fresh rebuilds only to refresh the data.

## The historical spread (fresh rebuilds)

> Superseded by [*The spread, measured*](#the-spread-measured), which repeats
> this exercise on the current code and reports per-engine bands. The passage
> below is the first time the problem was quantified and is kept for the
> record; its ±2.5-point figure was measured on the cohort engine alone and
> understates two of the three engines.

The four runs above used **one commit, one pinned data window, one seed**. Their
execution panels agree to `2e-6` relative — float noise in the vendor's
adjustment arithmetic, not revised data, with identical coverage. They produced
cohort CAGRs spanning 17.20% to 23.12%.

LightGBM splits flip on near-ties, the ranking changes, and a different fifteen
names get held. **All four passed every gate.**

Three consequences, and they are not small:

1. **Two-decimal precision is fiction.** Any figure from this pipeline carries
   roughly ±2.5 points of CAGR at one sigma on the cohort engine — and, as the
   current measurement shows, ±5.35 points on rank-hold. Every number this
   project has ever quoted implied a precision that does not exist.
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

The long-short book is now covered: `backtest-sp500 --mode long-short`, gated
alongside the other two, and measured above. Its previously quoted figures are
still retired — they were measured on the old panels and on raw-return CAPM.

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

Those 12 zero-coverage names turned out not to be renames but **reused
symbols** — see above. One remains: `SCG` (SCANA, acquired by Dominion 2019)
has a single eligible session and no price on it.

If the survivorship gate reports recoverable names, fill them first — Tiingo's
free tier allows ~50–76 new tickers per window, so this resumes:

```bash
uv run python scripts/recover_delisted.py --passes 8 --wait 3700
```

Expect the headline numbers to land within the spread above, not on them.
