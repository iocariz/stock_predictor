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
* **No engine's alpha clears |t| = 2, in any draw.** Long-short reaches
  +5.90% ± 0.61% at t = +1.54 and is the best-controlled book by a distance —
  beta 0.222, a drawdown a third of the others' — but it is no longer
  statistically distinguishable from them. Pre-committed evaluations reach
  +1.55 and +1.18. Positive tilt, no edge.
* **Any comparison must clear the spread to mean anything**: ±1.4 points of
  CAGR on long-short, ±3.7 on cohort, ±6.8 on rank-hold. Use
  `--replay-snapshot` for code comparisons; a fresh rebuild has a price.

---

## Provenance

| | |
|---|---|
| commit | `91635b0b4448` (clean tree) |
| run id | `20260906T103531Z_1e935e12` |
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
| `equity_prices_long` | `2a95c682575f175b` | 3,492,792 |
| `execution_prices` | `2a95c682575f175b` | 3,492,792 |
| `features_clean` | `d2a612c67ffc657e` | 1,937,884 |
| `labeled` | `21fe2bb3d749c6e3` | 3,492,792 |
| `macro` | `7196119df94c96af` | 4,228 |
| `sector_map` | `1eed8462eeea73d7` | 503 |
| `stints` | `bc11e6d48d9671c1` | 1,247 |
| `benchmark` | `0aeb727c4a9a2be7` | 4,190 |

Output hashes (`manifest["outputs"]`, checked on every verification, both
**recorded at write time** by the run that produced them):

| output | sha256 (16) |
|---|---|
| `wf_scored` | `f4ccf279d6e9bbdf` |
| `execution_prices` | `e267b2851323202a` |

Scored panel: 951,577 rows, 1,924 sessions, 642 tickers.
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
| point-in-time integrity | 0 of 951,577 scored rows outside index membership, on the half-open `[start_date, end_date)` convention production filters with; labels stop exactly at the last labelable session; execution covers every scored row; scored and execution prices agree on 951,577/951,577 cells |
| survivorship | 271 of 347 departed names carry prices **during their membership** (**78.1%**, the measured ceiling); departed names are scored on **99.9%** of the sessions they were members (117,973/118,147) |
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

**51 of 347 departed names are like this** in this panel — priced only outside
the window they were ever members. They were being counted as survivorship
recoveries, which is why coverage was once reported as 91.4% when the truth is
**78.1%**. The count moves between downloads (46 to 51 across rebuilds of
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
| long-short | 14.63% | 0.88 | -14.27% | +0.22 | +6.76% | +1.75 |
| cohort | 24.72% | 0.82 | -46.61% | +1.16 | +6.02% | +1.00 |
| rank-hold | 23.25% | 0.69 | -58.63% | +1.36 | +4.52% | +0.64 |
<!-- pinned-metrics:end -->

SPY over the same window: **17.55%** CAGR, beta 1.00 by construction.

**Borrow.** Those figures charge **0.5%** flat on the short leg, the
`LongShortConfig` default. The pipeline trades **1%**, which is above the
**0.90%** this book's short side measures under the stylised per-name proxy —
so the live book never charges less than the simulation it was validated
against. The difference is inside the ±1.50% draw-to-draw spread on alpha, so it
changes nothing that is claimed here. The sensitivity curve below predates the
disposal and grid fixes and overstates every row by roughly 3 points of alpha;
its shape — that borrow is not what decides this strategy — is unaffected.

The short book is *cheaper* than the universe (0.90% against 1.22%, a 0.74×
concentration) because the model ranks volatility positively and parks the
expensive, hard-to-borrow names in the **long** book, where no borrow is paid.
The opposite was assumed for a long time and was wrong.

Borrow is not what decides this strategy. Alpha holds t +2.58 at 0%, +2.45 at
1%, +2.31 at 2%, and only falls below 2 near **5%** — five times the measured
rate.

Generated by `scripts/pin_baseline_metrics.py`, not typed, and compared against
`expected_metrics.json` by a test.

**Read none of those numbers on their own.** They are one draw. The table below
is the one to quote from.

### Retraction: no engine clears |t| = 2, and the spread was overstated

Every revision of this document since the rebaseline said the long-short book
*"survives its own noise — all four draws clear |t| = 2."* **That is withdrawn.**
It was an artifact of four defects, and correcting them moved the result twice.

| | published | after disposal fixes | after grid fix |
|---|---|---|---|
| long-short alpha | +8.99% ± 1.47% | +7.09% ± 1.50% | **+5.90% ± 0.61%** |
| HAC t | +2.61 | +2.10 | **+1.54** |
| draws clearing t = 2 | 4 of 4 | 1 of 4 | **0 of 4** |

Four defects, in the order they were found:

* **The unpriced-gap counter measured from the wrong origin.** A holding that
  missed one session had its grace period already exhausted and was written off.
* **A written-off short was free money.** Settling at zero writes a long's claim
  off in full — the conservative reading. Applied to a *short* it erases a
  liability for nothing, the maximum possible profit. Sixteen dark shorts on the
  previous artifact were booked that way. Worth ~1.7 points of alpha.
* **The signal and the fill were the same bar.** The backtest ranks on one
  session and fills at the next, precisely so a score cannot trade on the bar it
  was computed from. The live path did both at once.
* **A missing session was closed up before the rolling windows ran.** Features
  were computed on a compacted grid, so `ret_1d` across a one-session gap read
  as a 1-day return while covering two.

### The spread was itself inflated

Fixing the grid did something the other corrections did not: it made the
pipeline **markedly more reproducible**.

| engine | 2 sd on CAGR, before | after | |
|---|---|---|---|
| long-short | ±3.32% | **±1.44%** | 2.3× tighter |
| cohort | ±7.48% | **±3.73%** | 2.0× tighter |
| rank-hold | ±10.69% | **±6.81%** | 1.6× tighter |

Much of what this document attributed to vendor float noise was gaps being
closed up differently between downloads. The panels really do differ by about
`2e-6`, but the compaction amplified that into several points of CAGR. The
noise floor is real and it was roughly twice as wide as it needed to be.

That also settles an attribution made while the fixes were landing. Rank-hold
fell 12.7 points between artifacts, which sat outside its *old* ±10.69% band.
Against the corrected distribution (22.79% ± 3.41%) the previous 35.94% sits
**3.9 standard deviations out**, so the fall is attributable to the fixes rather
than to the draw — but that could not be said with confidence until the band was
re-measured on the corrected code.

### What changed shape

* **Long-short no longer clears the line in any draw**: +1.37, +1.47, +1.58,
  +1.75. It is still the best-controlled book — beta 0.222 ± 0.010 and a
  drawdown a third of either long-only engine's — and its alpha is no longer
  distinguishable from the others'.
* **Cohort alpha stopped changing sign.** It was +5.00%, +2.93%, −1.68%, −0.83%
  on the old code; it is now +4.34% … +8.05%, consistently positive at t = +1.03.
  Still not significant, but no longer sign-unstable.
* **Rank-hold collapsed**, from +11.82% alpha to +3.97% at t = +0.51.

The three engines are now much closer to each other than this document has ever
shown them, and none of them is significant.

### The spread, measured

Four fresh rebuilds — **one commit, one pinned window, one seed**, differing
only in when the download ran. The panels agree to vendor float noise; LightGBM
splits flip on near-ties, a different fifteen names get held, and the curve
lands somewhere else.

| engine | CAGR (n=4) | Sharpe | alpha/yr | HAC t | range of t |
|---|---|---|---|---|---|
| long-short | 13.65% ± 0.72% | 0.80 ± 0.06 | +5.90% ± 0.61% | +1.54 ± 0.16 | +1.37 … +1.75 |
| cohort | 25.17% ± 1.86% | 0.82 ± 0.04 | +6.33% ± 1.56% | +1.03 ± 0.20 | +0.78 … +1.25 |
| rank-hold | 22.79% ± 3.41% | 0.67 ± 0.07 | +3.97% ± 2.91% | +0.51 ± 0.33 | +0.12 … +0.88 |

`uv run python scripts/baseline_spread.py artifacts/baseline artifacts/baseline_v3 artifacts/baseline_v4 artifacts/baseline_v5`

**What a difference has to clear before it means anything (2 sd on CAGR):**

| engine | band |
|---|---|
| long-short | **±1.44%** |
| cohort | **±7.48%** |
| rank-hold | **±10.69%** |

Most improvements this project has ever quoted sit inside one of those bands.
Use `--replay-snapshot` to compare code changes; a fresh rebuild now has a
measured price.

### Only one engine's result survives its own noise

**Long-short.** *No* draw clears |t| = 2 — +1.37, +1.47, +1.58, +1.75, mean
+1.54. It is still the best-controlled book: beta 0.222 ± 0.010, a drawdown a
third of either long-only engine's, and by far the tightest CAGR band (±0.72%
against ±1.86% and ±3.41%). What it is not, and this document said it was, is
significant. Four independent downloads agree on the statistic the
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

The holdout is measured as a **continuation**: the engine runs once over the
whole panel and the holdout window is read out of the running NAV, so the book
arrives at the split holding what it held, on the calendar it was already on.
Re-running the engine over a truncated panel — which is what this did at first —
restarts the rebalance schedule and begins flat, which measures a fresh
strategy launched on the holdout's first session rather than the one under test.

Run on **two independent artifacts**, because the spread showed that a single
draw cannot carry a conclusion:

| artifact | split | committed configuration | holdout alpha | HAC t | rank |
|---|---|---|---|---|---|
| current | 2023-01-01 | decile 0.10, 1.0x, 63d | +6.08% | **+1.55** | 3 / 18 |
| current | 2022-01-01 | decile 0.10, 1.0x, 63d | +3.89% | **+1.18** | 5 / 18 |
| superseded | 2023-01-01 | decile 0.05, 1.0x, 21d | +5.87% | +1.04 | 13 / 18 |
| superseded | 2022-01-01 | decile 0.10, 1.0x, 63d | +4.17% | +1.45 | 8 / 18 |

The two superseded rows predate the grid fix and are kept only to show the
procedure behaves the same way across artifacts.

Two things improved here even as the level fell. The procedure now commits to
the **same configuration at both splits** — decile 0.10, 1.0x, 63d — where it
previously picked a different winner each time, and the committed configuration
ranks 3rd and 5th of 18 rather than mid-pack. The selection is more stable on
the corrected panel. It is selecting more consistently among options that are
all indistinguishable from zero.

**The full-period t does not survive, and neither figure was ever close to 2.**
On the promoted artifact a pre-committed configuration reaches **+1.55** and
**+1.18**, against a full-period figure of +1.75. With the full period itself now
below the line, the holdout is no longer the binding constraint — the book does
not clear conventional significance even before pre-registration is applied.

**The search carries no information.** The committed configuration lands
mid-pack every time — 13th, 7th, 9th and 5th of 18. Choosing on the first
window tells you nothing about which configuration does well on the second.
Worse, **the winner is not even stable**: the two artifacts commit to three
different configurations across four runs (0.05/21d, 0.10/63d, 0.20/63d twice).
The selection procedure is picking noise, and that is now demonstrated rather
than suspected.

**What replicates across both artifacts and both splits (4 of 4):**

* **Rebalance frequency orders the grid monotonically**, always the same way.

  | rebalance | current, 2023 | current, 2022 | superseded, 2023 | superseded, 2022 |
  |---|---|---|---|---|
  | 21d | +0.40 … +1.23 | +0.24 … +1.29 | −0.39 … +1.04 | −0.19 … +1.45 |
  | 63d | +0.90 … +2.12 | +0.69 … +1.54 | +1.39 … +1.92 | +1.35 … +2.20 |
  | 126d | +0.92 … +1.48 | +0.32 … +0.50 | +1.56 … +2.48 | +0.57 … +1.66 |

  21-day is weakest in all four; longer holds are stronger. This is the most
  durable thing in the whole document — the only claim here that has survived
  an independent redraw.
* **Decile 0.20 is the worst of the three** in all four, so the effect lives in
  tighter selections.

*Corrections, from re-running on the new artifact.* Two claims the previous
revision made do **not** replicate, and both were stated too strongly from one
draw:

* *"21-day rebalancing destroys the effect."* It does not. It is consistently
  the weakest bucket, but on the current artifact it reaches +1.66 and +2.09.
  The ordering replicates; the word "destroys" was a description of one draw.
* *"Four configurations — all 21-day — go negative at the 2022 split."* On the
  current artifact **all 18 are positive at both splits** (grid median t +2.17
  and +2.06, with 10 of 18 above +2 each time). The 14-of-18 I reported was
  itself a draw, and so was the correction I built on it.

The grid also looks *better* on this artifact than the last — median holdout t
above +2 at both splits, against +1.92 and +1.02 before. That is the same
draw-to-draw variation seen everywhere else in this document, and it cuts both
ways: it is not evidence the book improved.

So the defensible claim is a **positive tilt that lives at 63-day holding
periods and tighter deciles, and that does not reach |t| = 2 in any draw or on
any split** — not an edge.

**This does not make it a strategy.** Three things stand against reading the
full-period figure as settled, and the holdout above confirms the first two:

1. **Multiplicity — demonstrated, not suspected.** Pre-committed evaluations
   reach +1.55 and +1.18 against a full-period +1.75, so the full period itself
   is now below the line and multiplicity is no longer the binding constraint. The full-period figure is the
   best of a correlated search, quoted as though it were a test.
2. **Draw dependence — also demonstrated.** Every number here moves when the
   data is re-downloaded from the same window; see [the spread](#the-spread-measured).
3. **31 rebalances, and it is not wired to anything live.** Seven and a half
   years at a 63-day horizon is a small number of independent decisions, and
   the live path trades the cohort engine — the one whose alpha changes sign
   between draws.

Against SPY, with CAPM on excess returns and Newey–West standard errors. These
are the pinned figures from [the results table](#results), measured on the
recorded benchmark rather than a fresh download:

| engine | beta | alpha/yr (this draw) | HAC t | alpha across 4 draws |
|---|---|---|---|---|
| cohort | **+1.16** | +6.02% | **+1.00** | +6.33% ± 1.56%, t +0.78…+1.25 |
| rank-hold | **+1.36** | +4.52% | **+0.64** | +3.97% ± 2.91%, t +0.12…+0.88 |

**Neither engine shows evidence of skill.** Cohort alpha is +6.33% ± 1.56% at
t = +1.03 — consistently positive now, and still not significant; rank-hold's is
+3.97% ± 2.91% at t = +0.51. Both carry beta well above 1 (1.16 and 1.36), so
most of what they beat the index by is leverage, not selection.

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

That artifact is **no longer retained**, so these three rows cannot be
re-derived locally. The mechanism they demonstrate is covered by
`tests/test_replay.py` and by the snapshot-integrity gate, which is what makes
the figures safe to keep as a record rather than as a checkable claim.

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
