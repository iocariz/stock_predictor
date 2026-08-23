# Architecture specification

Status: proposed  
Audience: coding agents and maintainers  
Scope: training, evaluation, backtesting, model promotion, and daily inference  
Normative language: **MUST**, **MUST NOT**, **SHOULD**, and **MAY** have their usual requirements meaning.

## 1. Purpose

Evolve `stock-predictor` into a system in which the same point-in-time data rules, feature definitions, portfolio policy, and cost assumptions are used in research and daily inference.

The architecture must make invalid states difficult to represent and methodological drift visible. A successful refactor is not merely a directory reorganization: it must close the correctness gaps listed in this document while preserving the useful behavior and command-line interfaces of the current package.

This is an incremental migration specification. Do not perform a big-bang rewrite.

## 2. Required outcomes

The completed architecture MUST provide:

1. An immutable, versioned model bundle containing everything needed to reproduce inference.
2. Explicit and separately validated contracts for membership, prices, features, labels, scores, and execution quotes.
3. A single pure portfolio-policy implementation shared by backtest and live paths.
4. A clear separation between signal generation and trade execution.
5. A backtest that uses an independent execution-price panel rather than prices recovered from scored rows.
6. Date-aware training and evaluation that match a cross-sectional ranking strategy.
7. Separate candidate evaluation, production refitting, and model promotion stages.
8. Thin CLIs, scripts, workflows, and notebooks that call package APIs instead of reimplementing methodology.
9. Deterministic runs with manifests, schema versions, hashes, and actionable validation failures.
10. Backward-compatible public commands during migration, with explicit warnings for legacy artifacts.

## 3. Non-goals

This work MUST NOT:

- Introduce a brokerage integration or place real external orders.
- Claim profitability or statistical significance.
- Silently change the default strategy, universe, holding period, or transaction costs.
- Replace LightGBM solely for architectural reasons.
- Add distributed infrastructure, a database, or a service layer without a demonstrated need.
- Treat type annotations as a substitute for runtime dataframe validation.
- Preserve a known-invalid behavior merely to keep old backtest numbers unchanged.

## 4. Existing foundations to preserve

The repository already contains useful foundations. The implementation MUST reuse or migrate them rather than create competing copies:

- `execution.py`: pure selection, weighting, sizing, cost, exit, and deterministic cohort-ID primitives.
- `universe.py`: deterministic universe selection, recorded live-universe resolution, coverage checks, and universe hashing.
- `freshness.py`: model and market-data freshness policies.
- `repro.py`: run identifiers, file hashing, manifests, and snapshots.
- `deploy.py`: explicit candidate validation, archival, and promotion.
- `execution_calendar.py`: exchange-session calculations.
- Existing point-in-time feature-staging, purge, live-safety, and parity tests.

There MUST be one authoritative implementation of each rule. Compatibility modules MAY re-export moved functions but MUST NOT fork their logic.

## 5. Architectural principles

### 5.1 Dependency direction

Dependencies MUST flow inward:

```text
CLI / scripts / notebooks / workflows
                  |
                  v
       application orchestration
          /       |        \
         v        v         v
       data    modeling   execution adapters
         \        |         /
          v       v        v
        contracts + pure portfolio policy
```

Domain contracts and pure policy code MUST NOT import CLI modules, filesystem orchestration, network providers, plotting code, or mutable portfolio persistence.

### 5.2 Pure decisions, impure adapters

The following operations MUST be deterministic and side-effect free for fixed inputs:

- Universe resolution from an explicit membership table and `UniverseSpec`.
- Feature and label construction.
- Cross-sectional scoring preparation.
- Portfolio selection, exits, weighting, and target sizing.
- Statistical metric calculations.

Network downloads, caching, filesystem writes, state persistence, plotting, and broker-specific translation MUST remain at adapter or workflow boundaries.

### 5.3 Fail closed

Live entry decisions MUST fail closed when required inputs are missing, stale, incompatible, or ambiguous. Exits and risk-reducing actions SHOULD remain possible when new entries are blocked.

Warnings are permitted for research-only limitations. A condition that could change live orders MUST be either resolved explicitly or block new entries.

### 5.4 Point-in-time semantics

Every table used in training or simulation MUST have an explicit `as_of` meaning. Data availability time, market observation time, membership time, label end time, signal time, and execution time MUST NOT be conflated.

## 6. Target capability boundaries

Exact filenames MAY vary when a move would create unnecessary churn, but these boundaries are mandatory:

```text
src/stock_predictor/
  contracts.py                 Typed records and dataframe validators
  data/
    universe.py                PIT membership and universe resolution
    prices.py                  Canonical adjusted-price ingestion
    cache.py                   Range- and provider-aware caching
    dataset.py                 Dataset assembly and manifests
  features/
    pipeline.py                Time-series then PIT then cross-sectional stages
    labels.py                  Forward labels and label-end dates
  modeling/
    train.py                   Fitting and purged validation
    score.py                   Model-family-aware scoring
    bundle.py                  ModelBundle read/write/validation
  strategy/
    policy.py                  Pure shared selection and exit policy
    costs.py                   Shared fill and cost assumptions
  simulation/
    backtest.py                Cash ledger and simulated execution adapter
  live/
    portfolio.py               State and live order adapter
    freshness.py               Entry-blocking safety checks
  evaluation/
    cross_sectional.py         Per-date ranking metrics
    portfolio.py               NAV and benchmark-relative metrics
    statistics.py              HAC and block-bootstrap utilities
  workflows/
    research.py                Candidate evaluation workflow
    production.py              Refit, validation, and promotion workflow
    predict.py                 Daily inference workflow
```

The existing flat modules MAY remain as facades during migration. Public imports and console commands SHOULD keep working until a separately documented major-version change.

## 7. Canonical contracts

All validators MUST return a normalized object or raise a specific validation exception. They MUST NOT silently drop invalid rows unless the calling policy explicitly requests it and records the count.

### 7.1 Membership panel

Required columns:

| Column | Meaning |
|---|---|
| `ticker` | Canonical symbol used by the selected provider |
| `start_date` | First inclusive index-membership session |
| `end_date` | Last inclusive session, nullable for a current member |

Requirements:

- Stints MUST be unique and non-overlapping per ticker after normalization.
- Membership on a date MUST be computed from the stints, never from the current constituent list.
- Symbol mappings and corporate-action aliases MUST be recorded, not applied invisibly.
- The membership source, retrieval time, and content hash MUST be written to the run manifest.

### 7.2 Price panel

Canonical long-form fields:

| Column | Required | Meaning |
|---|---:|---|
| `date` | yes | Exchange session represented by the bar |
| `ticker` | yes | Canonical symbol |
| `adj_close` | yes | Split/dividend-adjusted close under a declared convention |
| `volume` | yes | Volume under a declared raw/adjusted convention |
| `provider` | yes | Source of this observation |
| `observed_at` | for live data | Time the value was retrieved or published |
| `tradability` | recommended | `tradeable`, `missing`, `halted`, `delisted`, or `unknown` |

Requirements:

- `(date, ticker)` MUST be unique.
- Prices MUST be finite and positive when used for a fill.
- Provider adjustment conventions MUST be normalized before providers are combined.
- A training panel MUST validate recent and terminal coverage, not only the existence of any value for a ticker.
- A missing current quote MUST remain missing. It MUST NOT be turned into an executable quote by forward fill.
- Historical delisting treatment MUST use explicit evidence or an explicit conservative fallback. A ticker's last available row alone MUST NOT prove delisting.

### 7.3 Feature panel

Required identity fields are `date` and `ticker`; feature columns are declared by `feature_schema_version`.

The feature pipeline MUST preserve this order:

1. Compute time-series features on each ticker's full contiguous price history.
2. Apply point-in-time index membership.
3. Compute cross-sectional and sector-relative features only over eligible members on that date.
4. Join date-level macro and calendar features using information available as of the row's timestamp.
5. Join optional fundamentals using filing/publication availability dates.

Additional requirements:

- `(date, ticker)` MUST be unique and sorted deterministically before model fitting.
- Infinite values MUST be rejected or converted under an explicit recorded rule.
- Row eligibility MUST be based on a named policy. It MUST NOT depend on an accidental blanket `dropna()`.
- Training and inference MUST call the same feature functions and schema validator.
- Inference rows MUST have a current valid `adj_close` and the minimum price history required by the schema. Calendar-only values MUST NOT make an unpriced row scoreable.

### 7.4 Label panel

Every labeled row MUST include:

- `date`
- `ticker`
- `label_value`
- `label_end_date`
- `label_spec_version`

For rank objectives, the per-date grade MAY be stored as an additional column.

Requirements:

- A training split may contain a row only when its `label_end_date` is strictly before the first validation or test observation governed by the split.
- Purging MUST be based on dates and label windows, never raw row counts.
- Labels MUST be computed from full price histories before the PIT membership filter is applied to the labeled row.
- Rows without an observable terminal label MUST be excluded or handled by an explicitly named delisting policy. They MUST NOT be assigned the worst rank automatically.

### 7.5 Scored panel

Required columns:

- `signal_date`
- `ticker`
- `score`
- `score_kind`: `probability` or `rank_score`
- `model_id`
- `universe_hash`

`adj_close` MAY be included for diagnostics, but it is not authoritative execution data.

Requirements:

- A scored panel represents a decision signal, not a trade ledger or price source.
- Score semantics MUST be carried with the data. Rank scores MUST NOT be labeled or formatted as probabilities.
- Full cross-sections MUST reach the portfolio policy. Callers MUST NOT pre-truncate to a multiple of `top_n`.
- Duplicate ticker scores on one signal date MUST fail validation.

### 7.6 Execution quotes

An execution quote MUST contain ticker, session or timestamp, price, source, and freshness/tradability status.

- Simulated fills MUST use the exact configured execution session.
- Live fills MUST use a quote fresh enough for the configured policy.
- Forward fill MAY be used for valuation only when the stale age is retained and surfaced. It MUST NOT be used for entry or exit fills.
- Missing exits, halts, and delistings MUST follow a documented configurable policy and appear in the result diagnostics.

## 8. Model bundle

### 8.1 Storage

New training runs MUST write a versioned model bundle rather than an unstructured pickle plus an optional sidecar. The bundle MAY be a directory or a single archive, but promotion MUST be atomic as one logical unit.

Pickle content MUST be treated as trusted-local-only. Loaders MUST NOT imply that untrusted pickle files are safe.

### 8.2 Required metadata

Every new bundle MUST contain:

```text
bundle_schema_version
model_id
created_at_utc
code_revision and dirty flag
objective and score_kind
feature_schema_version and ordered feature list
label_spec_version, label target, and horizon
training_start, training_end, and final label date
validation design and purge length
effective tree count and fitted model family
hyperparameters and all random seeds
exact training-universe tickers and universe_hash
membership-source hash
data-provider and adjustment conventions
dataset/run manifest identifier
production strategy compatibility, including holding horizon
artifact hashes
```

### 8.3 Compatibility validation

Before scoring, the loader MUST verify:

- Supported bundle schema version.
- Exact feature names and order.
- Compatible objective and weighting mode.
- Compatible holding horizon and strategy mode.
- Exact recorded training universe, intersected with current PIT membership.
- Model and data freshness policy.
- Model object can score a small schema-valid sample.

Tickers added to the index after training MUST NOT silently join a capped model's cross-section. The policy for uncapped universes MUST still be explicit in metadata.

Legacy models MAY load through a compatibility reader. Missing metadata MUST produce explicit warnings and MUST block unattended promotion unless `force` is deliberately supplied.

## 9. Dataset construction and caching

### 9.1 Dataset builder

One application-level dataset builder MUST serve training, walk-forward scoring, backtesting preparation, and live feature construction.

It MUST accept explicit specifications rather than read global CLI state:

```python
DatasetSpec(
    start,
    end,
    universe_spec,
    provider_spec,
    feature_schema_version,
    label_spec=None,
)
```

It MUST return separately named outputs such as membership, raw prices, features, labels, and diagnostics. It MUST NOT return one overloaded dataframe whose role changes by caller.

### 9.2 Cache keys

Provider caches MUST include, at minimum:

- Provider identity and relevant provider configuration.
- Ticker.
- Requested start and end dates.
- Price adjustment convention.
- Data/schema version.

A cache hit MUST prove that the cached range covers the request. A short earlier request MUST NOT satisfy a later longer request. Negative cache entries MUST have an expiry or source-version rule.

### 9.3 Reproducibility

Every material run MUST produce a manifest containing input hashes, normalized configuration, row counts, coverage findings, output hashes, and code revision.

The exact sampled universe list MUST be stored both in the run manifest and the model bundle. Replaying a seed against a changed population is not reproduction.

## 10. Training architecture

### 10.1 Split discipline

- All splits MUST operate on ordered unique signal dates.
- The purge MUST guarantee no training label window crosses the next evaluation boundary.
- Hyperparameter selection, early stopping, candidate evaluation, and final reporting MUST have distinguishable datasets and roles.
- Global statistics used during fitting, including class weights, MUST be computed from each fold's training portion only.

### 10.2 Early stopping

The effective LightGBM tree count MUST use the library's documented `best_iteration_` semantics. Do not add one to a value that is already the number of fitted boosting iterations.

A regression test MUST compare the recorded tree count, the early-stopped booster tree count, and the tree count of the full-data refit.

### 10.3 Research candidate versus production fit

The training workflow MUST produce two conceptually separate results:

1. **Candidate evaluation:** all model and strategy choices are assessed strictly out of sample using a locked evaluation design.
2. **Production refit:** after a candidate is accepted, the chosen specification is fitted on all eligible data through a dynamic `as_of` cutoff.

The production refit MUST NOT overwrite or rewrite the candidate's evaluation results. Its bundle MUST link back to the accepted candidate and evaluation manifest.

Scheduled training MUST use an explicit current cutoff or `as_of` policy; it MUST NOT silently retrain forever through a fixed historical date.

## 11. Portfolio policy and execution

### 11.1 Shared policy

Backtest, paper, and live paths MUST call the same pure functions for:

- Minimum cross-section validation.
- Score thresholding.
- Rank offset and top-N selection.
- Weight construction.
- Rank-decay exits.
- Cost calculation and cash-constrained sizing.
- Deterministic decision or cohort identifiers.

Callers MUST pass the complete scored cross-section. A `rank_offset=10, top_n=5` policy must be able to select ranks 11–15 in every execution mode.

### 11.2 Safety precedence

Entry permission MUST be expressed as a conjunction of safety and scheduling decisions:

```python
may_open = safety_allows_entries and (
    force_schedule or not (repeat_signal or off_schedule)
)
```

`force_schedule` MAY override idempotency or scheduling gates. It MUST NOT override kill switches, stale-data blocks, schema incompatibility, invalid prices, or insufficient cash. A separate, conspicuous emergency override would require its own audit trail and is outside this specification.

### 11.3 Backtest inputs

The backtest API MUST separate signals from prices:

```python
BacktestInputs(
    scores=scored_panel,
    execution_prices=price_panel,
    membership=membership_panel,
    benchmark_prices=benchmark_panel,
)
```

The scored panel MUST NOT be pivoted and forward-filled to create execution prices.

For every requested fill, the result MUST record one of:

- Filled, with raw price, adjusted fill price, slippage, commission, and source date.
- Rejected, with a reason such as missing price, stale price, halted, not a member, insufficient cash, or policy block.

### 11.4 Ledger and valuation

- Cash, holdings, realized P&L, unrealized P&L, commissions, borrow, financing, and slippage MUST reconcile on every session.
- Daily NAV and all metrics derived from it MUST include fees when fees are enabled.
- Integer-share and fractional modes MUST share the same ledger semantics.
- Cash MUST NOT become negative because fees were applied after sizing.
- A stale mark MAY be used for NAV only if the age and affected exposure are reported.

### 11.5 Time parity and idempotency

- A signal from session `T` MUST map to the same eligible execution session in simulation and live workflows.
- The live workflow SHOULD derive the signal date from the scored data, not the wall clock.
- Repeating an unchanged signal MUST generate the same decision ID and MUST NOT duplicate positions or orders.
- Portfolio state writes MUST be atomic and recoverable from interruption.

## 12. Statistical and methodological requirements

### 12.1 Unit of evaluation

The primary evaluation unit for a cross-sectional ranker MUST be the signal date. The standard report MUST include:

- Per-date top-N forward excess return.
- Per-date rank IC, with the chosen correlation named.
- NDCG at the actually traded depth for ranking models.
- Precision@N on the actual signal date for binary models.
- Breadth and missingness by date.
- Portfolio results after realistic costs.

Pooled row-level ROC-AUC or PR-AUC MAY be shown as secondary classifier diagnostics. They MUST NOT be the main evidence for a per-date trading rule.

“Weekly Precision@N” is valid only when the strategy makes one explicitly selected decision per week. It MUST NOT select N rows from all ticker-days in a week.

### 12.2 Uncertainty

- Reports MUST state the number of signal dates and the number of economically non-overlapping periods.
- Overlapping-return analyses MUST use an explicit lag at least as long as the mechanical overlap, or a justified block bootstrap.
- Callers of relative metrics MUST pass the holding/overlap period explicitly. Defaulting silently to one day is not acceptable for multi-day strategies.
- Non-overlapping observations MUST NOT automatically be described as independent.
- Effect sizes and uncertainty intervals SHOULD accompany p-values or t-statistics.

### 12.3 Benchmark regressions

CAPM alpha MUST be estimated with excess returns:

```text
(strategy_return - risk_free_return)
    = alpha + beta * (benchmark_return - risk_free_return) + error
```

The report MUST name the risk-free series or scalar, sampling frequency, HAC lag, observations, beta, annualized alpha, and alpha t-statistic.

### 12.4 Research multiplicity

Every experiment that can influence the chosen production specification SHOULD be recorded in an experiment ledger containing configuration, data window, primary metric, and outcome.

After repeated feature, horizon, objective, offset, or portfolio searches on one panel:

- Nominal t-statistics MUST be labeled descriptive.
- The final claim MUST be evaluated on a locked untouched window, nested walk-forward design, or a justified multiple-testing correction.
- A sweep winner MUST NOT be presented as confirmatory evidence from the same sweep data.

No label or dollar-neutral construction may be called “market-neutral by construction.” Market beta must be measured on realized portfolio returns.

## 13. Evaluation outputs

Evaluation functions MUST return typed result objects in addition to printing or plotting. A result SHOULD include:

- Configuration and artifact identifiers.
- Point estimates and uncertainty measures.
- Sample dates and observation counts.
- Coverage and rejected-fill diagnostics.
- Cost breakdown.
- Benchmark and risk-free assumptions.
- Warnings and methodological limitations.

Plotting and terminal formatting MUST consume these result objects. They MUST NOT recompute core statistics independently.

## 14. Operational workflow

### 14.1 Required stages

```text
build dataset
  -> train/evaluate candidate
  -> validate candidate bundle
  -> refit production model
  -> validate production bundle
  -> promote atomically
  -> run dry inference
  -> optionally persist portfolio decision
```

Each arrow is a validation gate. A failed gate MUST leave the previously deployed model and portfolio state unchanged.

### 14.2 Promotion

- Candidate and deployed locations MUST be distinct.
- Promotion MUST validate loadability, required metadata, feature compatibility, freshness, and strategy horizon.
- The previous bundle MUST be archived before the deployed pointer changes.
- Promotion MUST be atomic. A crash MUST NOT leave the model and metadata from different versions.
- `--force` MUST be logged in the promotion manifest and MUST NOT erase validation findings.

### 14.3 Automation

- GitHub Actions artifact paths MUST match the paths actually produced by scripts.
- Training automation MUST upload the candidate, evaluation manifest, scored out-of-sample panel, and checksums as one run artifact.
- CI publication is not deployment to a live machine. Deployment MUST remain a separate authorized action.
- Shell scripts advertised for macOS MUST run under macOS's default Bash 3.2 or declare and check a newer Bash requirement before doing work.
- Secrets, models, portfolio state, and provider caches MUST remain outside version control.

## 15. Notebook requirements

Notebooks are explanatory clients, not alternative implementations.

- Notebooks MUST import package APIs for dataset construction, features, splitting, training, and evaluation.
- They MUST NOT implement alphabetical universe slicing, filtering membership before time-series features, unpurged row-based splits, or independent blanket `dropna()` logic.
- A notebook MUST declare the bundle schema, dataset manifest, and run identifier used for displayed results.
- Stored outputs SHOULD be cleared in version control unless they are deliberate, current, and tied to a committed manifest.
- CI MUST execute a lightweight notebook smoke test with network calls mocked or disabled.
- If a notebook cannot be maintained, archive it under a clearly marked legacy path and remove it from the README's current workflow.

## 16. Interfaces and compatibility

The following console commands SHOULD remain available during migration:

```text
train-sp500
backtest-sp500
predict-sp500
```

Existing Python imports MAY be maintained by thin re-export modules. Compatibility layers MUST:

- Call the new authoritative implementation.
- Emit a deprecation warning when appropriate.
- Have a removal version or migration note.
- Never preserve invalid scoring, execution-price, safety, or statistical behavior.

Configuration passed through CLI, Python, and stored manifests MUST normalize to the same typed configuration objects.

## 17. Error handling and observability

Use specific exception families for contract validation, incompatible bundles, data coverage, stale inputs, simulation failures, and promotion failures.

Every workflow MUST report:

- Run or decision ID.
- Model ID and bundle version.
- Dataset and universe hashes.
- Signal and execution dates.
- Row counts and cross-sectional breadth.
- Missing, stale, rejected, and filled names.
- Safety gates and whether new entries were allowed.
- Output paths and hashes.

Normal recoverable data gaps SHOULD become structured diagnostics. Broad exception handling is acceptable only at provider batch boundaries, where the failed ticker or batch and error category are retained.

## 18. Security and persistence

- Write portfolio state and promoted bundles to a temporary sibling path, flush, and atomically replace the destination.
- Validate the exact target before archival or replacement.
- Never deserialize an untrusted model artifact.
- Never log API keys, tokens, full environment dumps, or secret-bearing URLs.
- Manifests MAY record the names of relevant environment variables but MUST NOT record secret values.

## 19. Required acceptance tests

The implementation is incomplete until automated tests cover all of the following.

### 19.1 Data and features

- Time-series features remain correct across a membership exit and re-entry.
- Out-of-index names cannot affect same-date ranks or regime medians.
- A short-range provider cache cannot satisfy a longer request.
- Adjustment conventions match when Yahoo and Tiingo observations are combined.
- Missing terminal vendor data is not automatically labeled a delisting.
- A calendar-complete but unpriced inference row is rejected.
- Training and inference produce identical feature values for the same ticker-date and inputs.

### 19.2 Universe and bundle

- A capped training universe round-trips exactly through the bundle.
- Live resolution intersects the recorded universe with current PIT membership and does not reseed.
- Added post-training members are handled according to bundle policy.
- A feature reordering or schema-version mismatch blocks scoring.
- A legacy bundle produces explicit warnings and cannot be promoted unattended.
- A partially failed promotion leaves the old deployed bundle intact.

### 19.3 Portfolio and execution

- With `top_n=5` and `rank_offset=10`, all modes select ranks 11–15 from a sufficient cross-section.
- A forced rebalance cannot buy while the kill switch or freshness gate blocks entries.
- Replaying the same signal is idempotent.
- Missing exact entry or exit prices create rejected fills; they do not use a previous price.
- Valuation with a stale mark records its age and exposure.
- Fees, slippage, and financing reconcile through daily NAV.
- High fees cannot drive cash negative.
- Backtest and live policy produce identical targets from the same scores, state, configuration, and quotes, apart from declared share granularity.
- Signal-date-to-entry-date mapping is identical across simulation and live adapters.

### 19.4 Modeling and statistics

- No label window crosses a train/validation/test boundary.
- Fold-specific class weights use only fold-training labels.
- Recorded LightGBM tree count equals the early-stopped booster's effective count and the production refit's configured count.
- Per-date top-N metrics differ from and replace pooled weekly ticker-day selection.
- Synthetic CAPM data with a non-zero risk-free rate recovers the expected excess-return alpha and beta.
- Every sweep and reporting caller supplies the appropriate overlap/HAC period.
- A multiple-experiment report carries a descriptive or corrected-inference label.

### 19.5 Tooling and documentation

- Existing unit and integration tests pass.
- Ruff passes over the configured source, test, and script scope.
- All console scripts pass offline smoke tests.
- Maintained notebooks execute against mocked/local data.
- README commands reference real output paths and current artifact names.
- `specs.md`, README limitations, and `todo_list.md` do not contradict one another.

## 20. Migration plan

Each phase MUST leave the repository runnable and tested.

### Phase 0: characterize and freeze interfaces

1. Record current console commands, public imports, artifact formats, and baseline test results.
2. Add regression tests for every known correctness defect before changing implementation.
3. Mark historically invalid scored panels and notebook outputs as stale; do not use them as golden results.

Exit criterion: tests demonstrate the defects and current valid behavior separately.

### Phase 1: contracts and model bundle

1. Add typed configuration/result records and runtime dataframe validators.
2. Introduce the versioned model bundle and a legacy reader.
3. Make manifests and exact universe identity mandatory for new runs.
4. Make promotion atomic at bundle level.

Exit criterion: a new bundle can reproduce schema-valid inference without consulting CLI defaults.

### Phase 2: data and execution separation

1. Introduce the canonical long-form price panel and execution-quote validation.
2. Change backtest APIs to accept independent score and execution-price inputs.
3. Remove forward-filled prices from fill logic.
4. Add rejected-fill and stale-valuation diagnostics.

Exit criterion: no execution path obtains a fill price from a scored-panel forward fill.

### Phase 3: shared workflows and parity

1. Move orchestration into application workflows.
2. Keep CLIs and scripts as argument/config adapters.
3. Route every mode through the existing shared policy and cost primitives.
4. Enforce signal-date, safety, and idempotency parity.

Exit criterion: golden parity tests produce the same intended targets in simulation and live modes.

### Phase 4: evaluation and production fitting

1. Replace primary pooled evaluation with per-date metrics.
2. Correct CAPM to use excess returns and require explicit overlap settings.
3. Separate accepted candidate evaluation from production refitting.
4. Add experiment-ledger and multiple-testing disclosures.

Exit criterion: the standard report matches the deployed decision rule and can be reproduced from typed result artifacts.

### Phase 5: clients and cleanup

1. Convert notebooks to package API clients or archive them.
2. Update README, automation, and artifact paths.
3. Remove duplicated legacy implementations after their deprecation window.
4. Run the complete acceptance suite and a clean offline end-to-end smoke run.

Exit criterion: there is one authoritative path for each methodological and portfolio rule.

## 21. Coding-agent implementation rules

A coding agent implementing this specification MUST:

1. Read `README.md`, `todo_list.md`, this specification, and the affected tests before editing.
2. Inspect the working tree and preserve unrelated user changes.
3. Implement one migration phase or one coherent slice at a time.
4. Add or update tests in the same change as behavior.
5. Prefer adapters and re-exports over simultaneous mass renames.
6. Avoid new dependencies unless the standard library and current dependencies are insufficient; justify any addition.
7. Keep network access out of unit tests.
8. Use deterministic fixtures with explicit dates, tickers, prices, and expected outcomes.
9. Never regenerate or bless historical performance numbers merely because a test changed.
10. Report which requirements and acceptance tests were satisfied, deferred, or blocked.

When requirements conflict, correctness and fail-closed live safety take precedence over backward compatibility; point-in-time integrity takes precedence over reproducing old metrics.

## 22. Definition of done

The new architecture is complete only when:

- All required contracts and boundaries are implemented.
- All acceptance tests in section 19 pass.
- Research, backtest, and live paths share universe, feature, strategy, cost, and calendar definitions.
- Signals and execution prices are separate artifacts.
- A promoted bundle is self-describing, reproducible, validated, and atomically replaceable.
- Statistical reports evaluate the actual decision rule and state their uncertainty and selection limitations.
- Maintained notebooks and automation consume package workflows rather than duplicate them.
- The full test and lint suites pass in a clean checkout.
