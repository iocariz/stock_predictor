# Stock predictor — backlog / follow-ups

Persistent list of **remaining** execution-realism and related items (phases 3–5 and review gaps). Completed work: phases 0–2 and 6 (calendar parity, commissions, docs/tests, kill-switch `allow_buys`).

## Backtest engine

- [ ] **Phase 3 — Integer-share backtest mode**  
  Optional `execution_mode: fractional | integer_shares`; floor shares per name; carry residual cash explicitly; keep NAV consistent with discrete lots.

- [ ] **Phase 4 — Liquidity / tradability filters**  
  Optional panel columns or merge (e.g. ADV$, spread); skip or down-rank names; mirror rules in `generate_orders` / pick list.

- [ ] **Phase 5 — Regime-dependent slippage**  
  e.g. `slippage_bps = base + f(vix_percentile)` with cap; default off.

- [ ] **Fee-inclusive daily NAV**  
  Today: cohort `net_return` / `total_costs` include commissions; `_build_daily_nav` is still price-only, so Sharpe/CAGR from NAV can be optimistic when fees are large. Either apply entry/exit cash drag on NAV or add a second “cash-adjusted” series and metrics.

## Live / `predict-sp500`

- [ ] **Signal-day vs wall-clock entry parity**  
  Backtest enters **next** session after signal; predict uses `entry_on_or_after(today)`. If you run on the same calendar day as the last score bar, live entry can differ. Option: drive entry from `panel["date"].max()` (or explicit `--signal-date`) plus `next_trading_day` to match backtest.

- [ ] **Cash guard on buy loop**  
  Enforce `cash_used <= state.cash + cash_from_sells` (or scale down dollar targets) so stressed commissions / many names cannot drive `new_cash` negative.

- [ ] **Report: cost includes fees**  
  `print_signal_report` “~Cost” uses `shares * price` only; optionally show gross vs all-in including per-leg commission.

## Tooling / hygiene

- [x] **CI** — `.github/workflows/ci.yml`: pytest on 3.12/3.13 (with the
  `tiingo` extra so no provider test skips), `ruff check`, and an offline
  smoke job covering every console script and both backtest engines.

- [x] **Lint** — Ruff configured in `pyproject.toml` (`E4,E7,E9,F,I`) and
  green across `src/`, `tests/`, `scripts/`; enforced by CI.
  - [ ] **mypy** still outstanding — untyped pandas surfaces make this a
    real piece of work rather than a config line.

- [x] **`pyproject.toml` description** — replaced the placeholder.

- [x] **Optuna metric aligned with the trading rule** — `--optuna-metric
  {auto,pr_auc,ndcg,topn_excess,topn_ir}`; early stopping uses the same
  metric in all four places that stop. Empirically it did **not** improve the
  model (see the README results disclosure), so `--no-optuna` remains the
  recommendation for the rank objective. Kept because tuning on one metric
  while stopping on another was a correctness bug.

## Research: what is left to try

**State as of 2026-11-27 (see `BASELINE.md`).** The raw `sector` label was a
survivorship flag and has been removed from the model. Without it no engine
shows alpha distinguishable from zero, the long-short book fails the locked
holdout, and one rebuild that changed 0.1% of the training rows flipped the
holdout's sign. The figures quoted in the completed items below predate that
fix and are kept as a record of what was tried, not as results. Price and
volume features on the S&P 500 are treated as exhausted: anything new here has
to bring information that is not a derivative of the OHLCV panel, be graded
with `--label-target beta_resid`, and be judged on the median of the rebalance
offsets across more than one draw.

Experiments that came back negative or neutral: rank-band trading,
vol-adjusted labels, excess labels, the beta-residual label, Optuna tuning,
tuning-metric alignment, rebalance cadence. Under every configuration,
validation performance peaks at a handful of trees, which says the signal is
not in the current inputs.

- [x] **Label targets** — `vol_adj`, `excess_vol_adj` and `beta_resid` replayed
  on one snapshot. The volatility-scaled targets invert the bias instead of
  removing it; `beta_resid` is the least bad and still shows no selection
  (residual IC +0.005). Default stays `raw`.
- [x] **Rebalance cadence** — no cadence gives significant alpha; the spread
  between rebalance offsets is larger than the spread between cadences.
  - [ ] Staggered tranches would average the offset out instead of drawing one.
- [x] **Sector survivorship leak** — `sector` removed as a feature.
  - [ ] `--sector-neutral` still only knows the sector of current members.
- [x] **Delisting evidence** — `data/delisting_proceeds.csv`, 19 of 347
  departures, used by the live book and by the baseline's pinned schedule.
  - [ ] Only two rows were checked against a published source.
  - [ ] Other rebalance offsets, the labels, and `locked_holdout.py` do not
    use it yet.
- [x] **Corrupt vendor prints** — quarantined before training.
  - [ ] The real prices are not recovered; those names go dark instead.
- [ ] **Re-measure on the current baseline**: the draw-to-draw spread (four
  fresh rebuilds), the borrow sensitivity, and the README's label-target table
  all predate it.

- [x] **Fundamentals** — SEC EDGAR point-in-time, joined on filing date.
  Used heavily by the model (31-46% of gain) but the price-only control still
  wins at horizon 63; off by default.
- [x] **Longer horizon** — the biggest lever found. `--horizon 63` takes return
  IC from +0.001 to +0.047 and the decile spread to t +2.36 on 30 independent
  periods.
- [x] **Long-short engine** — `long_short.py`, with borrow and turnover costs.
  Median Sharpe 0.74 after costs; always sweep rebalance-schedule offsets.

- [x] **Per-name short borrow** — `borrow.py`. Consumes a real `borrow_rate`
  column when supplied, else a stylised volatility-percentile proxy, else a
  flat rate. The premise turned out to be **wrong**: the short book is cheaper
  than the universe (0.58x), not dearer, because the model ranks volatility
  positively and the expensive names sit in the *long* book. Impact on median
  Sharpe: 0.74 -> 0.73.
  - [ ] Real borrow data would still beat the proxy; the hook is there.
- [ ] **Recover the delisted names.** Departed members the vendors no longer
  serve are absent from the panel, and their absence flatters every result.
  `scripts/recover_delisted.py` fetches them from Tiingo a quota window at a
  time.
- [ ] **More independent periods.** 30 is better than 14 and still modest for
  a 63-day horizon.
- [ ] **Other feature families, not labels or tuners.** Every current feature is a
  price/volume/macro derivative of the same OHLCV panel. Candidates that add
  genuinely new information: earnings surprises, fundamentals (valuation,
  margins, revisions), analyst estimate revisions, short interest. Set the
  stopping rule before running it.

---

*Add new bullets here as you discover gaps; strike through or remove when done.*
