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

- [ ] **CI**  
  GitHub Actions: `uv sync --extra dev` + `pytest`.

- [ ] **Lint / types (optional)**  
  Ruff + mypy on `src/`.

- [ ] **`pyproject.toml` description**  
  Replace placeholder with one line aligned to README.

---

*Add new bullets here as you discover gaps; strike through or remove when done.*
