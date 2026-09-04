#!/usr/bin/env bash
# Orchestrate train → backtest → predict from the repo root.
# Override any default via environment variables (see README: Automation).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL:=artifacts/model.pkl}"            # what predict loads
# Training writes a candidate, never the deployed model. Promotion is a
# separate, validated step (`run_pipeline.sh deploy`), so a retrain cannot
# replace the model being traded the instant it finishes.
: "${CANDIDATE:=artifacts/model_candidate.pkl}"
: "${STATE:=portfolio_state.json}"
: "${WF_SCORES:=artifacts/wf_scored.parquet}"
# Full unfiltered download, used to price fills. The scored panel is PIT
# filtered, so a holding that leaves the index stops having rows and its last
# in-index price is carried forward -- fills then execute at a stale quote.
: "${EXECUTION_PRICES:=artifacts/execution_prices.parquet}"
: "${PLOTS_DIR:=artifacts/plots}"
# hybrid recovers delisted names from Tiingo, which an unbiased *training*
# panel needs. Live scoring only ever trades current index members, and
# yfinance serves 100% of those, so the daily run does not spend Tiingo quota
# on names it could not buy.
: "${TRAIN_PROVIDER:=hybrid}"
: "${PREDICT_PROVIDER:=yfinance}"
: "${SAMPLE_N:=10000}"
# The label horizon. HOLDING_DAYS derives from it below so the exit rule and
# the thing the model was trained to predict cannot drift apart: trading a
# 63-day signal on a 10-day exit is the mismatch this file exists to prevent.
: "${HORIZON:=63}"
# Evaluation holds data back so the walk-forward has something to measure.
# A production refit does not: it should learn through the newest labellable
# date, which moves. A fixed TRAIN_END made every scheduled retrain refit the
# same window, so the monthly cron learned nothing new.
: "${TRAIN_END:=2024-12-31}"
: "${TEST_START:=2025-01-01}"
# refit mode: train through the newest date a label can exist for, which is
# HORIZON sessions behind the last session available.
: "${REFIT:=0}"
: "${OBJECTIVE:=rank}"
# Tuning never improved the traded end of the ranking under any objective or
# metric tried (README: "The label mattered; tuning did not"), and it costs
# hours. Set USE_OPTUNA=1 to re-test that claim.
: "${USE_OPTUNA:=0}"
: "${SKIP_EARNINGS:=1}"
# --- Strategy: ONE definition, read by both the backtest and the live path ---
# These used to be passed to predict-sp500 only. backtest-sp500 got nothing, so
# it measured its own defaults: setting TOP_N=25 traded 25 names against a
# simulation of 15. Anything that changes what is held belongs here.
: "${TOP_N:=15}"
: "${MAX_COHORTS:=2}"
: "${HOLDING_DAYS:=$HORIZON}"
: "${SLIPPAGE_BPS:=5}"
: "${WEIGHTING:=equal}"
: "${EXIT_RANK:=40}"
: "${RANK_OFFSET:=0}"
: "${MIN_PROB:=}"
: "${MIN_CROSS_SECTION:=}"
: "${COMMISSION_PER_SHARE:=0}"
: "${COMMISSION_PER_ORDER:=0}"
# Friday matches backtest-sp500's default. predict-sp500 defaults to "any
# day", so a daily cron used to open cohorts on a schedule the backtest never
# simulated. Set to "any" only if you also change the backtest.
: "${REBALANCE_DAY:=Friday}"
# fixed = holding_days expiry (cohort engine); rank = exit_rank decay.
# The traded engine. long-short since 2026-09-05, on the evidence in
# BASELINE.md: it is the only engine whose alpha survives four independent
# rebuilds (t +2.44..+2.94) and the only one whose measured drawdown (-13.03%)
# fits the MAX_DD kill switch below. The long-only engines' alpha is
# indistinguishable from zero in every draw -- cohort's changes sign -- and
# their drawdowns are three times the switch.
#
# This is a tilt, not an established edge: it does not clear the locked holdout
# (+1.66/+1.87/+1.95/+2.12 across two artifacts and two splits). Set
# HOLD_MODE=fixed to go back.
: "${HOLD_MODE:=long-short}"   # fixed | rank | long-short
# Long-short only. Defaults follow the backtest's own defaults; the locked
# holdout does not identify a best configuration, so these are the documented
# starting point rather than a tuned one.
: "${DECILE:=0.10}"
: "${LONG_WEIGHT:=0.5}"
: "${SHORT_WEIGHT:=0.5}"
: "${REBALANCE_EVERY:=63}"
: "${MIN_NAMES_PER_SIDE:=3}"
: "${SHORT_BORROW_ANNUAL:=0}"
# Coherent with the traded engine for the first time: long-short's measured
# max drawdown is -13.03%, inside this switch. Under HOLD_MODE=fixed or rank it
# is not -- those engines draw down -45.94% and -51.90% -- so switching back
# means raising this or expecting a halt.
: "${MAX_DD:=0.15}"
: "${OPTUNA_TRIALS:=40}"
: "${TS_CV_SPLITS:=5}"
: "${EARNINGS_WORKERS:=8}"
: "${EXTRA_TRAIN_ARGS:=}"

# Selection rules, emitted one per line so both consumers read the same list.
# The flags below exist on backtest-sp500 and predict-sp500 alike; see
# tests/test_pipeline_flags.py, which fails if the two ever diverge again.
strategy_flags() {
  local -a f=(
    --top-n "$TOP_N"
    --holding-days "$HOLDING_DAYS"
    --max-cohorts "$MAX_COHORTS"
    --slippage-bps "$SLIPPAGE_BPS"
    --weighting "$WEIGHTING"
    --exit-rank "$EXIT_RANK"
    --rank-offset "$RANK_OFFSET"
    --commission-per-share "$COMMISSION_PER_SHARE"
    --commission-per-order "$COMMISSION_PER_ORDER"
  )
  # The long-short book is sized from both ends of the ranking and turns over
  # on a calendar, so it takes its own flags. Passed only in that mode: the
  # long-only paths reject them.
  if [[ "$HOLD_MODE" == "long-short" ]]; then
    f+=(--decile "$DECILE"
        --long-weight "$LONG_WEIGHT"
        --short-weight "$SHORT_WEIGHT"
        --rebalance-every "$REBALANCE_EVERY"
        --min-names-per-side "$MIN_NAMES_PER_SIDE"
        --short-borrow-annual "$SHORT_BORROW_ANNUAL")
  fi
  [[ -n "$MIN_PROB" ]] && f+=(--min-prob "$MIN_PROB")
  [[ -n "$MIN_CROSS_SECTION" ]] && f+=(--min-cross-section "$MIN_CROSS_SECTION")
  [[ "$REBALANCE_DAY" != "any" ]] && f+=(--rebalance-day "$REBALANCE_DAY")
  [[ ${#f[@]} -gt 0 ]] && printf '%s\n' "${f[@]}"
  return 0
}

train_full() {
  local -a opts=(--horizon "$HORIZON" --wf-top-k "$TOP_N")
  if [[ "$REFIT" == "1" ]]; then
    # Everything labellable; no held-back window, so the walk-forward in this
    # run is not a measurement. Use `evaluate` for that.
    opts+=(--train-through-latest)
  fi
  [[ "$OBJECTIVE" == "rank" ]] && opts+=(--rank-objective)
  [[ "$USE_OPTUNA" == "0" ]] && opts+=(--no-optuna)
  [[ "$SKIP_EARNINGS" == "1" ]] && opts+=(--skip-earnings)
  ${DRY_RUN:+echo} uv run train-sp500 \
    --provider "$TRAIN_PROVIDER" \
    --start "${TRAIN_START:-2010-01-01}" \
    --train-end "$TRAIN_END" \
    --test-start "$TEST_START" \
    --sample-n "$SAMPLE_N" \
    --optuna-trials "$OPTUNA_TRIALS" \
    --ts-cv-splits "$TS_CV_SPLITS" \
    --earnings-workers "$EARNINGS_WORKERS" \
    --plots-dir "$PLOTS_DIR" \
    --output-model "$CANDIDATE" \
    --wf-scores-path "$WF_SCORES" \
    --execution-prices-path "$EXECUTION_PRICES" \
    --run-backtest \
    ${opts[@]+"${opts[@]}"} \
    $EXTRA_TRAIN_ARGS
}

deploy_model() {
  ${DRY_RUN:+echo} uv run python scripts/deploy_model.py \
    "$CANDIDATE" "$MODEL" \
    --expected-horizon "$HORIZON" \
    --panel "$WF_SCORES" \
    "$@"
}

backtest_only() {
  local -a flags
  # `mapfile` is bash 4+; macOS ships bash 3.2, where it is simply not a
  # command and `flags` silently stays empty -- the strategy flags would have
  # vanished rather than errored. read -r in a while loop is portable.
  flags=()
  while IFS= read -r _line; do flags+=("$_line"); done < <(strategy_flags)
  local -a mode=()
  [[ "$HOLD_MODE" == "rank" ]] && mode=(--mode rank-hold)
  [[ "$HOLD_MODE" == "long-short" ]] && mode=(--mode long-short)
  # Absent used to mean "silently fall back to forward-filled prices", which
  # is a different backtest, not a smaller one.
  if [[ ! -f "$EXECUTION_PRICES" && -z "${DRY_RUN:-}" ]]; then
    echo "ERROR: no execution price panel at $EXECUTION_PRICES." >&2
    echo "  Produce one with: ./scripts/run_pipeline.sh train-full" >&2
    echo "  (it writes EXECUTION_PRICES from the same download as the scores)" >&2
    return 1
  fi
  local -a exec_px=(--execution-prices "$EXECUTION_PRICES")
  ${DRY_RUN:+echo} uv run backtest-sp500 "$WF_SCORES" \
    --plots-dir "$PLOTS_DIR" \
    ${exec_px[@]+"${exec_px[@]}"} \
    ${flags[@]+"${flags[@]}"} \
    ${mode[@]+"${mode[@]}"}
}

predict_daily() {
  local -a confirm_flag=()
  if [[ "${1:-}" == "--confirm" ]]; then
    confirm_flag=(--confirm)
  fi
  local -a flags
  # `mapfile` is bash 4+; macOS ships bash 3.2, where it is simply not a
  # command and `flags` silently stays empty -- the strategy flags would have
  # vanished rather than errored. read -r in a while loop is portable.
  flags=()
  while IFS= read -r _line; do flags+=("$_line"); done < <(strategy_flags)
  ${DRY_RUN:+echo} uv run predict-sp500 \
    --model "$MODEL" \
    --state "$STATE" \
    --sample-n "$SAMPLE_N" \
    --provider "$PREDICT_PROVIDER" \
    --hold-mode "$HOLD_MODE" \
    --max-drawdown "$MAX_DD" \
    ${flags[@]+"${flags[@]}"} \
    ${confirm_flag[@]+"${confirm_flag[@]}"}
}

usage() {
  cat <<'EOF'
Usage: ./scripts/run_pipeline.sh <command> [options]

Commands:
  train-full     Full train (Optuna, earnings, walk-forward, snapshots, macro merge,
                 writes MODEL + WF_SCORES, runs in-process backtest)
  backtest       backtest-sp500 on WF_SCORES → PLOTS_DIR
  predict        Daily predict (dry run unless: predict --confirm)
  evaluate       Train with a held-back window: the walk-forward measures it
  refit          Train through the newest labellable date: the model to deploy
  deploy         Promote CANDIDATE to MODEL after validation (--force overrides)

Environment (examples):
  MODEL=artifacts/model.pkl STATE=portfolio_state.json SAMPLE_N=10000
  TRAIN_PROVIDER=tiingo PREDICT_PROVIDER=tiingo
  TRAIN_START=2010-01-01 TRAIN_END=2024-12-31 TEST_START=2025-01-01

Strategy (one definition, applied to BOTH the backtest and the live path):
  HORIZON=63 TOP_N=15 HOLDING_DAYS=$HORIZON MAX_COHORTS=2 WEIGHTING=equal
  EXIT_RANK=40 RANK_OFFSET=0 MIN_PROB= MIN_CROSS_SECTION=
  REBALANCE_DAY=Friday  (use "any" to trade every session)
  HOLD_MODE=fixed|rank|long-short   COMMISSION_PER_SHARE=0 COMMISSION_PER_ORDER=0
  DECILE=0.10 LONG_WEIGHT=0.5 SHORT_WEIGHT=0.5 REBALANCE_EVERY=63
    (long-short only; MIN_NAMES_PER_SIDE=3 SHORT_BORROW_ANNUAL=0)

  DRY_RUN=1 prints the command instead of running it:
    DRY_RUN=1 TOP_N=25 ./scripts/run_pipeline.sh predict

Load .env from repo root if you use python-dotenv (Tiingo/FRED keys).
EOF
}

case "${1:-}" in
  train-full) train_full ;;
  evaluate)   REFIT=0 train_full ;;
  refit)      REFIT=1 train_full ;;
  backtest)   backtest_only ;;
  deploy)     shift; deploy_model "$@" ;;
  predict)    predict_daily "${2:-}" ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
