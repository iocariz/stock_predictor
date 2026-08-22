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
: "${EXECUTION_PRICES:=artifacts/hybrid_adj_close.parquet}"
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
: "${HOLD_MODE:=fixed}"
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
  [[ -n "$MIN_PROB" ]] && f+=(--min-prob "$MIN_PROB")
  [[ -n "$MIN_CROSS_SECTION" ]] && f+=(--min-cross-section "$MIN_CROSS_SECTION")
  [[ "$REBALANCE_DAY" != "any" ]] && f+=(--rebalance-day "$REBALANCE_DAY")
  printf '%s\n' "${f[@]}"
}

train_full() {
  local -a opts=(--horizon "$HORIZON" --wf-top-k "$TOP_N")
  [[ "$OBJECTIVE" == "rank" ]] && opts+=(--rank-objective)
  [[ "$USE_OPTUNA" == "0" ]] && opts+=(--no-optuna)
  [[ "$SKIP_EARNINGS" == "1" ]] && opts+=(--skip-earnings)
  ${DRY_RUN:+echo} uv run train-sp500 \
    --provider "$TRAIN_PROVIDER" \
    --start "${TRAIN_START:-2010-01-01}" \
    --train-end "${TRAIN_END:-2024-12-31}" \
    --test-start "${TEST_START:-2025-01-01}" \
    --sample-n "$SAMPLE_N" \
    --optuna-trials "$OPTUNA_TRIALS" \
    --ts-cv-splits "$TS_CV_SPLITS" \
    --earnings-workers "$EARNINGS_WORKERS" \
    --plots-dir "$PLOTS_DIR" \
    --output-model "$CANDIDATE" \
    --wf-scores-path "$WF_SCORES" \
    --run-backtest \
    "${opts[@]}" \
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
  mapfile -t flags < <(strategy_flags)
  local -a mode=()
  [[ "$HOLD_MODE" == "rank" ]] && mode=(--mode rank-hold)
  local -a exec_px=()
  [[ -f "$EXECUTION_PRICES" ]] && exec_px=(--execution-prices "$EXECUTION_PRICES")
  ${DRY_RUN:+echo} uv run backtest-sp500 "$WF_SCORES" \
    --plots-dir "$PLOTS_DIR" \
    "${exec_px[@]}" \
    "${flags[@]}" \
    "${mode[@]}"
}

predict_daily() {
  local -a confirm_flag=()
  if [[ "${1:-}" == "--confirm" ]]; then
    confirm_flag=(--confirm)
  fi
  local -a flags
  mapfile -t flags < <(strategy_flags)
  ${DRY_RUN:+echo} uv run predict-sp500 \
    --model "$MODEL" \
    --state "$STATE" \
    --sample-n "$SAMPLE_N" \
    --provider "$PREDICT_PROVIDER" \
    --hold-mode "$HOLD_MODE" \
    --max-drawdown "$MAX_DD" \
    "${flags[@]}" \
    "${confirm_flag[@]}"
}

usage() {
  cat <<'EOF'
Usage: ./scripts/run_pipeline.sh <command> [options]

Commands:
  train-full     Full train (Optuna, earnings, walk-forward, snapshots, macro merge,
                 writes MODEL + WF_SCORES, runs in-process backtest)
  backtest       backtest-sp500 on WF_SCORES → PLOTS_DIR
  predict        Daily predict (dry run unless: predict --confirm)
  deploy         Promote CANDIDATE to MODEL after validation (--force overrides)

Environment (examples):
  MODEL=artifacts/model.pkl STATE=portfolio_state.json SAMPLE_N=10000
  TRAIN_PROVIDER=tiingo PREDICT_PROVIDER=tiingo
  TRAIN_START=2010-01-01 TRAIN_END=2024-12-31 TEST_START=2025-01-01

Strategy (one definition, applied to BOTH the backtest and the live path):
  HORIZON=63 TOP_N=15 HOLDING_DAYS=$HORIZON MAX_COHORTS=2 WEIGHTING=equal
  EXIT_RANK=40 RANK_OFFSET=0 MIN_PROB= MIN_CROSS_SECTION=
  REBALANCE_DAY=Friday  (use "any" to trade every session)
  HOLD_MODE=fixed|rank  COMMISSION_PER_SHARE=0 COMMISSION_PER_ORDER=0

  DRY_RUN=1 prints the command instead of running it:
    DRY_RUN=1 TOP_N=25 ./scripts/run_pipeline.sh predict

Load .env from repo root if you use python-dotenv (Tiingo/FRED keys).
EOF
}

case "${1:-}" in
  train-full) train_full ;;
  backtest)   backtest_only ;;
  deploy)     shift; deploy_model "$@" ;;
  predict)    predict_daily "${2:-}" ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
