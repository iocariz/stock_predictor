#!/usr/bin/env bash
# Orchestrate train → backtest → predict from the repo root.
# Override any default via environment variables (see README: Automation).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL:=artifacts/model.pkl}"
: "${STATE:=portfolio_state.json}"
: "${WF_SCORES:=artifacts/wf_scored.parquet}"
: "${PLOTS_DIR:=artifacts/plots}"
: "${TRAIN_PROVIDER:=yfinance}"
: "${PREDICT_PROVIDER:=$TRAIN_PROVIDER}"
: "${SAMPLE_N:=10000}"
: "${TOP_N:=15}"
: "${MAX_COHORTS:=2}"
: "${HOLDING_DAYS:=10}"
: "${SLIPPAGE_BPS:=5}"
: "${WEIGHTING:=equal}"
: "${MAX_DD:=0.15}"
: "${OPTUNA_TRIALS:=40}"
: "${TS_CV_SPLITS:=5}"
: "${EARNINGS_WORKERS:=8}"

train_full() {
  uv run train-sp500 \
    --provider "$TRAIN_PROVIDER" \
    --start "${TRAIN_START:-2018-01-01}" \
    --train-end "${TRAIN_END:-2022-12-31}" \
    --test-start "${TEST_START:-2023-01-01}" \
    --sample-n "$SAMPLE_N" \
    --optuna-trials "$OPTUNA_TRIALS" \
    --ts-cv-splits "$TS_CV_SPLITS" \
    --earnings-workers "$EARNINGS_WORKERS" \
    --plots-dir "$PLOTS_DIR" \
    --output-model "$MODEL" \
    --wf-scores-path "$WF_SCORES" \
    --run-backtest
}

backtest_only() {
  uv run backtest-sp500 "$WF_SCORES" --plots-dir "$PLOTS_DIR"
}

predict_daily() {
  local -a confirm_flag=()
  if [[ "${1:-}" == "--confirm" ]]; then
    confirm_flag=(--confirm)
  fi
  uv run predict-sp500 \
    --model "$MODEL" \
    --state "$STATE" \
    --sample-n "$SAMPLE_N" \
    --provider "$PREDICT_PROVIDER" \
    --top-n "$TOP_N" \
    --max-cohorts "$MAX_COHORTS" \
    --holding-days "$HOLDING_DAYS" \
    --slippage-bps "$SLIPPAGE_BPS" \
    --weighting "$WEIGHTING" \
    --max-drawdown "$MAX_DD" \
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

Environment (examples):
  MODEL=artifacts/model.pkl STATE=portfolio_state.json SAMPLE_N=10000
  TRAIN_PROVIDER=tiingo PREDICT_PROVIDER=tiingo
  TRAIN_START=2018-01-01 TRAIN_END=2022-12-31 TEST_START=2023-01-01

Load .env from repo root if you use python-dotenv (Tiingo/FRED keys).
EOF
}

case "${1:-}" in
  train-full) train_full ;;
  backtest)   backtest_only ;;
  predict)    predict_daily "${2:-}" ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
