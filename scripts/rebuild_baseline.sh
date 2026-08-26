#!/usr/bin/env bash
# Rebuild the performance baseline from scratch, and record what produced it.
#
# Every number this project has quoted was measured on artifacts whose exact
# provenance was not recorded: which commit, which download, which config. Some
# of those numbers turned out to be wrong for reasons that had nothing to do
# with the strategy -- a look-ahead in cohort construction, a scored panel
# pricing its own fills, an alpha that was mostly the cash rate. There is no
# way to tell from an old artifact whether it predates a given fix.
#
# So the baseline is rebuilt as one addressable object: a directory holding the
# scores, the execution prices, the model, the manifest, and the resolved
# configuration that produced them. Re-running this script with the same
# BASELINE_DIR and the same commit must reproduce it byte for byte; that is
# checked by scripts/verify_baseline.py, not assumed.
#
#   ./scripts/rebuild_baseline.sh                 # full rebuild
#   BASELINE_DIR=artifacts/baseline_b ./scripts/rebuild_baseline.sh
#
# The download is the long pole and Tiingo's free tier is rate-limited; the
# provider stops cleanly and resumes, so re-running after a limit is hit picks
# up where it left off.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${BASELINE_DIR:=artifacts/baseline}"

# --- The pinned configuration. Changing any of these makes a new baseline, ---
# --- not a rerun of this one, which is why they are recorded to disk.      ---
: "${TRAIN_PROVIDER:=hybrid}"     # recovers delisted names; survivorship matters here
: "${TRAIN_START:=2010-01-01}"
: "${TRAIN_END:=2024-12-31}"
: "${TEST_START:=2025-01-01}"
: "${SAMPLE_N:=10000}"            # no cap: the whole point-in-time universe
: "${HORIZON:=63}"
: "${TOP_N:=15}"
: "${OBJECTIVE:=rank}"
: "${USE_OPTUNA:=0}"              # tuning never moved the traded end; see README
: "${SKIP_EARNINGS:=1}"
: "${SEED:=42}"

MODEL="$BASELINE_DIR/model.pkl"
WF_SCORES="$BASELINE_DIR/wf_scored.parquet"
EXECUTION_PRICES="$BASELINE_DIR/execution_prices.parquet"
PLOTS_DIR="$BASELINE_DIR/plots"
CONFIG="$BASELINE_DIR/config.json"
LOG="$BASELINE_DIR/train.log"

mkdir -p "$BASELINE_DIR"

# Record the configuration *before* the run, so a crashed run still says what
# it was attempting.
cat > "$CONFIG" <<JSON
{
  "git_commit": "$(git rev-parse HEAD)",
  "git_dirty": $(if [ -n "$(git status --porcelain)" ]; then echo true; else echo false; fi),
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "provider": "$TRAIN_PROVIDER",
  "start": "$TRAIN_START",
  "train_end": "$TRAIN_END",
  "test_start": "$TEST_START",
  "sample_n": $SAMPLE_N,
  "horizon": $HORIZON,
  "top_n": $TOP_N,
  "objective": "$OBJECTIVE",
  "optuna": $USE_OPTUNA,
  "skip_earnings": $SKIP_EARNINGS,
  "seed": $SEED
}
JSON
echo "Config -> $CONFIG"

opts=(--horizon "$HORIZON" --wf-top-k "$TOP_N" --seed "$SEED")
[[ "$OBJECTIVE" == "rank" ]] && opts+=(--rank-objective)
[[ "$USE_OPTUNA" == "0" ]] && opts+=(--no-optuna)
[[ "$SKIP_EARNINGS" == "1" ]] && opts+=(--skip-earnings)

echo "Training -> $LOG (this takes a while; the download dominates)"
uv run train-sp500 \
  --provider "$TRAIN_PROVIDER" \
  --start "$TRAIN_START" \
  --train-end "$TRAIN_END" \
  --test-start "$TEST_START" \
  --sample-n "$SAMPLE_N" \
  --plots-dir "$PLOTS_DIR" \
  --output-model "$MODEL" \
  --wf-scores-path "$WF_SCORES" \
  --execution-prices-path "$EXECUTION_PRICES" \
  --snapshot-dir "$BASELINE_DIR/snapshot" \
  ${opts[@]+"${opts[@]}"} 2>&1 | tee "$LOG"

echo
echo "Baseline artifacts in $BASELINE_DIR:"
ls -la "$BASELINE_DIR"
echo
echo "Next: uv run python scripts/verify_baseline.py $BASELINE_DIR"
