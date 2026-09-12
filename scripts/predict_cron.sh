#!/usr/bin/env bash
# The scheduled daily run, with the retry an unattended job needs.
#
# The first night the cron fired, it died on the coverage guard:
#
#   DownloadCoverageError: equity download: only 73/470 current index members
#   returned (15.5%), below the 98% threshold.
#
# The guard was right to stop -- a partial download silently changes the traded
# universe and every cross-sectional feature computed from it -- but the run was
# then simply lost until someone read the log. Yahoo throttles large batches.
#
# So: retry, and lower the batch size each time, which is the remedy the error
# message itself names. Retrying an identical request against a vendor that is
# throttling is a slower way to get the same answer.
#
# Only *transient* failures are retried. A missing model or a bad flag will
# return the same error on the third attempt, so it fails once and says so.
#
# Nothing here weakens the coverage guard: a run that cannot see the universe
# still refuses to trade. Retrying is about getting a clean run, never about
# accepting a dirty one.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PIPELINE_CMD="${PIPELINE_CMD:-./scripts/run_pipeline.sh}"
LOG_FILE="${LOG_FILE:-logs/predict.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_WAIT="${RETRY_WAIT:-300}"
# Each attempt asks for smaller batches than the last. The ladder starts at 50
# rather than the provider default of 100 because 100 is the size that was
# actually throttled: 73 of 470 members returned. Starting there would repeat a
# request already known to fail against this vendor.
BATCH_SIZES="${BATCH_SIZES:-50 25 10}"
export HOLD_MODE="${HOLD_MODE:-long-short}"

mkdir -p "$(dirname "$LOG_FILE")"

# Failures worth a second attempt: a throttled or partial download, and the
# network errors that produce one. Anything else is a real fault.
TRANSIENT='DownloadCoverageError|Too Many Requests|rate.?limit|Connection|Timeout|Temporary failure|EOF occurred'

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

attempt=0
for size in $BATCH_SIZES; do
  attempt=$((attempt + 1))
  [[ $attempt -gt $MAX_ATTEMPTS ]] && break

  log "attempt $attempt/$MAX_ATTEMPTS (batch size $size)"
  out="$(PREDICT_BATCH_SIZE="$size" "$PIPELINE_CMD" predict --confirm 2>&1)"
  rc=$?
  echo "$out" >> "$LOG_FILE"

  if [[ $rc -eq 0 ]]; then
    log "run succeeded on attempt $attempt"
    exit 0
  fi

  if ! echo "$out" | grep -qE "$TRANSIENT"; then
    log "attempt $attempt failed permanently (rc=$rc); not retrying"
    exit "$rc"
  fi

  log "attempt $attempt hit a transient failure (rc=$rc)"
  if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
    log "waiting ${RETRY_WAIT}s before retrying with a smaller batch"
    sleep "$RETRY_WAIT"
  fi
done

log "all $MAX_ATTEMPTS attempts failed; the book was not advanced"
exit 1
