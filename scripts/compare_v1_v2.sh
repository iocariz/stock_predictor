#!/usr/bin/env bash
set -euo pipefail

# Compare a baseline (v1) against a score-floor variant (v2) on the same
# scored panel, then print both reports side by side.
#
# Usage:
#   ./scripts/compare_v1_v2.sh artifacts/wf_scored.parquet
#
# Optional overrides (env vars):
#   TOP_N=10 HOLDING_DAYS=10 MAX_COHORTS=2 SLIPPAGE_BPS=5
#   MIN_PROB=0.55 CAPITAL=100000 PROVIDER=yfinance
#   PLOTS_DIR=artifacts/plots REPORTS_DIR=artifacts/reports
#
# Note: an earlier version of this script passed --max-sector-positions and
# --prob-calibration. Neither knob exists: sector data is never carried into
# the walk-forward scored panel, and calibration needs rolling out-of-fold
# fitting that has not been built. Both were removed rather than faked.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scored_path.parquet|csv>"
  exit 1
fi

SCORED_PATH="$1"

TOP_N="${TOP_N:-10}"
HOLDING_DAYS="${HOLDING_DAYS:-10}"
MAX_COHORTS="${MAX_COHORTS:-2}"
SLIPPAGE_BPS="${SLIPPAGE_BPS:-5}"
CAPITAL="${CAPITAL:-100000}"
PROVIDER="${PROVIDER:-yfinance}"
PLOTS_DIR="${PLOTS_DIR:-artifacts/plots}"
REPORTS_DIR="${REPORTS_DIR:-artifacts/reports}"
MIN_PROB="${MIN_PROB:-0.55}"

echo "============================================================"
echo "Strategy comparison: v1 baseline vs v2 (--min-prob ${MIN_PROB})"
echo "Scored panel: ${SCORED_PATH}"
echo "============================================================"
echo
mkdir -p "${REPORTS_DIR}" "${PLOTS_DIR}"

common_args=(
  --top-n "${TOP_N}"
  --holding-days "${HOLDING_DAYS}"
  --max-cohorts "${MAX_COHORTS}"
  --slippage-bps "${SLIPPAGE_BPS}"
  --capital "${CAPITAL}"
  --provider "${PROVIDER}"
)

echo "[1/2] Baseline v1"
uv run backtest-sp500 "${SCORED_PATH}" \
  "${common_args[@]}" \
  --plots-dir "${PLOTS_DIR}" | tee "${REPORTS_DIR}/v1_backtest.txt"

echo
echo "[2/2] V2 (score floor)"
uv run backtest-sp500 "${SCORED_PATH}" \
  "${common_args[@]}" \
  --min-prob "${MIN_PROB}" | tee "${REPORTS_DIR}/v2_backtest.txt"

echo
echo "Done."
echo "- Reports:"
echo "  - ${REPORTS_DIR}/v1_backtest.txt"
echo "  - ${REPORTS_DIR}/v2_backtest.txt"
echo "- Compare the ACTIVE-vs-benchmark alpha t-stat, not just total return."
echo "- Note: --compare-with runs ONE shared config on two panels, which is"
echo "  why two separate runs are used here for two configs on one panel."
