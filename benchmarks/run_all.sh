#!/usr/bin/env bash
# Full local run: capture launch identity, main matrix, audit, feature smoke.
# The server must already be up. Run only when nobody else is using it.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
PREFILL_SIZES="${PREFILL_SIZES:-512,2048,8192,32768,131072,200000,400000}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8}"
RESULTS_DIR="${RESULTS_DIR:-results/matrix-$(date -u +%Y%m%dT%H%M%SZ)}"

python3 benchmarks/capture_env.py
OUTPUT_TOKENS="$OUTPUT_TOKENS" PREFILL_SIZES="$PREFILL_SIZES" \
  CONCURRENCIES="$CONCURRENCIES" RESULTS_DIR="$RESULTS_DIR" \
  python3 benchmarks/matrix.py
./benchmarks/finalize.sh "$RESULTS_DIR"
python3 benchmarks/smoke.py
echo "matrix: $RESULTS_DIR"
