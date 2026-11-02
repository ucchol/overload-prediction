#!/bin/sh
# Independent implementation of Amendment 12 (decision time from trial structure), written blind from the plan.
# Runs the self checks, then the analysis with B bootstrap replicates (default 20), and writes
# verification/independent_time/independent_results.json.
# Usage: sh verification/run_independent_time.sh [B]
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-20}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
python -I verification/independent_time/checks.py
python -I verification/independent_time/run_time.py --B "$B" \
    --out "$REPO/verification/independent_time/independent_results.json"
