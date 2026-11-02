#!/bin/sh
# Run the blind independent implementation of Amendment 11 (H2, the return rule on gaze sequences).
# Usage: sh verification/run_independent_h2.sh [B]   (default B = 50, seeds 110000 + b)
# Reads data/snack, data/snack_gaze and data/reutskaja. Partial results go to
# verification/independent_h2/partial/ so an interrupted run resumes; delete that folder to start fresh.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
B="${1:-50}"
python -I "$REPO/verification/independent_h2/checks.py"
python -I "$REPO/verification/independent_h2/analysis.py" --repo "$REPO" --B "$B" \
    --out "$REPO/verification/independent_h2/independent_results.json"
