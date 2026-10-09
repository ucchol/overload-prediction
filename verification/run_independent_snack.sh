#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its snack implementation and compare.
# Usage: sh verification/run_independent_snack.sh [SNACK_DIR]   (default data/snack, see scripts/fetch_snack_data.sh)
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SNACK="${1:-$REPO/data/snack}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/snack" "$WORK/independent_snack"
cp "$SNACK"/setsize-*_desc-data.csv "$WORK/snack/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent_snack/*.py "$WORK/independent_snack/"
python "$WORK/independent_snack/checks_snack.py"
python "$WORK/independent_snack/run_snack.py" --B 1000
python "$REPO/verification/compare_h1_confirmatory.py" \
    --independent "$WORK/independent_snack/results.json" --results "$REPO/results/h1_confirmatory"
