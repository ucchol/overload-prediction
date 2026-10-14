#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its search model implementation and compare.
# Usage: sh verification/run_independent_search.sh [SNACK_DIR]   (default data/snack, see scripts/fetch_snack_data.sh)
# About 15 to 20 minutes for 1000 resamples on one process.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SNACK="${1:-$REPO/data/snack}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/snack" "$WORK/independent_snack" "$WORK/independent_search"
cp "$SNACK"/setsize-*_desc-data.csv "$WORK/snack/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent_snack/*.py "$WORK/independent_snack/"
cp "$REPO"/verification/independent_search/*.py "$WORK/independent_search/"
python -I "$WORK/independent_search/simulate.py" --out "$WORK/independent_search/sim/data" --lam 1.0 --seed 7
python -I "$WORK/independent_search/checks_search.py" "$WORK/independent_search/sim/data"
python -I "$WORK/independent_search/run_search.py" --real --out "$WORK/independent_search/results.json" --B 1000
python "$REPO/verification/compare_search_confirmatory.py" \
    --independent "$WORK/independent_search/results.json" --results "$REPO/results/search_confirmatory"
