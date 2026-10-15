#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its two look satisficing implementation and compare.
# Usage: sh verification/run_independent_two_look.sh [SNACK_DIR]   (default data/snack, see scripts/fetch_snack_data.sh)
# About 20 minutes for 1000 resamples on one process.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SNACK="${1:-$REPO/data/snack}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/snack" "$WORK/independent_snack" "$WORK/independent_search" "$WORK/independent_two_look"
cp "$SNACK"/setsize-*_desc-data.csv "$WORK/snack/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent_snack/*.py "$WORK/independent_snack/"
cp "$REPO"/verification/independent_search/*.py "$WORK/independent_search/"
cp "$REPO"/verification/independent_two_look/*.py "$WORK/independent_two_look/"
cd "$WORK/independent_two_look"
python -I simulate_tls.py --out sim/data --lam 1.2 --seed 5
python -I checks_two_look.py sim/data
python -I run_two_look.py --real --out results.json --B 1000
python "$REPO/verification/compare_two_look.py" --independent "$WORK/independent_two_look/results.json" \
    --results "$REPO/results/two_look_snack"
