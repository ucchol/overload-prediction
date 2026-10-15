#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its implementation of Amendment 6 (two look
# satisficing on three gaze datasets) and compare.
# Usage: sh verification/run_independent_gaze.sh [GAZE_DIR]   (default data/gaze, see scripts/fetch_gaze_data.sh)
# About 25 minutes for 1000 resamples on one process.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GAZE="${1:-$REPO/data/gaze}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/gaze" "$WORK/independent_snack" "$WORK/independent_search" "$WORK/independent_two_look" \
    "$WORK/independent_gaze"
cp "$GAZE"/krajbich2010.csv "$GAZE"/krajbich2011.csv "$GAZE"/folke2016.csv "$GAZE"/folke2016_original.csv "$WORK/gaze/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent_snack/*.py "$WORK/independent_snack/"
cp "$REPO"/verification/independent_search/*.py "$WORK/independent_search/"
cp "$REPO"/verification/independent_two_look/*.py "$WORK/independent_two_look/"
cp "$REPO"/verification/independent_gaze/*.py "$WORK/independent_gaze/"
cd "$WORK/independent_gaze"
python -I simulate_gaze.py --out sim/data --lam 0.5 --seed 9
python -I checks_gaze.py sim/data
python -I run_gaze.py --real --out results.json --B "${B:-1000}"
python "$REPO/verification/compare_two_look_gaze.py" --independent "$WORK/independent_gaze/results.json" \
    --results "$REPO/results/two_look_gaze"
