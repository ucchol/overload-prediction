#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its implementation of Amendment 7 (the search
# model's stopping rule on the data of Caplin, Dean and Martin) and compare.
# Usage: sh verification/run_independent_cdm.sh [CDM_DIR]   (default data/cdm, see data/README.md for the download)
# About an hour for 1000 resamples on one process. B=3 sh ... gives a quick smoke test.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CDM="${1:-$REPO/data/cdm}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/cdm" "$WORK/independent_snack" "$WORK/independent_search" "$WORK/independent_two_look" \
    "$WORK/independent_gaze" "$WORK/independent_cdm"
cp "$CDM/ProcessedCP_All_Final.csv" "$WORK/cdm/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent_snack/*.py "$WORK/independent_snack/"
cp "$REPO"/verification/independent_search/*.py "$WORK/independent_search/"
cp "$REPO"/verification/independent_two_look/*.py "$WORK/independent_two_look/"
cp "$REPO"/verification/independent_gaze/*.py "$WORK/independent_gaze/"
cp "$REPO"/verification/independent_cdm/*.py "$WORK/independent_cdm/"
cd "$WORK/independent_cdm"
mkdir -p sim
python -I simulate_cdm.py --scenario list --out sim/clicks_list.csv --seed 11
python -I simulate_cdm.py --scenario random --out sim/clicks_random.csv --seed 11
python -I checks_cdm.py sim
python -I run_cdm.py --real --out results.json --B "${B:-1000}"
python "$REPO/verification/compare_stopping_cdm.py" --independent "$WORK/independent_cdm/results.json" \
    --results "$REPO/results/stopping_cdm"
