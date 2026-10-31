#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its implementation of Amendment 8 (forecasting
# choice overload) and compare.
# Usage: sh verification/run_independent_overload.sh [CDM_DIR] [REUTSKAJA_DIR]   (defaults data/cdm and data/reutskaja)
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CDM="${1:-$REPO/data/cdm}"
REU="${2:-$REPO/data/reutskaja}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/cdm" "$WORK/reutskaja"
for d in independent_snack independent_search independent_two_look independent_gaze independent_cdm independent_overload; do
    mkdir -p "$WORK/$d"
    cp "$REPO"/verification/$d/*.py "$WORK/$d/"
done
cp "$CDM/ProcessedCP_All_Final.csv" "$WORK/cdm/"
cp "$REU/fixations_data.dta" "$WORK/reutskaja/"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cd "$WORK/independent_overload"
python -I run_ov.py --real --out results.json --B "${B:-1000}"
python "$REPO/verification/compare_overload.py" --independent "$WORK/independent_overload/results.json" \
    --results "$REPO/results/overload_premonition"
