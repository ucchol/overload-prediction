#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its blind check of mixed logit and of the fitted
# overload forecasts (original sample, every model, held out and fitted) and compare with this package.
# Usage: sh verification/run_independent_mxl.sh PATH_TO_CSV [CDM_DIR] [REUTSKAJA_DIR] [SNACK_DIR]
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CSV="$1"
CDM="${2:-$REPO/data/cdm}"
REU="${3:-$REPO/data/reutskaja}"
SNACK="${4:-$REPO/data/snack}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/cdm" "$WORK/reutskaja" "$WORK/snack" "$WORK/stimuli"
for d in independent independent_snack independent_search independent_cdm independent_overload independent_mxl; do
    mkdir -p "$WORK/$d"
    cp "$REPO"/verification/$d/*.py "$WORK/$d/"
done
cp "$CSV" "$WORK/Updated_Choice_Overload.csv"
cp "$CDM/ProcessedCP_All_Final.csv" "$WORK/cdm/"
cp "$REU/fixations_data.dta" "$WORK/reutskaja/"
cp "$SNACK"/setsize-*_desc-data.csv "$WORK/snack/"
cp "$REPO"/src/overload/stimuli/*.json "$WORK/stimuli/"
cp "$REPO/src/overload/coding.py" "$WORK/coding_definition.py"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cd "$WORK/independent_mxl"
python -I checks_mxl.py
python -I run_mxl.py
cd "$REPO"
python -I verification/compare_mxl.py "$WORK/independent_mxl/out.json"
