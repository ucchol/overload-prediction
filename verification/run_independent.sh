#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its implementation and compare.
# Usage: sh verification/run_independent.sh PATH_TO_CSV
# The data file is copied only into a temporary folder, which is removed at the end.
set -eu
CSV="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$CSV" "$WORK/Updated_Choice_Overload.csv"
mkdir -p "$WORK/stimuli" "$WORK/independent"
cp "$REPO"/src/overload/stimuli/*.json "$WORK/stimuli/"
cp "$REPO/src/overload/coding.py" "$WORK/coding_definition.py"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent/*.py "$WORK/independent/"
python "$WORK/independent/checks.py"
python "$WORK/independent/analysis.py" --B 1000
python "$REPO/verification/compare_independent.py" --data "$CSV" \
    --independent "$WORK/independent/results.json" \
    --baseline "$REPO/results/baseline" --h1 "$REPO/results/h1"
