#!/bin/sh
# Rebuild the folder layout the independent verifier worked in, run its implementation of the overload, decision
# time, search model and premonition analyses, and compare it with the package.
# Usage: sh verification/run_independent_process.sh PATH_TO_CSV
# The data file is copied only into a temporary folder, which is removed at the end. About ten minutes with two CPUs.
set -eu
CSV="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$CSV" "$WORK/Updated_Choice_Overload.csv"
mkdir -p "$WORK/stimuli" "$WORK/independent" "$WORK/independent_process"
cp "$REPO"/src/overload/stimuli/*.json "$WORK/stimuli/"
cp "$REPO/src/overload/coding.py" "$WORK/coding_definition.py"
cp "$REPO/ANALYSIS_PLAN.md" "$WORK/"
cp "$REPO"/verification/independent/*.py "$WORK/independent/"
cp "$REPO"/verification/independent_process/*.py "$WORK/independent_process/"
python "$WORK/independent_process/checks_process.py"
python "$WORK/independent_process/run_process.py" --B 1000
python "$REPO/verification/compare_process.py" --independent "$WORK/independent_process/results.json" \
    --overload "$REPO/results/overload" --process "$REPO/results/process_dev"
