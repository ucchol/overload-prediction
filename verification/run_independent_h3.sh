#!/bin/sh
# Rerun the independent implementation of test A3b of H3 (Amendment 9), file selection task.
# Usage: sh verification/run_independent_h3.sh PATH_TO_CSV [B]
# B defaults to 50 (replicates 0..B-1). Outputs go to verification/independent_h3/.
# The data definition (coding.py) is copied as coding_definition.py into a temporary folder, as in
# run_independent.sh. The data file is read in place and never copied.
set -eu
CSV="$1"
B="${2:-50}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$REPO/src/overload/coding.py" "$WORK/coding_definition.py"
python -I -B "$REPO/verification/independent_h3/h3.py" --csv "$CSV" --coding-dir "$WORK" --B "$B" \
    --out "$REPO/verification/independent_h3"
