#!/bin/sh
# Run the blind independent implementation of Amendment 10 (H1w) on the Aguiar et al. data.
# Usage: sh verification/run_independent_h1w.sh [DATA_DIR] [B]
# DATA_DIR holds menu_choice_low.csv, menu_choice_medium.csv and menu_choice_high.csv (default data/aguiar).
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$REPO/data/aguiar}"
B="${2:-1000}"
python -I "$REPO/verification/independent_h1w/checks.py"
python -I "$REPO/verification/independent_h1w/analysis.py" --data "$DATA" --B "$B" \
    --out "$REPO/verification/independent_h1w/independent_results.json"
