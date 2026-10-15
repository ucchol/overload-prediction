#!/bin/sh
# Fetch three open gaze datasets of value based choice, as prepared by Thomas, Molter, Krajbich, Heekeren and Mohr
# (2019, Nature Human Behaviour, MIT license), at the commit fixed in ANALYSIS_PLAN.md, Amendment 6.
# Krajbich, Armel and Rangel (2010), two foods. Krajbich and Rangel (2011), three foods. Folke, Jacobsen, Fleming
# and De Martino (2016), three snacks, with the original file for its confidence ratings.
# Usage: sh scripts/fetch_gaze_data.sh [TARGET_DIR]   (default data/gaze)
set -eu
COMMIT=4a4f230fb5aa2500d81bfc16b44da3478519ca4f
TARGET="${1:-data/gaze}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/athms/gaze-bias-differences.git "$WORK/repo"
git -C "$WORK/repo" checkout --quiet "$COMMIT" -- LICENSE \
    data/krajbich_2010_natneuro/krajbich2010_prep_all.csv \
    data/krajbich_2011_pnas/krajbich2011_prep_all.csv \
    data/folke_2016_nhb/folke2016_prep_all.csv \
    data/folke_2016_nhb/original/exp2_main_data.csv
mkdir -p "$TARGET"
cp "$WORK/repo/data/krajbich_2010_natneuro/krajbich2010_prep_all.csv" "$TARGET/krajbich2010.csv"
cp "$WORK/repo/data/krajbich_2011_pnas/krajbich2011_prep_all.csv" "$TARGET/krajbich2011.csv"
cp "$WORK/repo/data/folke_2016_nhb/folke2016_prep_all.csv" "$TARGET/folke2016.csv"
cp "$WORK/repo/data/folke_2016_nhb/original/exp2_main_data.csv" "$TARGET/folke2016_original.csv"
cp "$WORK/repo/LICENSE" "$TARGET/LICENSE"
git -C "$WORK/repo" rev-parse "$COMMIT" > "$TARGET/COMMIT"
echo "gaze data at commit $(cat "$TARGET/COMMIT") written to $TARGET"
