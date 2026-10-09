#!/bin/sh
# Fetch the open snack choice data of Thomas, Molter and Krajbich (2021, eLife, MIT license)
# at the commit fixed in ANALYSIS_PLAN.md, Amendment 3. Only the four aggregate files are kept.
# Usage: sh scripts/fetch_snack_data.sh [TARGET_DIR]   (default data/snack)
set -eu
COMMIT=f6ee87f
TARGET="${1:-data/snack}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/athms/many-item-choice.git "$WORK/repo"
git -C "$WORK/repo" checkout --quiet "$COMMIT" -- data/summary_files LICENSE
mkdir -p "$TARGET"
for n in 9 16 25 36; do
    cp "$WORK/repo/data/summary_files/setsize-${n}_desc-data.csv" "$TARGET/"
done
cp "$WORK/repo/LICENSE" "$TARGET/LICENSE"
git -C "$WORK/repo" rev-parse "$COMMIT" > "$TARGET/COMMIT"
echo "snack data at commit $(cat "$TARGET/COMMIT") written to $TARGET"
