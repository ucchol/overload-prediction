#!/bin/sh
# Fetch the per person gaze files of Thomas, Molter and Krajbich (2021, eLife, MIT license) at the commit
# of Amendment 3, for the return rule test (Amendment 11). Usage: sh scripts/fetch_snack_gaze.sh [TARGET_DIR]
set -eu
COMMIT=f6ee87f
TARGET="${1:-data/snack_gaze}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/athms/many-item-choice.git "$WORK/repo"
git -C "$WORK/repo" checkout --quiet "$COMMIT" -- data/subject_files LICENSE
mkdir -p "$TARGET"
cp "$WORK/repo"/data/subject_files/*_desc-gazes.csv "$TARGET/"
cp "$WORK/repo/LICENSE" "$TARGET/LICENSE"
git -C "$WORK/repo" rev-parse "$COMMIT" > "$TARGET/COMMIT"
echo "snack gaze files at commit $(cat "$TARGET/COMMIT") written to $TARGET"
