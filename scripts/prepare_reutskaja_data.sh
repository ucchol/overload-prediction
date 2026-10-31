#!/bin/sh
# Check the replication package of Reutskaja, Nagel, Camerer and Rangel (2011, American Economic Review, "Search
# Dynamics in Consumer Choice under Time Pressure"), downloaded by hand from openICPSR project 112415, version 1
# (CC BY 4.0, a free sign in is needed), and copy its trial file and license to the data folder. The trial file, not
# the zip, is checked against the hash fixed in ANALYSIS_PLAN.md, Amendment 8.
# Usage: sh scripts/prepare_reutskaja_data.sh PATH_TO/112415-V1.zip [TARGET_DIR]   (default data/reutskaja)
set -eu
ZIP="$1"
TARGET="${2:-data/reutskaja}"
EXPECTED=e98da9eb2a0b2d1e1472e84246dcd0bc641dfbc5c60746171cd087eedc599bee
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ZIP" -d "$WORK"
DTA="$(find "$WORK" -name fixations_data.dta | head -1)"
GOT="$(sha256sum "$DTA" | cut -d' ' -f1)"
if [ "$GOT" != "$EXPECTED" ]; then
    echo "the trial file differs from the one analyzed (SHA-256 $GOT)" >&2
    exit 1
fi
mkdir -p "$TARGET"
cp "$DTA" "$TARGET/fixations_data.dta"
cp "$(find "$WORK" -name LICENSE.txt | head -1)" "$TARGET/LICENSE.txt"
echo "trial file checked and written to $TARGET"
