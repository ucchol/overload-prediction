#!/bin/sh
# Check the replication package of Caplin, Dean and Martin (2011, American Economic Review, "Search and Satisficing"),
# downloaded by hand from openICPSR project 116100, version 1 (CC BY 4.0, a free sign in is needed), and copy its
# click file and license to the data folder. The package is zipped when it is downloaded, so the click file, not
# the zip, is checked against the hash fixed in ANALYSIS_PLAN.md, Amendment 7.
# Usage: sh scripts/prepare_cdm_data.sh PATH_TO/116100-V1.zip [TARGET_DIR]   (default data/cdm)
set -eu
ZIP="$1"
TARGET="${2:-data/cdm}"
EXPECTED=86adf9e6331192a06765224b41cc4e10d263a85e624cee4da141e684524f7d9e
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ZIP" -d "$WORK"
CSV="$WORK/20090912_data/Data/ProcessedCP_All_Final.csv"
GOT="$(sha256sum "$CSV" | cut -d' ' -f1)"
if [ "$GOT" != "$EXPECTED" ]; then
    echo "the click file differs from the one analyzed (SHA-256 $GOT)" >&2
    exit 1
fi
mkdir -p "$TARGET"
cp "$CSV" "$TARGET/ProcessedCP_All_Final.csv"
cp "$WORK/LICENSE.txt" "$TARGET/LICENSE.txt"
echo "click file checked and written to $TARGET"
