#!/usr/bin/env bash
# Dry-run apply every patch in a patches directory to flag conflicts.
# Usage: tools/check_patches.sh [patches-dir]   (default: llama.cpp.patches/patches)
#
# `git apply --check` failures are collected across the whole loop rather than
# aborting on the first — the point of the tool is to list every conflicting
# patch in one pass. The script exits non-zero if any patch fails.
set -uo pipefail

PATCHES_DIR="${1:-llama.cpp.patches/patches}"

rc=0
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        git apply --check "$patch_file" || rc=1
        #patch -p1 --dry-run < "$patch_file"
    fi
done

exit $rc
