#!/usr/bin/env bash
# Dry-run apply every patch in a patches directory to flag conflicts.
# Usage: tools/check_patches.sh [patches-dir]   (default: llama.cpp.patches/patches)
set -euo pipefail

PATCHES_DIR="${1:-llama.cpp.patches/patches}"

for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        git apply --check "$patch_file"
        #patch -p1 --dry-run < "$patch_file"
    fi
done
