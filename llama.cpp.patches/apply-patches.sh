#!/bin/bash
# Apply llamafile patches to llama.cpp submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/../llama.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$LLAMA_DIR"

# Check if status is dirty, if so, exit
if [ -n "$(git status --porcelain)" ]; then
    echo "Git status is dirty. Please commit or stash your changes before applying patches."
    exit 1
fi

echo "Applying patches to llama.cpp submodule..."

echo "Copying all files in llamafile-files to root directory..."
cp -r "$LLAMAFILE_FILES_DIR"/* .

../llama.cpp.patches/renames.sh

echo "Removing unnecessary files and directories..."
# If you want to clean up the original code, add your `rm` commands here.
# For example:
rm Makefile

cd ..
echo "Applying modifications to upstream files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        patch -p1 < "$patch_file"
    fi
done

echo ""
echo "Fetching prebuilt web UI assets from Hugging Face..."
# Upstream's tools/ui/scripts/ui-assets.cmake pulls the Svelte build outputs
# from the ggml-org/llama-ui HF bucket. We do the same here so the cosmocc
# build never has to run a JS toolchain. If the fetch fails (no network,
# version not yet published, HF down) we silently leave tools/ui/dist
# empty; BUILD.mk's embed step then generates a no-asset ui.cpp and the
# server still works, just without the web UI.
HF_BUCKET="${LLAMAFILE_UI_HF_BUCKET:-llama-ui}"
HF_BASE="https://huggingface.co/buckets/ggml-org/${HF_BUCKET}/resolve"
UI_DIST="llama.cpp/tools/ui/dist"
UI_ASSETS=(bundle.css bundle.js index.html loading.html)

# Derive the version tag the same way upstream's CMake does: prefer the
# build number embedded in `git describe` (e.g. b9341 -> b9341), fall back
# to "latest".
UI_VERSION="$(cd llama.cpp && git describe --tags --always 2>/dev/null | grep -oE '^b[0-9]+' || true)"
UI_CANDIDATES=()
if [ -n "$UI_VERSION" ]; then
    UI_CANDIDATES+=("$UI_VERSION")
fi
UI_CANDIDATES+=("latest")

mkdir -p "$UI_DIST"
ui_ok=false
for v in "${UI_CANDIDATES[@]}"; do
    echo "  trying $HF_BASE/$v ..."
    fail=false
    for asset in "${UI_ASSETS[@]}" checksums.txt; do
        if ! curl -fsSL --max-time 60 -o "$UI_DIST/$asset" \
                "$HF_BASE/$v/$asset?download=true"; then
            fail=true
            break
        fi
    done
    if $fail; then
        continue
    fi

    # Best-effort sha256 verification against checksums.txt (one "<hash>  <name>"
    # line per asset). Skip if shasum/sha256sum isn't around.
    if command -v shasum >/dev/null 2>&1; then
        sha_cmd="shasum -a 256"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha_cmd="sha256sum"
    else
        sha_cmd=""
    fi
    if [ -n "$sha_cmd" ] && [ -f "$UI_DIST/checksums.txt" ]; then
        bad=false
        for asset in "${UI_ASSETS[@]}"; do
            want=$(awk -v a="$asset" '$2 == a { print $1 }' "$UI_DIST/checksums.txt")
            got=$($sha_cmd "$UI_DIST/$asset" | awk '{print $1}')
            if [ -z "$want" ] || [ "$want" != "$got" ]; then
                echo "  checksum mismatch for $asset (want=$want got=$got)"
                bad=true
                break
            fi
        done
        if $bad; then
            continue
        fi
    fi

    echo "  fetched UI assets from $v"
    ui_ok=true
    break
done

if ! $ui_ok; then
    echo "  warning: could not download UI assets; server will build without the web UI"
    rm -f "$UI_DIST"/*
fi

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd llama.cpp && git reset --hard && git clean -fdx"
