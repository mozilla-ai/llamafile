#!/bin/bash
# Apply llamafile patches to the transcribe.cpp submodule.
#
# Mirrors whisper.cpp.patches/apply-patches.sh:
#   1. copy everything under llamafile-files/ into the submodule root
#      (this is how transcribe.cpp/BUILD.mk gets created)
#   2. run renames.sh for any file moves
#   3. apply every patch in patches/ with `patch -p1`
#
# The submodule is left dirty on purpose; these changes are never committed
# into the submodule. To reset:
#   cd transcribe.cpp && git reset --hard && git clean -fdx

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_DIR="$SCRIPT_DIR/../transcribe.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$TRANSCRIBE_DIR"

# Refuse to run on a dirty tree so we never stack patches on top of patches.
if [ -n "$(git status --porcelain)" ]; then
    echo "Git status is dirty. Please commit or stash your changes before applying patches."
    exit 1
fi

echo "Applying patches to transcribe.cpp submodule..."

echo "Copying all files in llamafile-files to root directory..."
cp -r "$LLAMAFILE_FILES_DIR"/* .

"$SCRIPT_DIR/renames.sh"

# Generate the version header that CMake's configure_file() produces:
# ggml/src/ggml.c #includes "ggml-version.h". It is written next to its
# source (as llama.cpp.patches/apply-patches.sh does for llama.cpp's copy)
# so BUILD.mk needs no -D flags. The vendored ggml is not a git checkout
# of ggml-org/ggml, so the commit comes from ggml/UPSTREAM.
echo "Generating ggml version header..."
ggml_version_major=$(grep -E 'GGML_VERSION_MAJOR [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_version_minor=$(grep -E 'GGML_VERSION_MINOR [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_version_patch=$(grep -E 'GGML_VERSION_PATCH [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_commit=$(sed -n 's/^sha:[[:space:]]*//p' ggml/UPSTREAM 2>/dev/null | cut -c1-7)
sed -e "s/@GGML_VERSION@/${ggml_version_major}.${ggml_version_minor}.${ggml_version_patch}/g" \
    -e "s/@GGML_BUILD_COMMIT@/${ggml_commit:-unknown}/g" \
    ggml/src/ggml-version.h.in > ggml/src/ggml-version.h

cd ..
echo "Applying modifications to upstream files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        patch -p1 < "$patch_file"
    fi
done

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd transcribe.cpp && git reset --hard && git clean -fdx"
