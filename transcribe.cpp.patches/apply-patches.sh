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
