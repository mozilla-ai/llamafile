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
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd llama.cpp && git reset --hard && git clean -fdx"
