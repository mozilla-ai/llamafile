#!/bin/bash
# Apply llamafile patches to agent.cpp submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR/../agent.cpp"
PATCHES_DIR="$SCRIPT_DIR/patches"
LLAMAFILE_FILES_DIR="$SCRIPT_DIR/llamafile-files"

cd "$AGENT_DIR"

# Check if status is dirty, if so, exit
if [ -n "$(git status --porcelain)" ]; then
    echo "Git status is dirty. Please commit or stash your changes before applying patches."
    exit 1
fi

echo "Applying patches to agent.cpp submodule..."

# Copy any llamafile-specific files into the submodule root (currently none)
if [ -n "$(ls -A "$LLAMAFILE_FILES_DIR" 2>/dev/null)" ]; then
    echo "Copying llamafile-files into agent.cpp..."
    cp -r "$LLAMAFILE_FILES_DIR"/* .
fi

cd ..
echo "Applying modifications to upstream files..."
shopt -s nullglob
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
echo "  cd agent.cpp && git reset --hard && git clean -fdx"
