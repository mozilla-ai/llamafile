#!/bin/bash
# Apply llamafile patches to whisper.cpp submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/../whisper.cpp.submodule"

cd "$WHISPER_DIR"

# Check if patches are already applied
if [ -f "BUILD.mk" ]; then
    echo "Patches appear to be already applied. Skipping..."
    exit 0
fi

echo "Applying patches to whisper.cpp submodule..."

# Apply patches for modified files
echo "Applying server.cpp patch..."
patch -p0 < "$SCRIPT_DIR/001-server-llamafile-integration.patch"

echo "Applying common.cpp patch..."
patch -p0 < "$SCRIPT_DIR/002-common-llamafile-integration.patch"

echo "Applying whisper.cpp core patch..."
patch -p0 < "$SCRIPT_DIR/003-whisper-core-llamafile-integration.patch"

echo "Applying common.h patch..."
patch -p0 < "$SCRIPT_DIR/004-common-header.patch"

# Copy new llamafile-specific files to the root of whisper.cpp submodule
echo "Copying llamafile-specific files..."
cp "$SCRIPT_DIR/BUILD.mk" .
cp "$SCRIPT_DIR/README.llamafile" .
cp "$SCRIPT_DIR/color.cpp" .
cp "$SCRIPT_DIR/color.h" .
cp "$SCRIPT_DIR/slurp.cpp" .
cp "$SCRIPT_DIR/slurp.h" .

# Copy server.cpp from examples/server/ to root
echo "Copying server.cpp to root..."
cp examples/server/server.cpp .

# Copy common files from examples/ to root
echo "Copying common files to root..."
cp examples/common.cpp .
cp examples/common.h .

# Copy core whisper files from src/ and include/ to root
echo "Copying whisper core files to root..."
cp src/whisper.cpp .
cp include/whisper.h .

# Copy httplib.h from examples/server/ to root
echo "Copying httplib.h to root..."
cp examples/server/httplib.h .

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd whisper.cpp.submodule && git reset --hard && git clean -fd"
