#!/bin/bash
# Apply llamafile patches to whisper.cpp submodule

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/../whisper.cpp"

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
cp $SCRIPT_DIR/README.md .
cp $SCRIPT_DIR/*.h .
cp $SCRIPT_DIR/*.c* .
cp $SCRIPT_DIR/*.hpp .
cp $SCRIPT_DIR/main.1* .
cp $SCRIPT_DIR/jfk.wav .
cp -r $SCRIPT_DIR/doc .

# Copy server.cpp from examples/server/ to root
echo "Copying server.cpp to root..."
cp examples/server/server.cpp .

# Copy common files from examples/ to root
echo "Copying common files to root..."
cp examples/common.cpp .
cp examples/common.h .

rm -rf bindings
rm -rf .github
rm -rf .devops
rm -rf examples
rm -rf ggml
rm -rf src
rm -rf tests
rm -rf scripts
rm -rf spm-headers
rm -rf cmake
rm -rf models
rm -rf samples
rm .gitignore
rm .gitmodules Package.swift README_sycl.md AUTHORS CMakeLists.txt Makefile
rm grammars/assistant.gbnf grammars/chess.gbnf grammars/colors.gbnf
rm include/whisper.h

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd whisper.cpp && git reset --hard && git clean -fd"
