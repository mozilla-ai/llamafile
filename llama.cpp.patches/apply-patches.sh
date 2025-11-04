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
rm -rf examples
rm -rf models
rm -rf ggml
rm -rf gguf-py
rm -rf tests
rm -rf src
rm -rf spm-headers
rm -rf scripts
rm -rf .clang-tidy
rm -rf .devops/
rm -rf .dockerignore
rm -rf .ecrc
rm -rf .editorconfig
rm -rf .flake8
rm -rf .github/
rm -rf .gitignore
rm -rf .gitmodules
rm -rf .pre-commit-config.yaml
rm -rf AUTHORS
rm -rf CMakeLists.txt
rm -rf CMakePresets.json
rm -rf CONTRIBUTING.md
rm -rf Makefile
rm -rf Package.swift
rm -rf README copy.llamafile
rm -rf README.md
rm -rf SECURITY.md
rm -rf ci/
rm -rf cmake/
rm -rf common/
rm -rf convert_hf_to_gguf.py
rm -rf convert_hf_to_gguf_update.py
rm -rf convert_llama_ggml_to_gguf.py
rm -rf convert_lora_to_gguf.py
rm -rf docs/
rm -rf flake.lock
rm -rf flake.nix
rm -rf grammars/
rm -rf include/
rm -rf media/
rm -rf mypy.ini
rm -rf pocs/
rm -rf poetry.lock
rm -rf prompts/
rm -rf pyproject.toml
rm -rf pyrightconfig.json
rm -rf requirements.txt
rm -rf requirements/
rm -rf scripts/
rm -rf server/themes/
rm -rf spm-headers/
rm -rf src/
rm -rf tests/

cd ..
echo "Applying modifications to upstream files..."
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        patch -p0 < "$patch_file"
    fi
done

echo ""
echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd llama.cpp && git reset --hard && git clean -fdx"
