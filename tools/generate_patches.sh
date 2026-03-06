#!/bin/bash

# --- Configuration ---
# Default output directory for the patch files
DEFAULT_OUTPUT_DIR="./candidate_patches"

# --- Argument Parsing ---
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--output-dir <directory>]"
            echo ""
            echo "Generate patches and collect new files from a git submodule."
            echo "Run this script from within a submodule directory (e.g., llama.cpp, whisper.cpp)."
            echo ""
            echo "Options:"
            echo "  --output-dir <dir>  Output directory (default: ./candidate_patches)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Output structure:"
            echo "  <output-dir>/patches/         Modified files as .patch files"
            echo "  <output-dir>/llamafile-files/ New (untracked) files"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output-dir <directory>]"
            exit 1
            ;;
    esac
done

# --- Detect Repository Name from Current Directory ---
REPO_NAME=$(basename "$(pwd)")

# Verify we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: Not inside a git repository."
    exit 1
fi

# --- User Confirmation ---
echo "Detected repository: $REPO_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""
read -p "Proceed with patch generation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# --- Main Logic ---

echo ""
echo "Starting patch generation..."
echo "---"

# Create output directories
PATCHES_DIR="${OUTPUT_DIR}/patches"
FILES_DIR="${OUTPUT_DIR}/llamafile-files"
mkdir -p "$PATCHES_DIR" "$FILES_DIR"

# Track counts
PATCH_COUNT=0
NEW_FILE_COUNT=0

# Get modified files (tracked files with changes)
MODIFIED_FILES=$(git status -s | grep '^ M\|^M ' | awk '{print $2}')

# Get new/untracked files
NEW_FILES=$(git status -s | grep '^??\|^A ' | awk '{print $2}')

# BUILD.mk is always present in submodules but may be gitignored on case-insensitive
# filesystems (macOS). Ensure it's included if it exists.
if [ -f "BUILD.mk" ] && ! echo "$NEW_FILES" | grep -q '^BUILD\.mk$'; then
    NEW_FILES="$NEW_FILES BUILD.mk"
fi

# --- Process Modified Files (create patches) ---
if [ -n "$MODIFIED_FILES" ]; then
    echo ""
    echo "Processing modified files..."

    for FILE_PATH in $MODIFIED_FILES; do
        echo "  $FILE_PATH"

        # Ensure the file actually exists
        if [ ! -f "$FILE_PATH" ]; then
            echo "    Skipping: File not found (deleted?)"
            continue
        fi

        # Run git diff for the file
        DIFF_OUTPUT=$(git diff "$FILE_PATH")

        # Check if the diff is empty
        if [ -z "$DIFF_OUTPUT" ]; then
            echo "    Skipping: Diff is empty"
            continue
        fi

        # Modify the diff output:
        # - Remove the "index..." row (the second line)
        # - Prepend "REPO_NAME/" to 'a/' and 'b/' in the paths
        MODIFIED_DIFF=$(echo "$DIFF_OUTPUT" | \
            awk 'NR == 2 {next} {print}' | \
            sed "s|^\(--- a/\)\(.*\)$|\1${REPO_NAME}/\2|" | \
            sed "s|^\(\+\+\+ b/\)\(.*\)$|\1${REPO_NAME}/\2|")

        # Generate the output filename (replace '/' with '_')
        BASE_FILENAME=$(echo "$FILE_PATH" | tr '/' '_')
        PATCH_FILENAME="${BASE_FILENAME}.patch"
        OUTPUT_PATH="${PATCHES_DIR}/${PATCH_FILENAME}"

        # Save the patch file
        echo "$MODIFIED_DIFF" > "$OUTPUT_PATH"
        echo "    -> $OUTPUT_PATH"
        ((PATCH_COUNT++))
    done
fi

# --- Process New Files (copy to llamafile-files) ---
if [ -n "$NEW_FILES" ]; then
    echo ""
    echo "Processing new files..."

    for FILE_PATH in $NEW_FILES; do
        echo "  $FILE_PATH"

        # Skip directories (git status shows them with trailing /)
        if [ -d "$FILE_PATH" ]; then
            echo "    Skipping: Is a directory"
            continue
        fi

        # Skip symlinks
        if [ -L "$FILE_PATH" ]; then
            echo "    Skipping: Is a symlink"
            continue
        fi

        # Ensure the file exists
        if [ ! -f "$FILE_PATH" ]; then
            echo "    Skipping: File not found"
            continue
        fi

        # Create destination directory structure
        DEST_PATH="${FILES_DIR}/${FILE_PATH}"
        DEST_DIR=$(dirname "$DEST_PATH")
        mkdir -p "$DEST_DIR"

        # Copy the file
        cp "$FILE_PATH" "$DEST_PATH"
        echo "    -> $DEST_PATH"
        ((NEW_FILE_COUNT++))
    done
fi

# --- Summary ---
echo ""
echo "---"
echo "Patch generation completed!"
echo "  Patches created: $PATCH_COUNT (in $PATCHES_DIR)"
echo "  New files copied: $NEW_FILE_COUNT (in $FILES_DIR)"
