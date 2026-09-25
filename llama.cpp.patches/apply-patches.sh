#!/bin/bash
# Apply llamafile patches to llama.cpp submodule

set -e

# Usage: apply-patches.sh [--tolerant]
#   (default)  strict — abort on the first patch that does not apply cleanly.
#              This is what `make setup` uses, so the common case fails early.
#   --tolerant apply every patch, leaving *.rej files for hunks that do not fit
#              and continuing instead of aborting; used during a llama.cpp bump
#              to reconcile drifted hunks by hand. Exits non-zero if any rejects.
TOLERANT=""
for arg in "$@"; do
    case "$arg" in
        --tolerant) TOLERANT=1 ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--tolerant]"; exit 1 ;;
    esac
done

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

# Generate the version headers that CMake's configure_file() produces. Since
# b11100 ggml/src/ggml.c #includes "ggml-version.h" and src/llama.cpp #includes
# "llama-version.h"; before that the same macros came from -D flags. They are
# written next to their sources (rather than into o/) so that every consumer
# finds them: the cosmocc make build, and llamafile/*.sh + *.bat, which compile
# ggml.c with -I ggml/src.
echo "Generating version headers..."
ggml_version_major=$(grep -E 'GGML_VERSION_MAJOR [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_version_minor=$(grep -E 'GGML_VERSION_MINOR [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_version_patch=$(grep -E 'GGML_VERSION_PATCH [0-9]+' ggml/CMakeLists.txt | sed 's/[^0-9]*//g')
ggml_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
llama_version=$(git describe --tags --always 2>/dev/null || echo "unknown")

sed -e "s/@GGML_VERSION@/${ggml_version_major}.${ggml_version_minor}.${ggml_version_patch}/g" \
    -e "s/@GGML_BUILD_COMMIT@/${ggml_commit}/g" \
    ggml/src/ggml-version.h.in > ggml/src/ggml-version.h

sed -e "s/@LLAMA_VERSION@/${llama_version}/g" \
    -e "s/@LLAMA_BUILD_COMMIT@/${ggml_commit}/g" \
    src/llama-version.h.in > src/llama-version.h

echo "Removing unnecessary files and directories..."
# If you want to clean up the original code, add your `rm` commands here.
# For example:
rm Makefile

cd ..
echo "Applying modifications to upstream files..."
FAILED_PATCHES=()
for patch_file in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch_file" ]; then
        echo "Applying $(basename "$patch_file")..."
        if [ -n "$TOLERANT" ]; then
            # Apply the hunks that fit, leave *.rej for the rest, and keep going.
            patch -p1 < "$patch_file" || FAILED_PATCHES+=("$(basename "$patch_file")")
        else
            # Strict (default): abort on the first patch that does not apply.
            patch -p1 < "$patch_file"
        fi
    fi
done

# Fetch the prebuilt web UI assets (see fetch-ui-assets.sh for details).
"$SCRIPT_DIR/fetch-ui-assets.sh"

echo ""
if [ ${#FAILED_PATCHES[@]} -gt 0 ]; then
    # patch leaves a *.orig backup beside each reject; drop them (the pristine
    # original is recoverable from git) so only the actionable *.rej remain and
    # generate-patches doesn't pick the backups up as new files.
    find llama.cpp -name '*.orig' -delete
    echo "Applied with rejects (tolerant mode): ${#FAILED_PATCHES[@]} patch(es) did not apply cleanly:"
    for p in "${FAILED_PATCHES[@]}"; do echo "  - $p"; done
    echo "Reconcile the drifted hunks in these reject files:"
    find llama.cpp -name '*.rej' | sed 's/^/  /'
    echo "  (edit each *.rej into place, then: find llama.cpp -name '*.rej' -delete)"
    exit 1
fi

echo "Patches applied successfully!"
echo "Note: These changes are not committed to the submodule."
echo "To reset the submodule to its clean state, run:"
echo "  cd llama.cpp && git reset --hard && git clean -fdx"
