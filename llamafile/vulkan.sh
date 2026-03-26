#!/bin/bash
#
# Vulkan build script for llamafile (parallel compilation)
#
# This script compiles the GGML Vulkan backend into a shared library.
# Unlike CUDA/ROCm, Vulkan uses standard C++ compilers (g++/clang++),
# not vendor-specific compilers.
#
# Build process:
#   1. Build vulkan-shaders-gen tool (C++17)
#   2. Generate shader C++ files from GLSL compute shaders using glslc
#   3. Compile ggml-vulkan.cpp with generated shaders
#   4. Compile core GGML sources
#   5. Link into ggml-vulkan.so
#
# Requirements:
#   - Vulkan SDK with glslc compiler
#   - C++17 compatible compiler (g++ or clang++)
#   - libvulkan development files
#
# Usage:
#   ./vulkan.sh              # Build with auto-detected parallelism
#   ./vulkan.sh -j16         # Build with 16 parallel jobs
#   ./vulkan.sh --clean      # Clean and rebuild
#   ./vulkan.sh --output /path/to/output.so
#
# Output: ~/ggml-vulkan.so (default)
#

set -e

# Source shared build functions
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/build-functions.sh"

# Parse arguments (sets JOBS, CLEAN)
parse_build_args "$@"

#
# Vulkan-specific configuration
#

# Determine library extension based on platform
# Cosmopolitan's cosmo_dlopen expects .dylib on macOS, .so on Linux
if [ "$(uname -s)" = "Darwin" ]; then
    DSO_EXT="dylib"
else
    DSO_EXT="so"
fi

OUTPUT="${OUTPUT:-${HOME}/ggml-vulkan.${DSO_EXT}}"

# Find compilers
CXX="${CXX:-$(command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null)}"
CC="${CC:-$(command -v gcc 2>/dev/null || command -v clang 2>/dev/null)}"

if [ -z "$CXX" ]; then
    echo "Error: No C++ compiler found (g++ or clang++)"
    exit 1
fi

# Find glslc (Vulkan shader compiler)
GLSLC="${GLSLC:-$(command -v glslc 2>/dev/null)}"
if [ -z "$GLSLC" ]; then
    # Try common Vulkan SDK locations
    if [ -n "$VULKAN_SDK" ] && [ -x "$VULKAN_SDK/bin/glslc" ]; then
        GLSLC="$VULKAN_SDK/bin/glslc"
    elif [ -x "/usr/bin/glslc" ]; then
        GLSLC="/usr/bin/glslc"
    else
        echo "Error: glslc not found. Please install the Vulkan SDK."
        echo "  Linux (Ubuntu/Debian): sudo apt install vulkan-sdk"
        echo "  Linux (Fedora): sudo dnf install vulkan-tools shaderc"
        echo "  Linux (Arch): sudo pacman -S vulkan-devel shaderc"
        echo "  macOS: brew install vulkan-sdk"
        echo "  Or set VULKAN_SDK environment variable"
        exit 1
    fi
fi

# Directory setup
LLAMAFILE_DIR="$SCRIPT_DIR"
LLAMA_CPP_DIR="$SCRIPT_DIR/../llama.cpp"
GGML_VULKAN_DIR="$LLAMA_CPP_DIR/ggml/src/ggml-vulkan"
SHADERS_DIR="$GGML_VULKAN_DIR/vulkan-shaders"

if [ ! -d "$GGML_VULKAN_DIR" ]; then
    echo "Error: Vulkan source directory not found: $GGML_VULKAN_DIR"
    exit 1
fi

# Get version info (sets GGML_VERSION, GGML_COMMIT)
get_ggml_version "$LLAMA_CPP_DIR"

# Build directory
BUILD_DIR="${HOME}/.cache/llamafile-vulkan-build"
setup_build_dir "$BUILD_DIR" "$CLEAN"

# Subdirectories
SHADERS_BUILD_DIR="$BUILD_DIR/shaders"
SPVDIR="$BUILD_DIR/spv"
mkdir -p "$SHADERS_BUILD_DIR" "$SPVDIR"

echo "Building ggml-vulkan.so (parallel)..."
echo "  Version: $GGML_VERSION (commit: $GGML_COMMIT)"
echo "  Source:  $GGML_VULKAN_DIR"
echo "  Output:  $OUTPUT"
echo "  Build:   $BUILD_DIR"
echo "  Jobs:    $JOBS"
echo "  CXX:     $CXX"
echo "  glslc:   $GLSLC"
echo ""

START_TIME=$(date +%s)

#
# Phase 1: Build vulkan-shaders-gen tool
#
echo "Phase 1: Building vulkan-shaders-gen..."

SHADERS_GEN_SRC="$SHADERS_DIR/vulkan-shaders-gen.cpp"
SHADERS_GEN_BIN="$BUILD_DIR/vulkan-shaders-gen"

if [ ! -f "$SHADERS_GEN_BIN" ] || [ "$SHADERS_GEN_SRC" -nt "$SHADERS_GEN_BIN" ]; then
    echo "  Compiling vulkan-shaders-gen.cpp..."
    $CXX -std=c++17 -O2 -o "$SHADERS_GEN_BIN" "$SHADERS_GEN_SRC" -lpthread
else
    echo "  vulkan-shaders-gen is up to date"
fi

#
# Phase 2: Generate shader header (contains declarations)
#
echo ""
echo "Phase 2: Generating shader header..."

SHADERS_HPP="$BUILD_DIR/ggml-vulkan-shaders.hpp"
if [ ! -f "$SHADERS_HPP" ]; then
    echo "  Generating ggml-vulkan-shaders.hpp..."
    "$SHADERS_GEN_BIN" \
        --output-dir "$SPVDIR" \
        --target-hpp "$SHADERS_HPP"
else
    echo "  Shader header already exists"
fi

#
# Phase 3: Compile shaders and generate C++ source files
#
echo ""
echo "Phase 3: Compiling shaders..."

# Collect shader files
SHADER_FILES=$(find "$SHADERS_DIR" -maxdepth 1 -name "*.comp" -type f | sort)
NUM_SHADERS=$(echo "$SHADER_FILES" | wc -l)
echo "  Found $NUM_SHADERS shader files"

count=0
for shader in $SHADER_FILES; do
    count=$((count + 1))
    shader_name=$(basename "$shader")
    shader_cpp="$SHADERS_BUILD_DIR/${shader_name}.cpp"

    # Skip if cpp file is newer than shader
    if [ -f "$shader_cpp" ] && [ "$shader_cpp" -nt "$shader" ]; then
        echo "  [$count/$NUM_SHADERS] Skipping: $shader_name (up to date)"
        continue
    fi

    echo "  [$count/$NUM_SHADERS] Compiling: $shader_name"
    "$SHADERS_GEN_BIN" \
        --glslc "$GLSLC" \
        --source "$shader" \
        --output-dir "$SPVDIR" \
        --target-hpp "$SHADERS_HPP" \
        --target-cpp "$shader_cpp" &

    # Limit parallel jobs
    running=$(jobs -r | wc -l)
    while [ "$running" -ge "$JOBS" ]; do
        sleep 0.1
        running=$(jobs -r | wc -l)
    done
done

echo ""
echo "Waiting for shader compilation to finish..."
wait

SHADER_TIME=$(date +%s)
echo "Shader compilation took $((SHADER_TIME - START_TIME)) seconds"

#
# Phase 4: Compile shader C++ files
#
echo ""
echo "Phase 4: Compiling shader C++ files..."

SHADER_CPP_FILES=$(find "$SHADERS_BUILD_DIR" -name "*.cpp" -type f)
NUM_SHADER_CPPS=$(echo "$SHADER_CPP_FILES" | wc -l)
echo "  Found $NUM_SHADER_CPPS shader C++ files"

CXX_FLAGS="-O2 -fPIC -std=c++17 \
    -I$LLAMA_CPP_DIR/ggml/include \
    -I$LLAMA_CPP_DIR/ggml/src \
    -I$BUILD_DIR \
    -DNDEBUG \
    -DGGML_BUILD=1 \
    -DGGML_SHARED=1 \
    -DGGML_MULTIPLATFORM"

count=0
for src in $SHADER_CPP_FILES; do
    count=$((count + 1))
    base=$(basename "$src" .cpp)
    obj="$BUILD_DIR/shader-${base}.o"

    # Skip if object file is newer than source
    if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
        echo "  [$count/$NUM_SHADER_CPPS] Skipping: $base (up to date)"
        continue
    fi

    echo "  [$count/$NUM_SHADER_CPPS] Compiling: $base"
    $CXX -c $CXX_FLAGS -o "$obj" "$src" &

    # Limit parallel jobs
    running=$(jobs -r | wc -l)
    while [ "$running" -ge "$JOBS" ]; do
        sleep 0.1
        running=$(jobs -r | wc -l)
    done
done

echo ""
echo "Waiting for shader C++ compilation to finish..."
wait

#
# Phase 5: Compile ggml-vulkan.cpp
#
echo ""
echo "Phase 5: Compiling ggml-vulkan.cpp..."

VULKAN_OBJ="$BUILD_DIR/ggml-vulkan.o"
VULKAN_SRC="$GGML_VULKAN_DIR/ggml-vulkan.cpp"

if [ ! -f "$VULKAN_OBJ" ] || [ "$VULKAN_SRC" -nt "$VULKAN_OBJ" ] || [ "$SHADERS_HPP" -nt "$VULKAN_OBJ" ]; then
    echo "  Compiling ggml-vulkan.cpp..."
    $CXX -c $CXX_FLAGS \
        -I$GGML_VULKAN_DIR \
        -o "$VULKAN_OBJ" "$VULKAN_SRC"
else
    echo "  ggml-vulkan.o is up to date"
fi

#
# Phase 6: Compile core GGML sources
#
echo ""
echo "Phase 6: Compiling core GGML sources..."
compile_ggml_core "$LLAMA_CPP_DIR" "$BUILD_DIR"

#
# Phase 7: Link
#
echo ""
echo "Phase 7: Linking..."

OBJ_FILES=$(find "$BUILD_DIR" -name "*.o" -type f | tr '\n' ' ')
NUM_OBJS=$(find "$BUILD_DIR" -name "*.o" -type f | wc -l)
echo "  Linking $NUM_OBJS object files..."

# Platform-specific link flags
LINK_FLAGS="-shared -fPIC"
if [ "$(uname -s)" = "Darwin" ]; then
    # On macOS, set rpath so libvulkan can be found at runtime
    # Check common locations for Vulkan SDK
    if [ -n "$VULKAN_SDK" ] && [ -d "$VULKAN_SDK/lib" ]; then
        VULKAN_LIB_PATH="$VULKAN_SDK/lib"
    elif [ -d "/usr/local/lib" ] && [ -f "/usr/local/lib/libvulkan.dylib" ]; then
        VULKAN_LIB_PATH="/usr/local/lib"
    elif [ -d "/opt/homebrew/lib" ] && [ -f "/opt/homebrew/lib/libvulkan.dylib" ]; then
        VULKAN_LIB_PATH="/opt/homebrew/lib"
    else
        echo "Warning: Could not find libvulkan location for rpath"
        VULKAN_LIB_PATH="/usr/local/lib"
    fi
    LINK_FLAGS="$LINK_FLAGS -Wl,-rpath,$VULKAN_LIB_PATH -L$VULKAN_LIB_PATH"
fi

$CXX $LINK_FLAGS -o "$OUTPUT" $OBJ_FILES -lvulkan -lpthread

# Done
END_TIME=$(date +%s)
echo ""
echo "Total time: $((END_TIME - START_TIME)) seconds"
echo ""
echo "Successfully built: $OUTPUT"
ls -lh "$OUTPUT"
