#!/bin/sh
#
# CUDA build script for llamafile
#
# This script compiles the GGML CUDA backend with TinyBLAS into a shared library.
# The resulting ggml-cuda.so can be:
#   1. Bundled into llamafile executables using zipalign
#   2. Used for runtime compilation when nvcc is available
#
# Usage:
#   ./cuda.sh              # Build with default settings
#   ./cuda.sh --help       # Show help
#
# Output: ~/ggml-cuda.so
#

set -e

# Default settings
OUTPUT="${HOME}/ggml-cuda.so"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"

# Check for nvcc
if [ ! -x "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    echo "Please install CUDA toolkit or set CUDA_PATH"
    exit 1
fi

# Get script directory (where llamafile sources are)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMAFILE_DIR="$SCRIPT_DIR"
LLAMA_CPP_DIR="$SCRIPT_DIR/../llama.cpp"
GGML_CUDA_DIR="$LLAMA_CPP_DIR/ggml/src/ggml-cuda"

# Check that source directories exist
if [ ! -d "$GGML_CUDA_DIR" ]; then
    echo "Error: CUDA source directory not found: $GGML_CUDA_DIR"
    exit 1
fi

# Create temporary build directory
TMP=$(mktemp -d) || exit 1
trap "rm -rf $TMP" EXIT

echo "Building ggml-cuda.so with TinyBLAS..."
echo "  Source: $GGML_CUDA_DIR"
echo "  Output: $OUTPUT"
echo "  Temp:   $TMP"

# Copy TinyBLAS files to temp directory
cp "$LLAMAFILE_DIR/tinyblas.h" "$TMP/"
cp "$LLAMAFILE_DIR/tinyblas.cu" "$TMP/"
cp "$LLAMAFILE_DIR/tinyblas-compat.h" "$TMP/"

# Collect all CUDA source files
# Start with tinyblas.cu which provides BLAS implementations
CUDA_SOURCES="$TMP/tinyblas.cu"

# Main CUDA files (not template-instances)
for f in "$GGML_CUDA_DIR"/*.cu; do
    if [ -f "$f" ]; then
        CUDA_SOURCES="$CUDA_SOURCES $f"
    fi
done

# Template instance files
for f in "$GGML_CUDA_DIR/template-instances"/*.cu; do
    if [ -f "$f" ]; then
        CUDA_SOURCES="$CUDA_SOURCES $f"
    fi
done

# Core GGML source files (make DSO self-contained)
# These are needed because cosmo_dlopen() cannot resolve symbols from the parent process
GGML_CORE_SOURCES="\
  $LLAMA_CPP_DIR/ggml/src/ggml.c \
  $LLAMA_CPP_DIR/ggml/src/ggml-alloc.c \
  $LLAMA_CPP_DIR/ggml/src/ggml-backend.cpp \
  $LLAMA_CPP_DIR/ggml/src/ggml-quants.c \
  $LLAMA_CPP_DIR/ggml/src/ggml-threading.cpp"

# Count sources
NUM_SOURCES=$(echo $CUDA_SOURCES | wc -w)
echo "  Sources: $NUM_SOURCES .cu files"

# Architecture flags for supported GPUs
# sm_75: Turing (RTX 2000 series, Tesla T4)
# sm_80: Ampere (RTX 3000 series, A100)
# sm_86: Ampere (RTX 3000 series mobile)
# sm_89: Ada Lovelace (RTX 4000 series, L40S)
# sm_90: Hopper (H100)
# Note: sm_60/sm_70 deprecated in CUDA 12.x
ARCH_FLAGS="\
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90"

# Include paths
INCLUDE_FLAGS="\
  -I$TMP \
  -I$LLAMA_CPP_DIR/ggml/include \
  -I$LLAMA_CPP_DIR/ggml/src \
  -I$GGML_CUDA_DIR"

# Common CUDA compiler flags
COMMON_FLAGS="\
  --use_fast_math \
  --extended-lambda \
  $INCLUDE_FLAGS \
  --forward-unknown-to-host-compiler \
  --compiler-options -fPIC,-O2 \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_MULTIPLATFORM \
  -DGGML_USE_TINYBLAS"

# Compile CUDA sources
echo "Compiling CUDA sources with nvcc..."
OBJ_FILES=""
count=0
for src in $CUDA_SOURCES; do
    count=$((count + 1))
    base=$(basename "$src" .cu)

    # Create unique name to avoid collisions between main files and template-instances
    if echo "$src" | grep -q "template-instances"; then
        obj="$TMP/ti-${base}.o"
    else
        obj="$TMP/${base}.o"
    fi

    echo "  [$count/$NUM_SOURCES] $base.cu"
    $NVCC -c $ARCH_FLAGS $COMMON_FLAGS -o "$obj" "$src"
    OBJ_FILES="$OBJ_FILES $obj"
done

# Compile core GGML sources (C/C++ files, not CUDA)
# These are needed to make the DSO self-contained since cosmo_dlopen()
# cannot resolve symbols from the parent process
echo ""
echo "Compiling core GGML sources..."
HOST_FLAGS="-fPIC -O2 -DNDEBUG -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_MULTIPLATFORM -DGGML_VERSION=\\\"0.9.4\\\" -DGGML_COMMIT=\\\"unknown\\\" -I$LLAMA_CPP_DIR/ggml/include -I$LLAMA_CPP_DIR/ggml/src"

for src in $GGML_CORE_SOURCES; do
    base=$(basename "$src")
    ext="${base##*.}"
    name="${base%.*}"
    obj="$TMP/ggml-core-${name}.o"

    echo "  Compiling: $base"
    if [ "$ext" = "c" ]; then
        gcc -c $HOST_FLAGS -o "$obj" "$src"
    else
        g++ -c $HOST_FLAGS -std=c++17 -o "$obj" "$src"
    fi
    OBJ_FILES="$OBJ_FILES $obj"
done

# Link into shared library
echo ""
echo "Linking..."
NUM_OBJS=$(echo $OBJ_FILES | wc -w)
echo "  Linking $NUM_OBJS object files..."
$NVCC --shared $ARCH_FLAGS -o "$OUTPUT" $OBJ_FILES -lcuda

echo ""
echo "Successfully built: $OUTPUT"
ls -lh "$OUTPUT"
