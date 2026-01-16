#!/bin/bash
#
# CUDA build script for llamafile (parallel compilation)
#
# This script compiles the GGML CUDA backend with TinyBLAS into a shared library.
# Unlike cuda.sh, this version compiles each .cu file separately in parallel,
# which can be significantly faster on multi-core systems.
#
# Usage:
#   ./cuda_parallel.sh              # Build with auto-detected parallelism
#   ./cuda_parallel.sh -j16         # Build with 16 parallel jobs
#   ./cuda_parallel.sh --clean      # Clean and rebuild
#
# Output: ~/ggml-cuda.so
#

set -e

# Default settings
OUTPUT="${HOME}/ggml-cuda.so"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CLEAN=0

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -j*)
            JOBS="${1#-j}"
            ;;
        --clean)
            CLEAN=1
            ;;
        --help)
            echo "Usage: $0 [-jN] [--clean]"
            echo "  -jN      Use N parallel jobs (default: auto-detect)"
            echo "  --clean  Clean build directory before building"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

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

# Create build directory
BUILD_DIR="${HOME}/.cache/llamafile-cuda-build"
if [ "$CLEAN" = "1" ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

echo "Building ggml-cuda.so with TinyBLAS (parallel)..."
echo "  Source: $GGML_CUDA_DIR"
echo "  Output: $OUTPUT"
echo "  Build:  $BUILD_DIR"
echo "  Jobs:   $JOBS"

# Copy TinyBLAS files to build directory
cp "$LLAMAFILE_DIR/tinyblas.h" "$BUILD_DIR/"
cp "$LLAMAFILE_DIR/tinyblas.cu" "$BUILD_DIR/"
cp "$LLAMAFILE_DIR/tinyblas-compat.h" "$BUILD_DIR/"

# Architecture flags for supported GPUs
# sm_75: Turing (RTX 2000 series, Tesla T4)
# sm_80: Ampere (RTX 3000 series, A100)
# sm_86: Ampere (RTX 3000 series mobile)
# sm_89: Ada Lovelace (RTX 4000 series, L40S)
# sm_90: Hopper (H100)
ARCH_FLAGS="\
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90"

# Common compiler flags
COMMON_FLAGS="\
  --use_fast_math \
  --extended-lambda \
  -I$BUILD_DIR \
  -I$LLAMA_CPP_DIR/ggml/include \
  -I$LLAMA_CPP_DIR/ggml/src \
  -I$GGML_CUDA_DIR \
  --forward-unknown-to-host-compiler \
  --compiler-options -fPIC,-O2 \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_MULTIPLATFORM \
  -DGGML_USE_TINYBLAS"

# Collect all CUDA source files
# Start with tinyblas.cu which must be compiled separately
CUDA_SOURCES="$BUILD_DIR/tinyblas.cu"

# Add all GGML CUDA files
for f in "$GGML_CUDA_DIR"/*.cu "$GGML_CUDA_DIR/template-instances"/*.cu; do
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

NUM_SOURCES=$(echo $CUDA_SOURCES | wc -w)
echo "  Sources: $NUM_SOURCES .cu files"
echo ""

echo "Compiling $NUM_SOURCES files with $JOBS parallel jobs..."
echo ""

START_TIME=$(date +%s)

# Compile all files in parallel using background jobs
count=0
total=$NUM_SOURCES
for src in $CUDA_SOURCES; do
    count=$((count + 1))
    base=$(basename "$src" .cu)

    # Create unique name to avoid collisions between main files and template-instances
    if echo "$src" | grep -q "template-instances"; then
        obj="$BUILD_DIR/ti-${base}.o"
    else
        obj="$BUILD_DIR/${base}.o"
    fi

    # Skip if object file is newer than source
    if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
        echo "[$count/$total] Skipping: $base.cu (up to date)"
        continue
    fi

    echo "[$count/$total] Compiling: $base.cu"
    $NVCC -c $ARCH_FLAGS $COMMON_FLAGS -o "$obj" "$src" &

    # Limit parallel jobs by waiting when we hit the limit
    running=$(jobs -r | wc -l)
    while [ "$running" -ge "$JOBS" ]; do
        sleep 0.1
        running=$(jobs -r | wc -l)
    done
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for remaining compilations to finish..."
wait

COMPILE_TIME=$(date +%s)
echo "Compilation took $((COMPILE_TIME - START_TIME)) seconds"
echo ""

# Compile core GGML sources (C/C++ files, not CUDA)
# These are needed to make the DSO self-contained since cosmo_dlopen()
# cannot resolve symbols from the parent process
echo "Compiling core GGML sources..."
HOST_FLAGS="-fPIC -O2 -DNDEBUG -DGGML_BUILD=1 -DGGML_SHARED=1 -DGGML_MULTIPLATFORM -DGGML_VERSION=\\\"0.9.4\\\" -DGGML_COMMIT=\\\"unknown\\\" -I$LLAMA_CPP_DIR/ggml/include -I$LLAMA_CPP_DIR/ggml/src"

for src in $GGML_CORE_SOURCES; do
    base=$(basename "$src")
    ext="${base##*.}"
    name="${base%.*}"
    obj="$BUILD_DIR/ggml-core-${name}.o"

    # Skip if object file is newer than source
    if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
        echo "  Skipping: $base (up to date)"
        continue
    fi

    echo "  Compiling: $base"
    if [ "$ext" = "c" ]; then
        gcc -c $HOST_FLAGS -o "$obj" "$src"
    else
        g++ -c $HOST_FLAGS -std=c++17 -o "$obj" "$src"
    fi
done
echo ""

echo "Linking..."

# Collect all object files
OBJ_FILES=$(find "$BUILD_DIR" -name "*.o" -type f | tr '\n' ' ')
NUM_OBJS=$(find "$BUILD_DIR" -name "*.o" -type f | wc -l)
echo "  Linking $NUM_OBJS object files..."

# Link into shared library
$NVCC --shared $ARCH_FLAGS -o "$OUTPUT" $OBJ_FILES -lcuda

END_TIME=$(date +%s)
echo ""
echo "Total time: $((END_TIME - START_TIME)) seconds"
echo ""
echo "Successfully built: $OUTPUT"
ls -lh "$OUTPUT"
