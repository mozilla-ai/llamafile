#!/bin/sh
#
# ROCm build script for llamafile
#
# This script compiles the GGML CUDA/HIP backend with TinyBLAS into a shared library
# for AMD GPUs using ROCm/HIP.
#
# The resulting ggml-rocm.so can be:
#   1. Bundled into llamafile executables using zipalign
#   2. Used for runtime compilation when hipcc is available
#
# Usage:
#   ./rocm.sh              # Build with default settings
#   ./rocm.sh --help       # Show help
#
# Output: ~/ggml-rocm.so
#

set -e

# Default settings
OUTPUT="${HOME}/ggml-rocm.so"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"

# Check for hipcc
if [ ! -x "$HIPCC" ]; then
    echo "Error: hipcc not found at $HIPCC"
    echo "Please install ROCm or set ROCM_PATH"
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

echo "Building ggml-rocm.so with TinyBLAS..."
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

# AMD GPU architecture targets
# gfx906:  Vega 20 (Radeon VII, MI50)
# gfx1030: RDNA2 (RX 6900 XT, RX 6800 series)
# gfx1031: RDNA2 (RX 6700 series)
# gfx1032: RDNA2 (RX 6600 series)
# gfx1100: RDNA3 (RX 7900 XTX, RX 7900 XT)
# gfx1101: RDNA3 (RX 7800 series)
# gfx1102: RDNA3 (RX 7600 series)
# gfx1103: RDNA3 (RX 7000 mobile)
AMD_TARGETS="gfx906,gfx1030,gfx1031,gfx1032,gfx1100,gfx1101,gfx1102,gfx1103"

# Include paths
INCLUDE_FLAGS="\
  -I$TMP \
  -I$LLAMA_CPP_DIR/ggml/include \
  -I$LLAMA_CPP_DIR/ggml/src \
  -I$GGML_CUDA_DIR"

# Common HIP compiler flags
COMMON_FLAGS="\
  -O2 \
  -fPIC \
  $INCLUDE_FLAGS \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_MULTIPLATFORM \
  -DGGML_USE_HIP=1 \
  -DGGML_USE_TINYBLAS=1 \
  -Wno-return-type \
  -Wno-unused-result \
  --amdgpu-target=$AMD_TARGETS"

# Compile CUDA/HIP sources
echo "Compiling HIP sources with hipcc..."
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
    $HIPCC -c $COMMON_FLAGS -o "$obj" "$src"
    OBJ_FILES="$OBJ_FILES $obj"
done

# Compile core GGML sources (C/C++ files, not HIP)
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
$HIPCC -shared -fPIC --amdgpu-target=$AMD_TARGETS -o "$OUTPUT" $OBJ_FILES

echo ""
echo "Successfully built: $OUTPUT"
ls -lh "$OUTPUT"
