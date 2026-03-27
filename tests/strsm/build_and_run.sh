#!/bin/bash
# -*- mode:sh;indent-tabs-mode:nil;tab-width:4;coding:utf-8 -*-
# vi: set et ft=sh ts=4 sts=4 sw=4 fenc=utf-8 :vi
#
# Copyright 2026 Mozilla.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build and run STRSM batched test: TinyBLAS vs cuBLAS
#
# Usage:
#   bash tests/strsm/build_and_run.sh              # Build and run
#   bash tests/strsm/build_and_run.sh --build-only  # Build only
#   bash tests/strsm/build_and_run.sh --clean        # Clean and rebuild
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMAFILE_DIR="$REPO_DIR/llamafile"
OUTPUT_DIR="$REPO_DIR/o/tests/strsm"
OUTPUT="$OUTPUT_DIR/strsm_test"

CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"

BUILD_ONLY=0
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --build-only)
            BUILD_ONLY=1
            ;;
        --clean)
            CLEAN=1
            ;;
        --help)
            echo "Usage: $0 [--build-only] [--clean]"
            echo "  --build-only  Build without running"
            echo "  --clean       Clean and rebuild"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Check nvcc
if [ ! -x "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    echo "Install CUDA toolkit or set CUDA_PATH"
    exit 1
fi

# Check cublas
CUBLAS_LIB=""
for dir in "$CUDA_PATH/lib64" "$CUDA_PATH/lib"; do
    if [ -f "$dir/libcublas.so" ]; then
        CUBLAS_LIB="$dir"
        break
    fi
done
if [ -z "$CUBLAS_LIB" ]; then
    echo "Error: libcublas.so not found in $CUDA_PATH/lib64 or $CUDA_PATH/lib"
    exit 1
fi

# Auto-detect GPU architecture
detect_gpu_arch() {
    if command -v nvidia-smi &>/dev/null; then
        # Get compute capability from nvidia-smi
        local cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ -n "$cc" ] && [ "$cc" -gt 0 ] 2>/dev/null; then
            echo "$cc"
            return
        fi
    fi
    # Default: compile for common architectures
    echo ""
}

GPU_ARCH=$(detect_gpu_arch)
if [ -n "$GPU_ARCH" ]; then
    ARCH_FLAGS="-gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}"
    echo "Detected GPU: sm_${GPU_ARCH}"
else
    ARCH_FLAGS="-gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_80,code=sm_80 \
                -gencode arch=compute_86,code=sm_86 \
                -gencode arch=compute_89,code=sm_89 \
                -gencode arch=compute_90,code=sm_90"
    echo "No GPU detected, compiling for sm_75/80/86/89/90"
fi

# Clean
if [ "$CLEAN" = "1" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "Cleaning..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# Copy tinyblas files to build dir to avoid include path conflicts.
# The llamafile/ directory has its own string.h that clashes with system headers
# when added via -I. This is the same approach used by cuda.sh.
cp "$LLAMAFILE_DIR/tinyblas.h" "$OUTPUT_DIR/"
cp "$LLAMAFILE_DIR/tinyblas.cu" "$OUTPUT_DIR/"

# Build
echo "Building strsm_test..."
echo "  Sources: strsm_test.cu + tinyblas.cu"
echo "  Output:  $OUTPUT"

$NVCC \
    "$SCRIPT_DIR/strsm_test.cu" \
    "$OUTPUT_DIR/tinyblas.cu" \
    -I "$OUTPUT_DIR" \
    -o "$OUTPUT" \
    $ARCH_FLAGS \
    -lcublas -lcuda \
    --use_fast_math \
    --extended-lambda \
    -O2 \
    -std=c++17

echo "Build successful: $OUTPUT"
echo ""

# Run
if [ "$BUILD_ONLY" = "0" ]; then
    echo "Running..."
    echo ""
    "$OUTPUT"
fi
