#!/bin/bash
# -*- mode:sh;indent-tabs-mode:nil;tab-width:4;coding:utf-8 -*-
# vi: set et ft=sh ts=4 sts=4 sw=4 fenc=utf-8 :vi
#
# Copyright 2024 Mozilla Foundation
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
# CUDA build script for llamafile (parallel compilation)
#
# This script compiles the GGML CUDA backend into a shared library.
# By default it uses TinyBLAS, but can optionally use NVIDIA cuBLAS.
#
# Usage:
#   ./cuda.sh              # Build with TinyBLAS (default)
#   ./cuda.sh --cublas     # Build with NVIDIA cuBLAS
#   ./cuda.sh -j16         # Build with 16 parallel jobs
#   ./cuda.sh --clean      # Clean and rebuild
#   ./cuda.sh --output /path/to/output.so
#
# Output: ~/ggml-cuda.so (default)
#

set -e

# Source shared build functions
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/build-functions.sh"

#
# Parse arguments (handle --cublas locally, delegate rest to shared function)
#

USE_CUBLAS=0
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --cublas)
            USE_CUBLAS=1
            ;;
        --help)
            echo "Usage: $0 [-jN] [--clean] [--cublas] [--output PATH]"
            echo "  -jN       Use N parallel jobs (default: auto-detect)"
            echo "  --clean   Clean build directory before building"
            echo "  --cublas  Use NVIDIA cuBLAS instead of TinyBLAS"
            echo "  --output  Output path for shared library"
            exit 0
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# Parse common arguments (sets JOBS, CLEAN)
parse_build_args "${ARGS[@]}"

#
# CUDA-specific configuration
#

OUTPUT="${OUTPUT:-${HOME}/ggml-cuda.so}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"

# Check for nvcc
if [ ! -x "$NVCC" ]; then
    echo "Error: nvcc not found at $NVCC"
    echo "Please install CUDA toolkit or set CUDA_PATH"
    exit 1
fi

# Check for cuBLAS if requested
if [ "$USE_CUBLAS" = "1" ]; then
    if [ ! -f "$CUDA_PATH/lib64/libcublas.so" ] && [ ! -f "$CUDA_PATH/lib/libcublas.so" ]; then
        echo "Warning: libcublas.so not found in $CUDA_PATH/lib64 or $CUDA_PATH/lib"
        echo "cuBLAS is required at runtime for this build"
    fi
fi

# Directory setup
LLAMAFILE_DIR="$SCRIPT_DIR"
LLAMA_CPP_DIR="$SCRIPT_DIR/../llama.cpp"
GGML_CUDA_DIR="$LLAMA_CPP_DIR/ggml/src/ggml-cuda"

if [ ! -d "$GGML_CUDA_DIR" ]; then
    echo "Error: CUDA source directory not found: $GGML_CUDA_DIR"
    exit 1
fi

# Get version info (sets GGML_VERSION, GGML_COMMIT)
get_ggml_version "$LLAMA_CPP_DIR"

# Build directory (separate for TinyBLAS vs cuBLAS to avoid conflicts)
if [ "$USE_CUBLAS" = "1" ]; then
    BUILD_DIR="${HOME}/.cache/llamafile-cuda-cublas-build"
    BLAS_NAME="cuBLAS"
    BLAS_DEFINE="-DGGML_USE_CUBLAS"
    EXTRA_INCLUDES=""
    EXTRA_SOURCES=""
    LINK_LIBS="-lcuda -lcublas"
else
    BUILD_DIR="${HOME}/.cache/llamafile-cuda-build"
    BLAS_NAME="TinyBLAS"
    BLAS_DEFINE="-DGGML_USE_TINYBLAS"
    EXTRA_INCLUDES="-I$BUILD_DIR"
    EXTRA_SOURCES="$BUILD_DIR/tinyblas.cu"
    LINK_LIBS="-lcuda"
fi

setup_build_dir "$BUILD_DIR" "$CLEAN"

echo "Building ggml-cuda.so with $BLAS_NAME (parallel)..."
echo "  Version: $GGML_VERSION (commit: $GGML_COMMIT)"
echo "  Source: $GGML_CUDA_DIR"
echo "  Output: $OUTPUT"
echo "  Build:  $BUILD_DIR"
echo "  Jobs:   $JOBS"

# Copy TinyBLAS files if needed
if [ "$USE_CUBLAS" = "0" ]; then
    cp "$LLAMAFILE_DIR/tinyblas.h" "$BUILD_DIR/"
    cp "$LLAMAFILE_DIR/tinyblas.cu" "$BUILD_DIR/"
    cp "$LLAMAFILE_DIR/tinyblas-compat.h" "$BUILD_DIR/"
fi

# NVIDIA GPU architecture targets
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

# Enabling Blackwell native codegen requires CUDA 13.x
CUDA_VERSION=$("$NVCC" --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
HOST_ARCH=$(uname -m)

# Blackwell aarch64 non-server platforms (sm_110: Jetson Thor & family, sm_121: DGX Spark GB10)
if [ "$HOST_ARCH" = "aarch64" ] && [ "${CUDA_VERSION%%.*}" = "13" ]; then
    ARCH_FLAGS="\
  -gencode arch=compute_110f,code=sm_110f \
  -gencode arch=compute_121a,code=sm_121a \
  --compress-mode=size"

# Blackwell GPUs: CUDA 13.x append sm_120 family GPU support (RTX 5000 series, RTX PRO Blackwell)
elif [ "${CUDA_VERSION%%.*}" = "13" ]; then
    ARCH_FLAGS="$ARCH_FLAGS \
  -gencode arch=compute_120f,code=sm_120f \
  --compress-mode=size"
fi

# NVCC compiler flags
COMMON_FLAGS="\
  --use_fast_math \
  --extended-lambda \
  $EXTRA_INCLUDES \
  -I$LLAMA_CPP_DIR/ggml/include \
  -I$LLAMA_CPP_DIR/ggml/src \
  -I$GGML_CUDA_DIR \
  --forward-unknown-to-host-compiler \
  --compiler-options -fPIC,-O2 \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_MULTIPLATFORM \
  $BLAS_DEFINE"

# Collect sources
collect_gpu_sources "$GGML_CUDA_DIR" "$EXTRA_SOURCES"
echo "  Sources: $NUM_SOURCES .cu files"
echo ""

START_TIME=$(date +%s)

# Compile GPU sources
compile_gpu_sources_parallel "$NVCC" "$ARCH_FLAGS" "$COMMON_FLAGS" "$BUILD_DIR" "$JOBS"

COMPILE_TIME=$(date +%s)
echo "Compilation took $((COMPILE_TIME - START_TIME)) seconds"
echo ""

# Compile core GGML sources
compile_ggml_core "$LLAMA_CPP_DIR" "$BUILD_DIR"

# Link
link_shared_library "$NVCC" "--shared" "$ARCH_FLAGS" "$BUILD_DIR" "$OUTPUT" "$LINK_LIBS"

# Done
if [ "$USE_CUBLAS" = "1" ]; then
    print_build_summary "$OUTPUT" "$START_TIME" "Note: This library requires libcublas.so at runtime"
else
    print_build_summary "$OUTPUT" "$START_TIME"
fi
