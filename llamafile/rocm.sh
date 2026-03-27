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
# ROCm build script for llamafile (parallel compilation)
#
# This script compiles the GGML CUDA/HIP backend with TinyBLAS into a shared library
# for AMD GPUs using ROCm/HIP.
#
# Usage:
#   ./rocm.sh              # Build with auto-detected parallelism
#   ./rocm.sh -j16         # Build with 16 parallel jobs
#   ./rocm.sh --clean      # Clean and rebuild
#   ./rocm.sh --output /path/to/output.so
#
# Output: ~/ggml-rocm.so (default)
#

set -e

# Source shared build functions
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/build-functions.sh"

# Parse arguments (sets JOBS, CLEAN)
parse_build_args "$@"

#
# ROCm/HIP specific configuration
#

OUTPUT="${OUTPUT:-${HOME}/ggml-rocm.so}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/bin/hipcc"

# Check for hipcc
if [ ! -x "$HIPCC" ]; then
    echo "Error: hipcc not found at $HIPCC"
    echo "Please install ROCm or set ROCM_PATH"
    exit 1
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

# Build directory
BUILD_DIR="${HOME}/.cache/llamafile-rocm-build"
setup_build_dir "$BUILD_DIR" "$CLEAN"

echo "Building ggml-rocm.so with TinyBLAS (parallel)..."
echo "  Version: $GGML_VERSION (commit: $GGML_COMMIT)"
echo "  Source: $GGML_CUDA_DIR"
echo "  Output: $OUTPUT"
echo "  Build:  $BUILD_DIR"
echo "  Jobs:   $JOBS"

# Copy TinyBLAS files to build directory
cp "$LLAMAFILE_DIR/tinyblas.h" "$BUILD_DIR/"
cp "$LLAMAFILE_DIR/tinyblas.cu" "$BUILD_DIR/"
cp "$LLAMAFILE_DIR/tinyblas-compat.h" "$BUILD_DIR/"

# AMD GPU architecture targets
# gfx906:  Vega 20 (Radeon VII, MI50)
# gfx1030: RDNA2 (RX 6900 XT, RX 6800 series)
# gfx1031: RDNA2 (RX 6700 series)
# gfx1032: RDNA2 (RX 6600 series)
# gfx1100: RDNA3 (RX 7900 XTX, RX 7900 XT)
# gfx1101: RDNA3 (RX 7800 series)
# gfx1102: RDNA3 (RX 7600 series)
# gfx1103: RDNA3 (RX 7000 mobile)
ARCH_FLAGS="\
  --offload-arch=gfx906 \
  --offload-arch=gfx1030 \
  --offload-arch=gfx1031 \
  --offload-arch=gfx1032 \
  --offload-arch=gfx1100 \
  --offload-arch=gfx1101 \
  --offload-arch=gfx1102 \
  --offload-arch=gfx1103"

# HIP compiler flags
COMMON_FLAGS="\
  -O2 \
  -fPIC \
  -I$BUILD_DIR \
  -I$LLAMA_CPP_DIR/ggml/include \
  -I$LLAMA_CPP_DIR/ggml/src \
  -I$GGML_CUDA_DIR \
  -I$ROCM_PATH/include \
  -DNDEBUG \
  -DGGML_BUILD=1 \
  -DGGML_SHARED=1 \
  -DGGML_MULTIPLATFORM \
  -DGGML_USE_HIP=1 \
  -DGGML_USE_TINYBLAS=1 \
  -Wno-return-type \
  -Wno-unused-result"

# Collect sources (TinyBLAS + GGML CUDA)
collect_gpu_sources "$GGML_CUDA_DIR" "$BUILD_DIR/tinyblas.cu"
echo "  Sources: $NUM_SOURCES .cu files"
echo ""

START_TIME=$(date +%s)

# Compile GPU sources
compile_gpu_sources_parallel "$HIPCC" "$ARCH_FLAGS" "$COMMON_FLAGS" "$BUILD_DIR" "$JOBS"

COMPILE_TIME=$(date +%s)
echo "Compilation took $((COMPILE_TIME - START_TIME)) seconds"
echo ""

# Compile core GGML sources
compile_ggml_core "$LLAMA_CPP_DIR" "$BUILD_DIR"

# Link
link_shared_library "$HIPCC" "-shared -fPIC" "$ARCH_FLAGS" "$BUILD_DIR" "$OUTPUT" ""

# Done
print_build_summary "$OUTPUT" "$START_TIME"
