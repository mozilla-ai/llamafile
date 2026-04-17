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
# Shared build functions for llamafile GPU backends
#
# This file contains common functions used by cuda.sh and rocm.sh
# to reduce code duplication while keeping each script's toolchain-specific
# configuration clear and readable.
#
# Usage: source this file from a build script, then call the functions
#

# Parse common command-line arguments
# Sets: JOBS, CLEAN, OUTPUT (if --output provided)
# Args: all script arguments ($@)
parse_build_args() {
    JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    CLEAN=0

    while [ $# -gt 0 ]; do
        case "$1" in
            -j*)
                JOBS="${1#-j}"
                ;;
            --clean)
                CLEAN=1
                ;;
            --output)
                shift
                OUTPUT="$1"
                ;;
            --output=*)
                OUTPUT="${1#--output=}"
                ;;
            --help)
                echo "Usage: $0 [-jN] [--clean] [--output PATH]"
                echo "  -jN       Use N parallel jobs (default: auto-detect)"
                echo "  --clean   Clean build directory before building"
                echo "  --output  Output path for shared library"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done
}

# Extract GGML version from CMakeLists.txt or environment
# Sets: GGML_VERSION, GGML_COMMIT
# Args: $1 = LLAMA_CPP_DIR
get_ggml_version() {
    local llama_cpp_dir="$1"

    if [ -z "$GGML_VERSION" ]; then
        GGML_VERSION_MAJOR=$(grep 'set(GGML_VERSION_MAJOR' "$llama_cpp_dir/ggml/CMakeLists.txt" 2>/dev/null | sed 's/[^0-9]*//g')
        GGML_VERSION_MINOR=$(grep 'set(GGML_VERSION_MINOR' "$llama_cpp_dir/ggml/CMakeLists.txt" 2>/dev/null | sed 's/[^0-9]*//g')
        GGML_VERSION_PATCH=$(grep 'set(GGML_VERSION_PATCH' "$llama_cpp_dir/ggml/CMakeLists.txt" 2>/dev/null | sed 's/[^0-9]*//g')
        GGML_VERSION="${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}"
        if ! echo "$GGML_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
            echo "Error: Invalid GGML version format: '$GGML_VERSION'"
            exit 1
        fi
    fi
    if [ -z "$GGML_COMMIT" ]; then
        GGML_COMMIT=$(cd "$llama_cpp_dir/ggml" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    fi
}

# Setup and clean build directory
# Args: $1 = BUILD_DIR, $2 = CLEAN (0 or 1)
setup_build_dir() {
    local build_dir="$1"
    local clean="$2"

    if [ "$clean" = "1" ] && [ -d "$build_dir" ]; then
        echo "Cleaning build directory..."
        rm -rf "$build_dir"
    fi
    mkdir -p "$build_dir"
}

# Collect CUDA/HIP source files
# Sets: CUDA_SOURCES, NUM_SOURCES
# Args: $1 = GGML_CUDA_DIR, $2 = extra sources (optional, e.g., tinyblas.cu path)
collect_gpu_sources() {
    local ggml_cuda_dir="$1"
    local extra_sources="$2"

    CUDA_SOURCES="$extra_sources"

    for f in "$ggml_cuda_dir"/*.cu "$ggml_cuda_dir/template-instances"/*.cu; do
        if [ -f "$f" ]; then
            CUDA_SOURCES="$CUDA_SOURCES $f"
        fi
    done

    NUM_SOURCES=$(echo $CUDA_SOURCES | wc -w)
}

# Compile GPU sources in parallel
# Args: $1 = compiler, $2 = arch_flags, $3 = common_flags, $4 = build_dir, $5 = jobs
compile_gpu_sources_parallel() {
    local compiler="$1"
    local arch_flags="$2"
    local common_flags="$3"
    local build_dir="$4"
    local jobs="$5"

    echo "Compiling $NUM_SOURCES files with $jobs parallel jobs..."
    echo ""

    local count=0
    local total=$NUM_SOURCES

    for src in $CUDA_SOURCES; do
        count=$((count + 1))
        local base=$(basename "$src" .cu)

        # Create unique name to avoid collisions between main files and template-instances
        local obj
        if echo "$src" | grep -q "template-instances"; then
            obj="$build_dir/ti-${base}.o"
        else
            obj="$build_dir/${base}.o"
        fi

        # Skip if object file is newer than source
        if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
            echo "[$count/$total] Skipping: $base.cu (up to date)"
            continue
        fi

        echo "[$count/$total] Compiling: $base.cu"
        $compiler -c $arch_flags $common_flags -o "$obj" "$src" &

        # Limit parallel jobs by waiting when we hit the limit
        local running=$(jobs -r | wc -l)
        while [ "$running" -ge "$jobs" ]; do
            sleep 0.1
            running=$(jobs -r | wc -l)
        done
    done

    echo ""
    echo "Waiting for remaining compilations to finish..."
    wait
}

# Compile core GGML C/C++ sources
# Args: $1 = LLAMA_CPP_DIR, $2 = BUILD_DIR
compile_ggml_core() {
    local llama_cpp_dir="$1"
    local build_dir="$2"

    local ggml_core_sources="\
        $llama_cpp_dir/ggml/src/ggml.c \
        $llama_cpp_dir/ggml/src/ggml-alloc.c \
        $llama_cpp_dir/ggml/src/ggml-backend.cpp \
        $llama_cpp_dir/ggml/src/ggml-quants.c \
        $llama_cpp_dir/ggml/src/ggml-threading.cpp"

    echo "Compiling core GGML sources..."

    local host_flags=(
        -fPIC -O2 -DNDEBUG
        -DGGML_BUILD=1
        -DGGML_SHARED=1
        -DGGML_MULTIPLATFORM
        "-DGGML_VERSION=\"$GGML_VERSION\""
        "-DGGML_COMMIT=\"$GGML_COMMIT\""
        -I"$llama_cpp_dir/ggml/include"
        -I"$llama_cpp_dir/ggml/src"
    )

    for src in $ggml_core_sources; do
        local base=$(basename "$src")
        local ext="${base##*.}"
        local name="${base%.*}"
        local obj="$build_dir/ggml-core-${name}.o"

        # Skip if object file is newer than source
        if [ -f "$obj" ] && [ "$obj" -nt "$src" ]; then
            echo "  Skipping: $base (up to date)"
            continue
        fi

        echo "  Compiling: $base"
        if [ "$ext" = "c" ]; then
            gcc -c "${host_flags[@]}" -o "$obj" "$src"
        else
            g++ -c "${host_flags[@]}" -std=c++17 -o "$obj" "$src"
        fi
    done
    echo ""
}

# Link object files into shared library
# Args: $1 = linker command, $2 = linker_flags (e.g., "--shared" or "-shared -fPIC")
#       $3 = arch_flags, $4 = build_dir, $5 = output, $6 = extra_libs
#
# On Linux, the C++ runtime (libstdc++, libgcc_s) is linked statically so the
# shipped .so does not carry versioned GLIBCXX symbol requirements from the
# build host. Without this, users on distros with an older libstdc++ than the
# build machine (e.g. Pop!_OS 22.04 / Ubuntu 22.04 with GCC 12) fail to load
# the library with "GLIBCXX_3.4.xx not found". Windows achieves the same via
# /MT; Darwin uses libc++ and is unaffected.
link_shared_library() {
    local linker="$1"
    local linker_flags="$2"
    local arch_flags="$3"
    local build_dir="$4"
    local output="$5"
    local extra_libs="$6"

    local static_cxx_flags=""
    if [ "$(uname -s)" = "Linux" ]; then
        case "$(basename "$linker")" in
            nvcc)
                # nvcc requires -Xcompiler to forward flags to the host driver.
                static_cxx_flags="-Xcompiler=-static-libstdc++,-static-libgcc"
                ;;
            *)
                # hipcc, g++, clang++ accept these driver flags directly
                static_cxx_flags="-static-libstdc++ -static-libgcc"
                ;;
        esac
    fi

    echo "Linking..."

    local obj_files=$(find "$build_dir" -name "*.o" -type f | tr '\n' ' ')
    local num_objs=$(find "$build_dir" -name "*.o" -type f | wc -l)
    echo "  Linking $num_objs object files..."

    $linker $linker_flags $arch_flags $static_cxx_flags -o "$output" $obj_files $extra_libs
}

# Print build summary
# Args: $1 = output file, $2 = start_time, $3 = optional note
print_build_summary() {
    local output="$1"
    local start_time="$2"
    local note="$3"

    local end_time=$(date +%s)
    echo ""
    echo "Total time: $((end_time - start_time)) seconds"
    echo ""
    echo "Successfully built: $output"
    if [ -n "$note" ]; then
        echo "$note"
    fi
    ls -lh "$output"
}
