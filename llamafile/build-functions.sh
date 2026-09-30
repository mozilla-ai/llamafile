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

# Collect CUDA/HIP source files with selective template inclusion
# Sets: CUDA_SOURCES, NUM_SOURCES
# Args: $1 = GGML_CUDA_DIR
#       $2 = caller-supplied sources prepended to the list (e.g., tinyblas.cu
#            for the default TinyBLAS build; empty for the --cublas build)
#       $3 = NO_IQ_QUANTS (optional, "1" to exclude IQ quant MMQ templates)
#       $4 = FA_ALL_QUANTS (optional, "1" to compile every fattn-vec quant combo
#            instead of the default four; mirrors upstream's
#            GGML_CUDA_FA_QUANTS=all)
# Sets: CUDA_SOURCES, NUM_SOURCES, CUDA_FA_DEFINES (append the last to the
#       compiler flags -- fattn.cu does not compile without it)
# The K-V types fattn.cu knows about, and the combinations compiled by default.
# Both mirror ggml/CMakeLists.txt's GGML_CUDA_FA_QUANTS default and the FA_TYPES
# list in ggml/cmake/common.cmake -- keep in sync on a llama.cpp bump.
FA_TYPES="q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16"
FA_DEFAULT_COMBINATIONS="q4_0-q4_0 q8_0-q8_0 f16-f16 bf16-bf16"

collect_gpu_sources() {
    local ggml_cuda_dir="$1"
    local caller_sources="$2"
    local no_iq_quants="${3:-0}"
    local fa_all_quants="${4:-0}"

    CUDA_SOURCES="$caller_sources"

    # 1. Main CUDA sources (always included)
    for f in "$ggml_cuda_dir"/*.cu; do
        [ -f "$f" ] && CUDA_SOURCES="$CUDA_SOURCES $f"
    done

    local ti_dir="$ggml_cuda_dir/template-instances"

    # 2. fattn-mma and fattn-tile instances (always included)
    for f in "$ti_dir"/fattn-mma-*.cu "$ti_dir"/fattn-tile-*.cu; do
        [ -f "$f" ] && CUDA_SOURCES="$CUDA_SOURCES $f"
    done

    # 3. fattn-vec: since b11100 fattn.cu gates every K-V combination on a
    #    GGML_CUDA_FA_<K>_<V> macro that must be defined to 0 or 1 for *all* of
    #    them (upstream's ggml_cuda_fattn_vec_instances() in
    #    ggml/cmake/common.cmake emits them). Picking source files is no longer
    #    enough: an undefined macro is a hard `identifier is undefined` error in
    #    the `if constexpr`. So derive both the sources and the macros from one
    #    list of combinations.
    local fa_combinations
    if [ "$fa_all_quants" = "1" ]; then
        fa_combinations=""
        for type_v in $FA_TYPES; do
            for type_k in $FA_TYPES; do
                fa_combinations="$fa_combinations $type_k-$type_v"
            done
        done
    else
        fa_combinations="$FA_DEFAULT_COMBINATIONS"
    fi

    for combination in $fa_combinations; do
        f="$ti_dir/fattn-vec-instance-$combination.cu"
        if [ ! -f "$f" ]; then
            echo "Error: FlashAttention template instance not found: $f" >&2
            return 1
        fi
        CUDA_SOURCES="$CUDA_SOURCES $f"
    done

    # One -D per combination, 0 or 1, for every type pair fattn.cu can ask about.
    # No spaces in any value, so these survive the unquoted word-splitting that
    # compile_gpu_sources_parallel does on its common_flags argument.
    CUDA_FA_DEFINES=""
    for type_v in $FA_TYPES; do
        for type_k in $FA_TYPES; do
            local compiled=0
            case " $fa_combinations " in
                *" $type_k-$type_v "*) compiled=1 ;;
            esac
            local macro
            macro=$(echo "GGML_CUDA_FA_${type_k}_${type_v}" | tr '[:lower:]' '[:upper:]')
            CUDA_FA_DEFINES="$CUDA_FA_DEFINES -D$macro=$compiled"
        done
    done
    # Reported by ggml_backend_cuda_get_features as "FA_QUANTS" (guarded by
    # #ifdef, so a build without it just omits the feature).
    local fa_quants_csv
    fa_quants_csv=$(echo $fa_combinations | tr ' ' ',')
    CUDA_FA_DEFINES="$CUDA_FA_DEFINES -DGGML_CUDA_FA_QUANTS=\"$fa_quants_csv\""

    # 4. mmf instances (always included)
    for f in "$ti_dir"/mmf-*.cu; do
        [ -f "$f" ] && CUDA_SOURCES="$CUDA_SOURCES $f"
    done

    # 5. mmq instances: include all, but optionally exclude IQ quant templates
    for f in "$ti_dir"/mmq-*.cu; do
        if [ -f "$f" ]; then
            if [ "$no_iq_quants" = "1" ]; then
                case "$(basename "$f")" in
                    mmq-instance-iq*) continue ;;
                esac
            fi
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
    local pids=()
    local fail=0

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
        pids+=("$!")

        # Limit parallel jobs by waiting when we hit the limit
        local running=$(jobs -r | wc -l)
        while [ "$running" -ge "$jobs" ]; do
            sleep 0.1
            running=$(jobs -r | wc -l)
        done
    done

    echo ""
    echo "Waiting for remaining compilations to finish..."
    # Reap each job individually and check its status. A bare `wait` returns 0
    # even when a background nvcc/hipcc failed, which would silently drop the
    # object and let the link produce an incomplete library (bit us on
    # out-prod.cu / cublasSgemmBatched). Fail loudly instead.
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    if [ "$fail" -ne 0 ]; then
        echo "Error: one or more GPU source compilations failed (see errors above)" >&2
        return 1
    fi
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
        $llama_cpp_dir/ggml/src/ggml-backend-meta.cpp \
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
