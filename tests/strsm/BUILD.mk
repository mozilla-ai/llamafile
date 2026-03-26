#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

# ==============================================================================
# STRSM Manual Tests (TinyBLAS vs cuBLAS)
# ==============================================================================
# These are manual CUDA tests for validating tinyblasStrsmBatched correctness
# and performance against cuBLAS. They are NOT included in `make check` because:
#   - They require an NVIDIA GPU with CUDA toolkit installed
#   - They require cuBLAS (libcublas.so) for the reference implementation
#   - They are compiled with nvcc, not cosmocc
#   - They are primarily for development validation
#
# To build and run:
#   bash tests/strsm/build_and_run.sh
#
# To build only:
#   bash tests/strsm/build_and_run.sh --build-only
#
# The test compiles both tinyblas.cu and cublas into one binary, running the
# same STRSM operation through both backends and comparing results directly
# on the GPU.
