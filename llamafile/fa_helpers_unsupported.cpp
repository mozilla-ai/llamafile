// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Fallback stubs for flash-attention helpers (issue #975).
//
// Returning false tells the caller to use the upstream ggml helper.
// Selected at runtime by sgemm.cpp when the CPU lacks AVX-512F (and
// any future per-arch FA helper variants we add).

#include "sgemm.h"
#include "ggml-impl.h"
#include <cstdint>

extern "C" {

bool llamafile_fa_vec_dot_f16_unsupported(int, float *, const void *,
                                           const void *) {
    return false;
}

bool llamafile_fa_fp16_to_fp32_row_unsupported(const void *, float *,
                                                int64_t) {
    return false;
}

bool llamafile_fa_simd_gemm_unsupported(float *, const float *, const float *,
                                         int, int, int) {
    return false;
}

} // extern "C"
