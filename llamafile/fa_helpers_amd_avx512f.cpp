// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// AVX-512F-optimized flash-attention helpers (issue #975).
//
// Routes around ops.cpp's f16 helpers (ggml_vec_dot_f16,
// ggml_fp16_to_fp32_row), which cosmocc compiles at AVX2 baseline so
// the APE binary stays portable. These wrappers are compiled with
// per-arch flags via BUILD.mk and dispatched at runtime by sgemm.cpp
// based on X86_HAVE(AVX512F).
//
// Functions return true when handled by the optimized kernel; the
// caller (the FA function in ops.cpp) falls back to the upstream
// helper otherwise.

#ifdef __x86_64__

#include "sgemm.h"
#include "ggml-impl.h"
#include <immintrin.h>
#include <cstdint>

extern "C" {

bool llamafile_fa_vec_dot_f16_amd_avx512f(int n, float *s,
                                           const void *vx, const void *vy) {
    const ggml_fp16_t *x = (const ggml_fp16_t *)vx;
    const ggml_fp16_t *y = (const ggml_fp16_t *)vy;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    // Unrolled by 2: 32 f16 elements per iteration.
    for (; i + 32 <= n; i += 32) {
        __m512 xv0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        __m512 yv0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(y + i)));
        __m512 xv1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 16)));
        __m512 yv1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(y + i + 16)));
        acc0 = _mm512_fmadd_ps(xv0, yv0, acc0);
        acc1 = _mm512_fmadd_ps(xv1, yv1, acc1);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    for (; i + 16 <= n; i += 16) {
        __m512 xv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        __m512 yv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(y + i)));
        acc0 = _mm512_fmadd_ps(xv, yv, acc0);
    }
    float sum = _mm512_reduce_add_ps(acc0);
    for (; i < n; ++i) {
        sum += GGML_COMPUTE_FP16_TO_FP32(x[i]) * GGML_COMPUTE_FP16_TO_FP32(y[i]);
    }
    *s = sum;
    return true;
}

bool llamafile_fa_fp16_to_fp32_row_amd_avx512f(const void *vx, float *y,
                                                int64_t n) {
    const ggml_fp16_t *x = (const ggml_fp16_t *)vx;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i)));
        _mm512_storeu_ps(y + i, v);
    }
    for (; i < n; ++i) {
        y[i] = GGML_COMPUTE_FP16_TO_FP32(x[i]);
    }
    return true;
}

} // extern "C"

#endif // __x86_64__
