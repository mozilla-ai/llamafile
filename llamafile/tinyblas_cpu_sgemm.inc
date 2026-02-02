// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tinyblas_cpu.h"

//
//
//                                ██████╗ ██╗   █████╗ ██████╗
//         ██████╗██╗██╗ ██╗██═██╗██╔══██╗██║  ██╔══██╗██╔═══╝
//         ╚═██╔═╝██║███▄██║██ ██║██████╔╝██║  ███████║██████╗
//           ██║  ██║██▀███║╚███╔╝██╔══██╗██║  ██╔══██║╔═══██║
//           ██║  ██║██║ ██║ ███║ ██████╔╝████╗██║  ██║██████║
//           ╚═╝  ╚═╝╚═╝ ╚═╝ ╚══╝ ╚═════╝ ╚═══╝╚═╝  ╚═╝╚═════╝
//
//                   BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// With the F32, F16, and BF16 data types, the accumulation of roundoff
// errors will only grow logarithmically, thanks to the ruler function.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

namespace {

template <typename TC>
bool llamafile_sgemm_impl(long m, long n, long k, const void *A, long lda, const void *B, long ldb,
                          TC *C, long ldc, int ith, int nth, int Atype, int Btype, int Ctype) {

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return NOT_SUPPORTED;
#if defined(__AVX512F__)
        tinyBLAS<0, 16, __m512, __m512, float, float, TC> tb{
            k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<0, 8, __m256, __m256, float, float, TC> tb{
            k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_NEON)
        tinyBLAS<0, 4, float32x4_t, float32x4_t, float, float, TC> tb{
            k, (const float *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return NOT_SUPPORTED;
#endif
    }

    case GGML_TYPE_BF16: {
#if defined(__AVX512BF16__)
        if (Btype == GGML_TYPE_F32 && n <= 2) {
            tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
                k, (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n);
            return true;
        }
        if (Btype == GGML_TYPE_F32)
            return WANT_QUANTIZATION;
        if (Btype != GGML_TYPE_BF16)
            return NOT_SUPPORTED;
        if (n > 1) {
            tinyBLAS<0, 32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, TC> tb{
                k, (const ggml_bf16_t *)A, lda, (const ggml_bf16_t *)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n);
            return true;
        } else {
            tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, TC> tb{
                k, (const ggml_bf16_t *)A, lda, (const ggml_bf16_t *)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n);
            return true;
        }
#elif defined(__AVX512F__)
        tinyBLAS<0, 16, __m512, __m512, ggml_bf16_t, float, TC> tb{
            k, (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__AVX2__)
        if (Btype != GGML_TYPE_F32)
            return NOT_SUPPORTED;
        tinyBLAS<0, 8, __m256, __m256, ggml_bf16_t, float, TC> tb{
            k, (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (Btype != GGML_TYPE_F32)
            return NOT_SUPPORTED;
        tinyBLAS<0, 4, float32x4_t, float32x4_t, ggml_bf16_t, float, TC> tb{
            k, (const ggml_bf16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return NOT_SUPPORTED;
#endif
    }

    case GGML_TYPE_F16: {
#if defined(__AVX512F__)
        if (Btype == GGML_TYPE_F32 && n <= 2) {
            tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, float, TC> tb{
                k, (const ggml_fp16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n);
            return true;
        }
        if (Btype == GGML_TYPE_F32)
            return WANT_QUANTIZATION;
        if (Btype != GGML_TYPE_F16)
            return NOT_SUPPORTED;
        tinyBLAS<0, 16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, TC> tb{
            k, (const ggml_fp16_t *)A, lda, (const ggml_fp16_t *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (X86_CHECK(F16C)) {
            if (Btype == GGML_TYPE_F32 && n <= 2) {
                tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, float, TC> tb{
                    k, (const ggml_fp16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
                tb.matmul(m, n);
                return true;
            }
            if (Btype == GGML_TYPE_F32)
                return WANT_QUANTIZATION;
            if (Btype != GGML_TYPE_F16)
                return NOT_SUPPORTED;
            tinyBLAS<0, 8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, TC> tb{
                k, (const ggml_fp16_t *)A, lda, (const ggml_fp16_t *)B, ldb, C, ldc, ith, nth};
            tb.matmul(m, n);
            return true;
        } else {
            return NOT_SUPPORTED;
        }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        if (n < 2)
            // TODO(jart): Why is ggml_vec_dot_f16_unroll() so fast at matvec?
            return NOT_PROFITABLE;
        if (Btype == GGML_TYPE_F32)
            return WANT_QUANTIZATION;
        if (Btype != GGML_TYPE_F16)
            return NOT_SUPPORTED;
        tinyBLAS<0, 8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, TC> tb{
            k, (const ggml_fp16_t *)A, lda, (const ggml_fp16_t *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (n < 2 && !FLAG_precise)
            // TODO(jart): Why is ggml_vec_dot_f16_unroll() so fast at matvec?
            return NOT_PROFITABLE;
        if (Btype != GGML_TYPE_F32)
            return NOT_SUPPORTED;
        tinyBLAS<0, 4, float32x4_t, float32x4_t, ggml_fp16_t, float, TC> tb{
            k, (const ggml_fp16_t *)A, lda, (const float *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return NOT_SUPPORTED;
#endif
    }

    case GGML_TYPE_Q8_0: {
        if (Btype == GGML_TYPE_F32)
            return WANT_QUANTIZATION;
        if (Btype != GGML_TYPE_Q8_0)
            return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
        tinyBLAS_Q0_AVX2<0, block_q8_0, block_q8_0, TC> tb{
            k, (const block_q8_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<0, block_q8_0, block_q8_0, TC> tb{
            k, (const block_q8_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return NOT_SUPPORTED;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype == GGML_TYPE_F32)
            return WANT_QUANTIZATION;
        if (Btype != GGML_TYPE_Q8_0)
            return NOT_SUPPORTED;
#if defined(__AVX2__) || defined(__AVX512F__)
        tinyBLAS_Q0_AVX2<0, block_q4_0, block_q8_0, TC> tb{
            k, (const block_q4_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<0, block_q4_0, block_q8_0, TC> tb{
            k, (const block_q4_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
        return true;
#else
        return NOT_SUPPORTED;
#endif
    }

    default:
        return NOT_SUPPORTED;
    }

    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)Atype;
    (void)Btype;
}

} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc, 0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @param precision may be used to control the internal compute type
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(long m, long n, long k, const void *A, long lda, const void *B, long ldb,
                     void *C, long ldc, int ith, int nth, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);

#if defined(__x86_64__)
    if (X86_CHECK(AVX2) && X86_CHECK(FMA)) {
        if (Btype == GGML_TYPE_Q8_K && Ctype == GGML_TYPE_F32) {
            if (iqk_mul_mat(m, n, k * QK_K, Atype, A, B, (float *)C, ldc, ith, nth)) {
                return true;
            }
        }
        if ((Btype == GGML_TYPE_Q8_0 || Btype == GGML_TYPE_Q8_1) && Ctype == GGML_TYPE_F32) {
            assert(QK8_0 == 32 && //
                   QK8_1 == 32 && //
                   QK4_0 == 32 && //
                   QK4_1 == 32 && //
                   QK5_0 == 32 && //
                   QK5_1 == 32);
            if (iqk_mul_mat(m, n, k * QK8_0, Atype, A, B, (float *)C, ldc, ith, nth)) {
                return true;
            }
        }
    }
#elif defined __aarch64__ && defined __ARM_FEATURE_DOTPROD && !defined _MSC_VER
    if (Btype == GGML_TYPE_Q8_K && Ctype == GGML_TYPE_F32) {
        if (iqk_mul_mat(m, n, k * QK_K, Atype, A, B, (float *)C, ldc, ith, nth)) {
            return true;
        }
    }
    if ((Btype == GGML_TYPE_Q8_0 || Btype == GGML_TYPE_Q8_1) && Ctype == GGML_TYPE_F32) {
        assert(QK8_0 == 32 && //
               QK8_1 == 32 && //
               QK4_0 == 32 && //
               QK4_1 == 32 && //
               QK5_0 == 32 && //
               QK5_1 == 32);
        if (iqk_mul_mat(m, n, k * QK8_0, Atype, A, B, (float *)C, ldc, ith, nth)) {
            return true;
        }
    }
#endif

    switch (Ctype) {
    case GGML_TYPE_F32:
        return llamafile_sgemm_impl(m, n, k, A, lda, B, ldb, (float *)C, ldc, ith, nth, Atype,
                                    Btype, Ctype);
    default:
        return NOT_SUPPORTED;
    }
}
