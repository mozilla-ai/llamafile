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

#pragma once

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "log.h"
#include "sgemm.h"
#include <cosmo.h>

#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"

#define CHUNK 8
#define ROW_ALIGN 64
#define MATRIX_ALIGN 4096
#define MAX_ALIGN 4096

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#if 0
#define NOT_SUPPORTED tinyBLAS_not_supported(__FILE__, __LINE__)
#else
#define NOT_SUPPORTED false
#endif
#define NOT_PROFITABLE false
#define WANT_QUANTIZATION false

namespace {

bool tinyBLAS_not_supported(const char *file, int line) {
    tinylogf("%s:%d: tinyBLAS not supported\n", file, line);
    return false;
}

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}
inline float unhalf(ggml_bf16_t d) {
    return GGML_BF16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MATRIX MEMORY INDEXING

#define NCA 1
#define NCB 2
#define NCC 4

#define INDEX(A, lda, j, i) (CONFIG & NC##A ? ((T##A **)A)[j] + i : A + lda * (j) + i)

////////////////////////////////////////////////////////////////////////////////////////////////////
// GGML TYPE TRAITS

template <typename T>
struct ggml_type_trait;
template <>
struct ggml_type_trait<float> {
    static constexpr ggml_type id = GGML_TYPE_F32;
};
template <>
struct ggml_type_trait<ggml_bf16_t> {
    static constexpr ggml_type id = GGML_TYPE_BF16;
};
template <>
struct ggml_type_trait<ggml_fp16_t> {
    static constexpr ggml_type id = GGML_TYPE_F16;
};
template <>
struct ggml_type_trait<block_q8_0> {
    static constexpr ggml_type id = GGML_TYPE_Q8_0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) {
    return _mm_add_ps(x, y);
}
inline __m128 sub(__m128 x, __m128 y) {
    return _mm_sub_ps(x, y);
}
inline __m128 mul(__m128 x, __m128 y) {
    return _mm_mul_ps(x, y);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 add(__m256 x, __m256 y) {
    return _mm256_add_ps(x, y);
}
inline __m256 sub(__m256 x, __m256 y) {
    return _mm256_sub_ps(x, y);
}
inline __m256 mul(__m256 x, __m256 y) {
    return _mm256_mul_ps(x, y);
}
#endif // __AVX__

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) {
    return _mm512_add_ps(x, y);
}
inline __m512 sub(__m512 x, __m512 y) {
    return _mm512_sub_ps(x, y);
}
inline __m512 mul(__m512 x, __m512 y) {
    return _mm512_mul_ps(x, y);
}
#endif // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) {
    return vaddq_f32(x, y);
}
inline float32x4_t sub(float32x4_t x, float32x4_t y) {
    return vsubq_f32(x, y);
}
inline float32x4_t mul(float32x4_t x, float32x4_t y) {
    return vmulq_f32(x, y);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) {
    return vaddq_f16(x, y);
}
inline float16x8_t sub(float16x8_t x, float16x8_t y) {
    return vsubq_f16(x, y);
}
inline float16x8_t mul(float16x8_t x, float16x8_t y) {
    return vmulq_f16(x, y);
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

/**
 * Computes a * b + c with error correction.
 *
 * @see W. Kahan, "Further remarks on reducing truncation errors,"
 *    Communications of the ACM, vol. 8, no. 1, p. 40, Jan. 1965,
 *    doi: 10.1145/363707.363723.
 */
template <typename T, typename U>
inline U madder(T a, T b, U c, U *e) {
    U y = sub(mul(a, b), *e);
    U t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}

#ifdef __ARM_NEON
inline float32x4_t badder(float32x4_t a, float b, float32x4_t c, float32x4_t *e) {
    float32x4_t y = sub(vmulq_n_f32(a, b), *e);
    float32x4_t t = add(c, y);
    *e = sub(sub(t, c), y);
    return t;
}
#endif

#if defined(__FMA__)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__ARM_FEATURE_FMA)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, a, b);
}
#if 0 // todo: this specialization chops gcc 12.3 performance in half
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER) && 0
template <>
inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vfmaq_f16(c, b, a);
}
#endif
#endif
#endif

#if defined(__AVX512BF16__)
template <>
inline __m512 madd(__m512bh x, __m512bh y, __m512 z) {
    return _mm512_dpbf16_ps(z, x, y);
}
template <>
inline __m512 madder(__m512bh x, __m512bh y, __m512 z, __m512 *_) {
    return _mm512_dpbf16_ps(z, x, y);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
inline float hsum(float32x4_t x) {
    return vaddvq_f32(x);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
inline float hsum(float16x8_t x) {
    // todo: this works great on clang but it produces terrible code on gcc 12.3
    return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)), vcvt_f32_f16(vget_high_f16(x))));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U>
T load(const U *);

template <>
inline float load(const float *p) {
    return *p;
}
template <>
inline float load(const ggml_fp16_t *p) {
    return unhalf(*p);
}
template <>
inline float load(const ggml_bf16_t *p) {
    return unhalf(*p);
}

#if defined(__ARM_NEON)
template <>
inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
template <>
inline float32x4_t load(const ggml_bf16_t *p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16((const unsigned short *)p), 16));
}
#if !defined(_MSC_VER)
template <>
inline float16x8_t load(const ggml_fp16_t *p) {
    return vld1q_f16((const float16_t *)p);
}
template <>
inline float32x4_t load(const ggml_fp16_t *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif // __AVX2__

#if defined(__F16C__)
template <>
inline __m256 load(const ggml_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <>
inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <>
inline __m512 load(const ggml_fp16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
template <>
inline __m512 load(const ggml_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <>
inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
template <>
inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
#endif // __AVX512BF16__

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT OUTPUT STREAMING

inline void store(float *p, float f) {
    *p = f;
}

inline void store(ggml_fp16_t *p, float f) {
    *p = GGML_FP32_TO_FP16(f);
}

inline void store(ggml_bf16_t *p, float f) {
    *p = GGML_FP32_TO_BF16(f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int CONFIG, int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(long k, const TA *A, long lda, const TB *B, long ldb, TC *C, long ldc, int ith,
             int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n) {
        mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

#if VECTOR_REGISTERS == 32
        switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
        case 0x55:
            mc = 5;
            nc = 5;
            gemm<5, 5>(m0, m, n0, n);
            break;
        case 0x54:
        case 0x53:
        case 0x52:
        case 0x45:
        case 0x44:
        case 0x43:
        case 0x42:
        case 0x35:
        case 0x34:
        case 0x33:
        case 0x32:
        case 0x25:
        case 0x24:
        case 0x23:
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x51:
        case 0x41:
        case 0x31:
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x15:
        case 0x14:
        case 0x13:
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
#endif

#if VECTOR_REGISTERS == 16
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 3)) {
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
        case 0x42:
        case 0x33:
        case 0x32:
        case 0x23:
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x41:
        case 0x31:
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x13:
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
#endif

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        D stack[bsr(k / CHUNK + 1) + 1][RN][RM];
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;

            size_t chunk, sp = 0;
            int i, j, rule, step = 2;
            for (chunk = 0; chunk + KN * CHUNK * 4 <= k; chunk += KN * CHUNK * 4, step += 2, ++sp) {

                D Cv[RN][RM] = {};
                for (long l = 0; l < KN * CHUNK * 4; l += KN)
                    if constexpr (RM<=RN)
#                       pragma GCC unroll 100
                        for (j = 0; j < RN; ++j)
#                           pragma GCC unroll 100
                            for (i = 0; i < RM; ++i)
                                Cv[j][i] = madd(load<V>(INDEX(A, lda, ii + i, chunk + l)), //
                                                load<V>(INDEX(B, ldb, jj + j, chunk + l)), //
                                                Cv[j][i]);
                    else
#                       pragma GCC unroll 100
                        for (i = 0; i < RM; ++i)
#                           pragma GCC unroll 100
                            for (j = 0; j < RN; ++j)
                                Cv[j][i] = madd(load<V>(INDEX(A, lda, ii + i, chunk + l)), //
                                                load<V>(INDEX(B, ldb, jj + j, chunk + l)), //
                                                Cv[j][i]);

                for (rule = bsr(step & -step); --rule;)
                    for (--sp, j = 0; j < RN; ++j)
                        for (i = 0; i < RM; ++i)
                            Cv[j][i] += stack[sp][j][i];

                for (j = 0; j < RN; ++j)
                    for (i = 0; i < RM; ++i)
                        stack[sp][j][i] = Cv[j][i];
            }

            D Cv[RN][RM] = {};
            for (; chunk + KN <= k; chunk += KN)
                if constexpr (RM<=RN)
#                   pragma GCC unroll 100
                    for (j = 0; j < RN; ++j)
#                       pragma GCC unroll 100
                        for (i = 0; i < RM; ++i)
                            Cv[j][i] = madd(load<V>(INDEX(A, lda, ii + i, chunk)), //
                                            load<V>(INDEX(B, ldb, jj + j, chunk)), //
                                            Cv[j][i]);
                else
#                   pragma GCC unroll 100
                    for (i = 0; i < RM; ++i)
#                       pragma GCC unroll 100
                        for (j = 0; j < RN; ++j)
                            Cv[j][i] = madd(load<V>(INDEX(A, lda, ii + i, chunk)), //
                                            load<V>(INDEX(B, ldb, jj + j, chunk)), //
                                            Cv[j][i]);
               

            while (sp--)
                for (j = 0; j < RN; ++j)
                    for (i = 0; i < RM; ++i)
                        Cv[j][i] += stack[sp][j][i];

            float Cf[RN][RM];
            for (j = 0; j < RN; ++j)
                for (i = 0; i < RM; ++i)
                    Cf[j][i] = hsum(Cv[j][i]);

            for (; chunk < k; ++chunk)
                if constexpr (RM<=RN)
                    for (j = 0; j < RN; ++j)
                        for (i = 0; i < RM; ++i)
                            Cf[j][i] = fmaf(load<float>(INDEX(A, lda, ii + i, chunk)), //
                                            load<float>(INDEX(B, ldb, jj + j, chunk)), //
                                            Cf[j][i]);
                else
                    for (i = 0; i < RM; ++i)
                        for (j = 0; j < RN; ++j)
                            Cf[j][i] = fmaf(load<float>(INDEX(A, lda, ii + i, chunk)), //
                                            load<float>(INDEX(B, ldb, jj + j, chunk)), //
                                            Cf[j][i]);

            for (j = 0; j < RN; ++j)
                for (i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), Cf[j][i]);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
template <int CONFIG, typename TA, typename TB, typename TC>
class tinyBLAS_Q0_ARM {
  public:
    tinyBLAS_Q0_ARM(long k, const TA *A, long lda, const TB *B, long ldb, TC *C, long ldc, int ith,
                    int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n) {
        mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3, false>(m0, m, n0, n);
                break;
            case 0x32:
            case 0x23:
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2, false>(m0, m, n0, n);
                break;
            case 0x31:
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, false>(m0, m, n0, n);
                break;
            case 0x13:
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, false>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, false>(m0, m, n0, n);
                break;
            default:
                return;
            }
        } else {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3, true>(m0, m, n0, n);
                break;
            case 0x32:
            case 0x23:
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2, true>(m0, m, n0, n);
                break;
            case 0x31:
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, true>(m0, m, n0, n);
                break;
            case 0x13:
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, true>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, true>(m0, m, n0, n);
                break;
            default:
                return;
            }
        }

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN, int PRECISE>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            float32x4_t Ce[RN][RM] = {};
            for (int l = 0; l < k; ++l)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i) {
                        float32x4_t a = vcvtq_f32_s32(vdotq_s32(
                            vdotq_s32(vdupq_n_s32(0), load_lo(INDEX(A, lda, ii + i, l)),
                                      load_lo(INDEX(B, ldb, jj + j, l))),
                            load_hi(INDEX(A, lda, ii + i, l)), load_hi(INDEX(B, ldb, jj + j, l))));
                        float b = unhalf(INDEX(A, lda, ii + i, l)->d) *
                                  unhalf(INDEX(B, ldb, jj + j, l)->d);
                        if (PRECISE)
                            Cv[j][i] = badder(a, b, Cv[j][i], &Ce[j][i]);
                        else
                            Cv[j][i] = vmlaq_n_f32(Cv[j][i], a, b);
                    }
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), hsum(Cv[j][i]));
        }
    }

    inline int8x16_t load_lo(const block_q8_0 *b) {
        return vld1q_s8(b->qs);
    }

    inline int8x16_t load_hi(const block_q8_0 *b) {
        return vld1q_s8(b->qs + 16);
    }

    inline int8x16_t load_lo(const block_q4_0 *b) {
        return vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs), vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    inline int8x16_t load_hi(const block_q4_0 *b) {
        return vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)), vdupq_n_s8(0x8));
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};
#endif // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__)
template <int CONFIG, typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX2 {
  public:
    tinyBLAS_Q0_AVX2(long k, const TA *A, long lda, const TB *B, long ldb, TC *C, long ldc, int ith,
                     int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(long m, long n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(long m0, long m, long n0, long n) {
        long mc, nc, mp, np;

#if VECTOR_REGISTERS == 32
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3, false>(m0, m, n0, n);
                break;
            case 0x32:
            case 0x23:
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2, false>(m0, m, n0, n);
                break;
            case 0x31:
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, true>(m0, m, n0, n);
                break;
            case 0x13:
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, true>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, true>(m0, m, n0, n);
                break;
            default:
                return;
            }
        } else {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3)) {
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3, true>(m0, m, n0, n);
                break;
            case 0x32:
            case 0x23:
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2, true>(m0, m, n0, n);
                break;
            case 0x31:
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, true>(m0, m, n0, n);
                break;
            case 0x13:
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, true>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, true>(m0, m, n0, n);
                break;
            default:
                return;
            }
        }
#endif

#if VECTOR_REGISTERS == 16
        if (!FLAG_precise) {
            switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 2)) {
            case 0x32:
                mc = 3;
                nc = 2;
                gemm<3, 2, false>(m0, m, n0, n);
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                gemm<2, 3, false>(m0, m, n0, n);
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2, false>(m0, m, n0, n);
                break;
            case 0x31:
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, false>(m0, m, n0, n);
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, false>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, false>(m0, m, n0, n);
                break;
            default:
                return;
            }
        } else {
            switch ((MIN(m - m0, 2) << 4) | MIN(n - n0, 1)) {
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1, true>(m0, m, n0, n);
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2, true>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1, true>(m0, m, n0, n);
                break;
            default:
                return;
            }
        }
#endif

        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN, int PRECISE>
    NOINLINE void gemm(long m0, long m, long n0, long n) {
        long ytiles = RM > 1 ? (m - m0) / RM : 1;
        long xtiles = RN > 1 ? (n - n0) / RN : 1;
        long tiles = xtiles * ytiles;
        long duty = (tiles + nth - 1) / nth;
        long start = duty * ith;
        long end = start + duty;
        if (end > tiles)
            end = tiles;
        for (long job = start; job < end; ++job) {
            long ii = m0 + job / xtiles * RM;
            long jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            __m256 Ce[RN][RM] = {};
            for (long l = 0; l < k; ++l)
#pragma GCC unroll 100
                for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                    for (int i = 0; i < RM; ++i) {
                        __m256 a = _mm256_set1_ps(unhalf(INDEX(A, lda, ii + i, l)->d) *
                                                  unhalf(INDEX(B, ldb, jj + j, l)->d));
                        __m256 b = updot(_mm256_sign_epi8(load(INDEX(A, lda, ii + i, l)),
                                                          load(INDEX(A, lda, ii + i, l))),
                                         _mm256_sign_epi8(load(INDEX(B, ldb, jj + j, l)),
                                                          load(INDEX(A, lda, ii + i, l))));
                        if (PRECISE)
                            Cv[j][i] = madder(a, b, Cv[j][i], &Ce[j][i]);
                        else
                            Cv[j][i] = madd(a, b, Cv[j][i]);
                    }
#pragma GCC unroll 100
            for (int j = 0; j < RN; ++j)
#pragma GCC unroll 100
                for (int i = 0; i < RM; ++i)
                    store(INDEX(C, ldc, jj + j, ii + i), hsum(Cv[j][i]));
        }
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m256i load(const block_q4_0 *b) {
        __m128i x = _mm_loadu_si128((const __m128i *)b->qs);
        return _mm256_sub_epi8(_mm256_and_si256(_mm256_set1_epi8(15),
                                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                                        _mm_srli_epi16(x, 4), 1)),
                               _mm256_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const long k;
    const long lda;
    const long ldb;
    const long ldc;
    const int ith;
    const int nth;
};
#endif // __AVX2__

} // namespace
