// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat.inc
// Copyrigth 2024 Iwan Kawrakow.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
// Copyright 2024 Iwan Kawrakow
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

#include <cstring>
#include <type_traits>
#if defined __x86_64__ || defined __aarch64__ || defined(_M_X64)

#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "sgemm.h"

// For i-quants, I had to explicitely specify which
// functions to inline / not inline (at least for some
// of the functions), else performance would be significantly
// lower. This is worrysome as things can change with,
// e.g., a different compiler version or running on a different
// CPU.
#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((always_inline))
#endif

#define GGML_COMMON_IMPL_C
#include "llama.cpp/ggml-common.h"

// clang-format off

// This matrix - vector and matrix - matrix multiplication implementation
// for legacy quants, k-quants and i-quants makes prompt processing 150-200%
// (legacy and k-quants) or 250-400% (i-quants) faster.
// compared to mainline llama.cpp (and llamafile).
// It provides implementations for ARM_NEON (all quants) and AVX2
// (all quants except sub-4 bit i-quants).
//
// Main idea is that unpacking the quants and the block scales to
// be ready for dot products with the corresponding Q8_Y quants
// takes time (here 'Y' stands for K, 0, or 1, depending on quantization type).
// Hence, if we are performing a QX x Q8_Y matrix matrix
// multiplication (as needed for prompt processing), we can get
// a significant speedup by reusing the unpacked QX quants and scales
// for multiplication with several Q8_K columns. We also achieve fewer
// loads from memory, which is the main purpose of tiling in general
// purpose matrix multiplication packages.

#include <utility>
#include <array>

#endif

namespace {

typedef struct {
    int32_t i1;
    int32_t i2;
} mmid_row_mapping;

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y = 0;
    int           ne11;
    const mmid_row_mapping * row_mapping = nullptr;
    size_t        bs2 = 0;

    inline const char * src1_row(int iy) const {
        if (!row_mapping) return cy + (cur_y + iy)*by;
        int i11 = row_mapping[cur_y + iy].i1 % ne11;
        int i12 = row_mapping[cur_y + iy].i2;
        return cy + (i11 + i12*ne11)*by;
    }

    inline void store(int ix, int iy, float result) const {
        *(dst_row(iy) + ix) = result;
        //dst_row(iy)[ix] = result;
    }
    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i12 = row_mapping[cur_y + iy].i2;
        int i1  = row_mapping[cur_y + iy].i1;
        int i2  = i12;
        return s + i1*bs + i2*bs2;
    }
};

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);

struct MulMat {
    std::array<mul_mat_t, 8> funcs = {};
    //inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
    IQK_NOINLINE void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X and M2 Max CPUs (but differences to other tile size are small)
        int n_step = (nrc_y - info.cur_y)/funcs.size();
        if (n_step > 0) {
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    funcs.back()(n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += funcs.size();
                }
            }
            info.cur_y += funcs.size() * n_step;
        }
        int n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }
    static IQK_NOINLINE bool set_mul_mat(int typeA, int ne00, MulMat& mm, int& row_size_q8, int Ny);
private:
    template <typename Dequantizer> static IQK_NOINLINE void set_functions(MulMat& m);
};

inline void make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

const uint64_t keven_signs[128] = {
    0x0101010101010101, 0xff010101010101ff, 0xff0101010101ff01, 0x010101010101ffff,
    0xff01010101ff0101, 0x0101010101ff01ff, 0x0101010101ffff01, 0xff01010101ffffff,
    0xff010101ff010101, 0x01010101ff0101ff, 0x01010101ff01ff01, 0xff010101ff01ffff,
    0x01010101ffff0101, 0xff010101ffff01ff, 0xff010101ffffff01, 0x01010101ffffffff,
    0xff0101ff01010101, 0x010101ff010101ff, 0x010101ff0101ff01, 0xff0101ff0101ffff,
    0x010101ff01ff0101, 0xff0101ff01ff01ff, 0xff0101ff01ffff01, 0x010101ff01ffffff,
    0x010101ffff010101, 0xff0101ffff0101ff, 0xff0101ffff01ff01, 0x010101ffff01ffff,
    0xff0101ffffff0101, 0x010101ffffff01ff, 0x010101ffffffff01, 0xff0101ffffffffff,
    0xff01ff0101010101, 0x0101ff01010101ff, 0x0101ff010101ff01, 0xff01ff010101ffff,
    0x0101ff0101ff0101, 0xff01ff0101ff01ff, 0xff01ff0101ffff01, 0x0101ff0101ffffff,
    0x0101ff01ff010101, 0xff01ff01ff0101ff, 0xff01ff01ff01ff01, 0x0101ff01ff01ffff,
    0xff01ff01ffff0101, 0x0101ff01ffff01ff, 0x0101ff01ffffff01, 0xff01ff01ffffffff,
    0x0101ffff01010101, 0xff01ffff010101ff, 0xff01ffff0101ff01, 0x0101ffff0101ffff,
    0xff01ffff01ff0101, 0x0101ffff01ff01ff, 0x0101ffff01ffff01, 0xff01ffff01ffffff,
    0xff01ffffff010101, 0x0101ffffff0101ff, 0x0101ffffff01ff01, 0xff01ffffff01ffff,
    0x0101ffffffff0101, 0xff01ffffffff01ff, 0xff01ffffffffff01, 0x0101ffffffffffff,
    0xffff010101010101, 0x01ff0101010101ff, 0x01ff01010101ff01, 0xffff01010101ffff,
    0x01ff010101ff0101, 0xffff010101ff01ff, 0xffff010101ffff01, 0x01ff010101ffffff,
    0x01ff0101ff010101, 0xffff0101ff0101ff, 0xffff0101ff01ff01, 0x01ff0101ff01ffff,
    0xffff0101ffff0101, 0x01ff0101ffff01ff, 0x01ff0101ffffff01, 0xffff0101ffffffff,
    0x01ff01ff01010101, 0xffff01ff010101ff, 0xffff01ff0101ff01, 0x01ff01ff0101ffff,
    0xffff01ff01ff0101, 0x01ff01ff01ff01ff, 0x01ff01ff01ffff01, 0xffff01ff01ffffff,
    0xffff01ffff010101, 0x01ff01ffff0101ff, 0x01ff01ffff01ff01, 0xffff01ffff01ffff,
    0x01ff01ffffff0101, 0xffff01ffffff01ff, 0xffff01ffffffff01, 0x01ff01ffffffffff,
    0x01ffff0101010101, 0xffffff01010101ff, 0xffffff010101ff01, 0x01ffff010101ffff,
    0xffffff0101ff0101, 0x01ffff0101ff01ff, 0x01ffff0101ffff01, 0xffffff0101ffffff,
    0xffffff01ff010101, 0x01ffff01ff0101ff, 0x01ffff01ff01ff01, 0xffffff01ff01ffff,
    0x01ffff01ffff0101, 0xffffff01ffff01ff, 0xffffff01ffffff01, 0x01ffff01ffffffff,
    0xffffffff01010101, 0x01ffffff010101ff, 0x01ffffff0101ff01, 0xffffffff0101ffff,
    0x01ffffff01ff0101, 0xffffffff01ff01ff, 0xffffffff01ffff01, 0x01ffffff01ffffff,
    0x01ffffffff010101, 0xffffffffff0101ff, 0xffffffffff01ff01, 0x01ffffffff01ffff,
    0xffffffffffff0101, 0x01ffffffffff01ff, 0x01ffffffffffff01, 0xffffffffffffffff,
};

}

bool iqk_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, const void * B,
        float * C, long stride_C, int ith, int nth) {

    MulMat mm;
    int row_size_q8;
    if (!MulMat::set_mul_mat(typeA, ne00, mm, row_size_q8, Ny)) {
        return false;
    }

    auto row_size_qx = ggml_row_size((ggml_type)typeA, ne00);

    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;

    DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, (size_t)row_size_q8, 0, 1, nullptr, 0};

    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);

    return true;
}

bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11, int typeA, const void * A, const void * B,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth) {
    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
    assert(row_mapping != nullptr);

    MulMat mm;
    int row_size_q8;
    if (!MulMat::set_mul_mat(typeA, ne00, mm, row_size_q8, Ny)) {
        return false;
    }
    int row_size_qx = ggml_row_size((ggml_type)typeA, ne00);
    int nrc_x = (Nx + nth - 1)/nth;
    int first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), (size_t)row_size_q8, 0, ne11, row_mapping, nb2/sizeof(float)};
    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);
    return true;
}

#if defined __x86_64__ || defined(_M_X64)

#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif

namespace {

inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)


template <int nrc, typename block_q8 = block_q8_K> struct Q8 {

    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

#ifdef HAVE_FANCY_SIMD
    inline __m512i load_quants(int iy, int i, int j) const { return _mm512_loadu_si512((const __m512i*)y[iy][i].qs + j); }
#else
    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].qs + j); }
#endif
    inline __m256i load_bsums(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].bsums); }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

// Handles q4_K and q5_K scales/mins
struct Scales8K {
    template <typename Q8>
    inline __m256i process_mins_and_scales(const uint8_t * data, float c, int i, const Q8& q8, __m256 * accd) {
        make_q4_scales(data, utmp);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        accum_mins(mins128, q8, i, c, accd);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        return MM256_SET_M128I(sc128, sc128);
    }
#ifdef HAVE_FANCY_SIMD
    template <typename Q8>
    inline __m512i process_mins_and_scales_64(const uint8_t * data, float c, int i, const Q8& q8, __m256 * accd) {
        auto scales = process_mins_and_scales(data, c, i, q8, accd);
        return _mm512_inserti32x8(_mm512_castsi256_si512(scales), scales, 1);
    }
#endif
    template <typename Q8>
    inline void accum_mins(const __m128i& mins128, const Q8& q8, int i, float c, __m256 * accd) const {
        const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, shuffles[1]), _mm_shuffle_epi8(mins128, shuffles[0]));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i q8s = q8.load_bsums(iy, i);
            const __m256i prod = _mm256_madd_epi16(mins, q8s);
            accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(c*q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[iy]);
        }
    }
#ifdef HAVE_FANCY_SIMD
    const __m512i shuffles512[2] = {
        _mm512_set_epi64(0x0706070607060706, 0x0302030203020302, 0x0706070607060706, 0x0302030203020302,
                         0x0504050405040504, 0x0100010001000100, 0x0504050405040504, 0x0100010001000100),
        _mm512_set_epi64(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a,
                         0x0d0c0d0c0d0c0d0c, 0x0908090809080908, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
#endif
    const __m128i shuffles[2] = {_mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100),
                                 _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908)};

    uint32_t utmp[4];
};

template <typename Q8>
inline void process_mins_16(const __m256i& all_scales, const Q8& q8, int i, float d, __m256 * accm) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(all_scales, q8.load_bsums(iy, i));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
}
inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET_M128I(l_scales, l_scales);
    scales[1] = MM256_SET_M128I(h_scales, h_scales);
}

struct ScaleQ3 {
    inline __m128i make_scales(const uint16_t * s8) const {
        const uint16_t * scales16 = (const uint16_t *)s8;
        uint32_t aux0 = scales16[0] | (scales16[1] << 16);
        uint32_t aux1 = scales16[2] | (scales16[3] << 16);
        uint32_t aux2 = scales16[4] | (scales16[5] << 16);
        __m128i scales128 = _mm_set_epi32(
            ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030),
            ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030),
             (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030),
             (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030));
        return _mm_add_epi8(scales128, m32);
    }
    const __m128i m32 = _mm_set1_epi8(-32);
};

struct ScaleIQ4XS {
    inline __m128i make_scales(const uint32_t scales_l, const uint16_t scales_h) {
        uint32_t tmp32 = scales_h | (scales_h << 14);
        const __m128i sh = _mm_slli_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(tmp32), hshift), hmask), 4);
        const __m128i sl = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(scales_l), lshift), lmask);
        return _mm_add_epi16(_mm_or_si128(sh, _mm_cvtepi8_epi16(_mm_shuffle_epi8(sl, lshuffle))), m32);
    }
    const __m128i hshift = _mm_set_epi32(12, 8, 4, 0);
    const __m128i lshift = _mm_set_epi32(4, 0, 4, 0);
    const __m128i hmask  = _mm_set1_epi16(0x03);
    const __m128i lmask  = _mm_set1_epi8(0xf);
    const __m128i lshuffle = _mm_set_epi32(0x07030602, 0x05010400, 0x07030602, 0x05010400);
    const __m128i m32 = _mm_set1_epi16(-32);
};

template <typename Block>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx) : vx(vx), bx(bx) {}
    inline void new_row(int ix) {
        x = (const Block *)((const char *)vx + bx*ix);
    }

    const void *  vx;
    size_t        bx;
    const Block * x;

    float d;
};

#ifdef HAVE_FANCY_SIMD
//====================================== Zen4 ==================================================

struct BlockPermuter {
    const __m512i permute1 = _mm512_set_epi64(11, 10,  9,  8, 3, 2, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
};

struct Q4Bits {
    inline void prepare(const uint8_t * q4) {
        auto q4bits = _mm512_loadu_si512((const __m512i*)q4 + 0);
        auto tmp1 = _mm512_and_si512(q4bits, ml);
        auto tmp2 = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        values[0] = _mm512_permutex2var_epi64(tmp1, perm.permute1, tmp2);
        values[1] = _mm512_permutex2var_epi64(tmp1, perm.permute2, tmp2);
        q4bits = _mm512_loadu_si512((const __m512i*)q4 + 1);
        tmp1 = _mm512_and_si512(q4bits, ml);
        tmp2 = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        values[2] = _mm512_permutex2var_epi64(tmp1, perm.permute1, tmp2);
        values[3] = _mm512_permutex2var_epi64(tmp1, perm.permute2, tmp2);
    }
    inline void prepare64(const uint8_t * q4) {
        auto q4bits = _mm512_loadu_si512((const __m512i*)q4 + 0);
        values[0] = _mm512_and_si512(q4bits, ml);
        values[1] = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        q4bits = _mm512_loadu_si512((const __m512i*)q4 + 1);
        values[2] = _mm512_and_si512(q4bits, ml);
        values[3] = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
    }
    __m512i values[4];
    const __m512i ml = _mm512_set1_epi8(0xf);
    BlockPermuter perm;
};

struct Q2Bits {
    inline void prepare(const uint8_t * q2) {

        auto q2bits = _mm512_loadu_si512((const __m512i*)q2);
        auto tmp = _mm512_srli_epi16(q2bits, 2);

        values[0] = _mm512_permutex2var_epi64(q2bits, perm.permute1, tmp);
        values[2] = _mm512_permutex2var_epi64(q2bits, perm.permute2, tmp);
        values[1] = _mm512_and_si512(_mm512_srli_epi16(values[0], 4), ml);
        values[3] = _mm512_and_si512(_mm512_srli_epi16(values[2], 4), ml);
        values[0] = _mm512_and_si512(values[0], ml);
        values[2] = _mm512_and_si512(values[2], ml);
    }
    __m512i values[4];
    const __m512i ml = _mm512_set1_epi8(0x03);
    BlockPermuter perm;
};

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        auto all_scales = s8k.process_mins_and_scales_64(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
    }

    Q4Bits bits;
    Scales8K s8k;
};

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {
    DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs);
        auto scales128 = siq4.make_scales(*(const uint32_t *)x[i].scales_l, x[i].scales_h);
        s8k.accum_mins(scales128, q8, i, -128.f*d, accd);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
    }
    static __m512i load_values() {
        static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
        auto val256 = MM256_SET_M128I(val128, val128);
        return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
    }
    inline void prepare(const uint8_t * q4) {
        bits.prepare64(q4);
        // We now have in bits.valuse[0]: 0...15, 32...47, 64...79, 96...111
        //                bits.valuse[1]: 16..31, 48...63, 80...95, 112..127
        //                etc.
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]));
        bits.values[0] = _mm512_shuffle_epi8(values, tmp);
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]));
        bits.values[2] = _mm512_shuffle_epi8(values, tmp);
    }

    Q4Bits bits;
    Scales8K s8k;
    ScaleIQ4XS siq4;
    const __m512i values;
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
};

struct HighBit5 {
    inline void apply(const uint8_t * h, Q4Bits& bits) {
        auto hbits256 = _mm256_loadu_si256((const __m256i *)h);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(hbits256), _mm256_srli_epi16(hbits256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(hbits, mh));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh));
    }
    const __m512i mh = _mm512_set1_epi8(0x10);
};

struct HighBit3 {
    inline void apply(const uint8_t * h, Q2Bits& bits) {
        auto hbits256 = _mm256_loadu_si256((const __m256i *)h);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(hbits256), _mm256_srli_epi16(hbits256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(hbits, mh));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 4), mh));
    }
    const __m512i mh = _mm512_set1_epi8(0x04);
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        hbits.apply(x[i].qh, bits);
        auto all_scales = s8k.process_mins_and_scales_64(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
    }

    Q4Bits bits;
    HighBit5 hbits;
    Scales8K s8k;
};

struct Scale16 {
    inline void make_scales(const __m128i& scales8, __m512i * scales) const {
        auto all_scales8 = MM256_SET_M128I(scales8, scales8);
        auto scales1 = _mm256_shuffle_epi8(all_scales8, shuffle1);
        auto scales2 = _mm256_shuffle_epi8(all_scales8, shuffle2);
        scales[0] = _mm512_cvtepi8_epi16(scales1);
        scales[1] = _mm512_cvtepi8_epi16(scales2);
    }
    template <typename Q8>
    inline void process_mins_and_scales(int i, float c, const __m128i& mins8, const __m128i& scales8,
        const Q8& q8, __m256 * accm, __m512i * scales) const {
        process_mins_16(_mm256_cvtepi8_epi16(mins8), q8, i, c, accm);
        make_scales(scales8, scales);
    }
    const __m256i shuffle1 = _mm256_set_epi32(0x07070707, 0x03030303, 0x06060606, 0x02020202,
                                              0x05050505, 0x01010101, 0x04040404, 0x00000000);
    const __m256i shuffle2 = _mm256_set_epi32(0x0f0f0f0f, 0x0b0b0b0b, 0x0e0e0e0e, 0x0a0a0a0a,
                                              0x0d0d0d0d, 0x09090909, 0x0c0c0c0c, 0x08080808);
};

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        sc16.process_mins_and_scales(i, -GGML_FP16_TO_FP32(x[i].dmin), mins8, scales8, q8, accm, scales);
    }

    Q2Bits bits;
    Scale16 sc16;
    const __m128i m4 = _mm_set1_epi8(0xf);

};

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        hbits.apply(x[i].hmask, bits);
        auto scales128 = sc3.make_scales((const uint16_t *)x[i].scales);
        sc16.process_mins_and_scales(i, -4.f*d, scales128, scales128, q8, accm, scales);
    }

    Q2Bits bits;
    HighBit3 hbits;
    ScaleQ3 sc3;
    Scale16 sc16;
    const __m128i m4  = _mm_set1_epi8(0xf);
    const __m128i m32 = _mm_set1_epi8(-32);
};

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare64(x[i].ql);
        add_high_bits(x[i].qh, bits);
        auto scales128 = _mm_loadu_si128((const __m128i *)x[i].scales);
        sc16.process_mins_and_scales(i, -32.f*d, scales128, scales128, q8, accm, scales);
    }

    inline void add_high_bits(const uint8_t * qh, Q4Bits& bits) const {
        auto hbits = _mm512_loadu_si512((const __m512i *)qh);
        auto tmp1 = _mm512_and_si512(_mm512_slli_epi16(hbits, 4), mh);
        auto tmp2 = _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_permutex2var_epi64(tmp1, bits.perm.permute1, tmp2));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_permutex2var_epi64(tmp1, bits.perm.permute2, tmp2));
        tmp1 = _mm512_and_si512(hbits, mh);
        tmp2 = _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh);
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_permutex2var_epi64(tmp1, bits.perm.permute1, tmp2));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_permutex2var_epi64(tmp1, bits.perm.permute2, tmp2));
    }

    Q4Bits bits;
    HighBit3 hbits;
    Scale16 sc16;

    const __m512i mh = _mm512_set1_epi8(0x30);

};

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accm[nrc_y];
    __m512  accd[nrc_y];
    __m512i scales[2];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm512_setzero_ps();
        for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accm, scales);

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[0], q8.load_quants(iy, i, 0));
                const __m512i p2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[1], q8.load_quants(iy, i, 1));
                const __m512i p3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[2], q8.load_quants(iy, i, 2));
                const __m512i p4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[3], q8.load_quants(iy, i, 3));
                auto sumi = _mm512_dpwssd_epi32(_mm512_setzero_si512(), scales[0], _mm512_packs_epi32(p1, p2));
                sumi = _mm512_dpwssd_epi32(sumi, scales[1], _mm512_packs_epi32(p3, p4));
                accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(deq.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
            info.store(ix, iy, hsum_float_8(_mm256_add_ps(accm[iy], sum256)));
        }

    }
}

#else
// ===================================== Vanilla AVX2 =====================================

struct Q4Bits {
    inline void prepare(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[2] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare64(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[1] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare16(const uint8_t * q4, int j) {
        values[0] = dequant16(q4 + 64*j +  0);
        values[1] = dequant16(q4 + 64*j + 16);
        values[2] = dequant16(q4 + 64*j + 32);
        values[3] = dequant16(q4 + 64*j + 48);
    }
    inline __m256i dequant16(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i aux256 = MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128);
        return _mm256_and_si256(ml, aux256);
    };
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0xf);
};

struct Q2Bits {
    inline void prepare(const uint8_t * q2, int j) {
        auto q2bits = _mm256_loadu_si256((const __m256i *)q2 + j);
        values[0] = _mm256_and_si256(q2bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), ml);
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0x03);
};

struct HighBit5 {
    inline void load(const uint8_t * h) { hbits = _mm256_loadu_si256((const __m256i *)h); }
    inline void apply(Q4Bits& bits, bool do_shift) {
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 3), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
        if (do_shift) {
            hbits = _mm256_srli_epi16(hbits, 4);
        }
    }
    const __m256i mh = _mm256_set1_epi8(0x10);
    __m256i hbits;
};

struct HighBit3 {
    inline void load(const uint8_t * h) { hbits = _mm256_loadu_si256((const __m256i *)h); }
    inline void apply(Q2Bits& bits, bool do_shift) {
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh));
        if (do_shift) {
            hbits = _mm256_srli_epi16(hbits, 4);
        }
    }
    const __m256i mh = _mm256_set1_epi8(0x04);
    __m256i hbits;
};

inline __m256i get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}

inline void set_scales_8(const __m256i& all_scales, int j, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+3));
}

template <typename Q8, typename Bits>
inline void multiply_add(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    if (j == 0) {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 0)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 1)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 2)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 3)));
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p1, p3), _mm256_add_epi32(p2, p4));
        }
    } else {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 5)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 6)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 7)));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
        }
    }
}

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return s8k.process_mins_and_scales(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }

    Q4Bits bits;
    Scales8K s8k;
};

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {
    DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales128 = siq4.make_scales(*(const uint32_t *)x[i].scales_l, x[i].scales_h);
        s8k.accum_mins(scales128, q8, i, -128.f*d, accd);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }

    static __m256i load_values() {
        static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
        return MM256_SET_M128I(val128, val128);
    }

    Q4Bits bits;
    Scales8K s8k;
    ScaleIQ4XS siq4;
    const __m256i values;
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        hbits.load(x[i].qh);
        return s8k.process_mins_and_scales(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        hbits.apply(bits, j == 0);
    }

    Q4Bits  bits;
    HighBit5 hbits;
    Scales8K s8k;
};

template <typename Q8>
inline void process_mins_and_scales_16(const __m128i& scales128, const Q8& q8, int i, float d,
    __m256 * accm, __m256i * scales) {
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    process_mins_16(all_scales, q8, i, d, accm);
    prepare_scales_16(all_scales, scales);
}

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        hbits.load(x[i].hmask);
        process_mins_and_scales_16(sc3.make_scales((const uint16_t *)x[i].scales), q8, i, -4.f*d, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        hbits.apply(bits, j == 0);
    }

    Q2Bits  bits;
    HighBit3 hbits;
    ScaleQ3 sc3;

    const __m128i m32 = _mm_set1_epi8(-32);
};

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        process_mins_16(_mm256_cvtepi8_epi16(mins8), q8, i, -GGML_FP16_TO_FP32(x[i].dmin), accm);
        prepare_scales_16(_mm256_cvtepi8_epi16(scales8), scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }

    Q2Bits  bits;

    const __m128i m4 = _mm_set1_epi8(0xf);
};

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        process_mins_and_scales_16(_mm_loadu_si128((const __m128i *)x[i].scales), q8, i, -32.f*d, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare64(x[i].ql, j);
        auto hbits = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 2), mh));
    }

    Q4Bits  bits;
    const __m256i mh = _mm256_set1_epi8(0x30);
};

inline __m256i get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

inline void set_scales_16(const __m256i& all_scales, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(3));
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qY_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(info);

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    Dequantizer deq(vx, bx);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accd, all_scales);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j);
                set_scales_16(all_scales[j], scales);
                multiply_add(deq.bits, scales, j, i, q8, sumi);
            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(iy, i)), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }

}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            auto all_scales = deq.new_block(i, q8, accd);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {

                deq.prepare(i, j);

                set_scales_8(all_scales, j, scales);

                multiply_add(deq.bits, scales, j, i, q8, sumi);

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8.scale(iy, i));
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }
}
#endif  // Zen4 or vanilla AVX2

//
// ============================== Legacy quants
//

struct DotHelper {
    const __m256i m1 = _mm256_set1_epi16(1);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
    }
#else
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_madd_epi16(m1, _mm256_maddubs_epi16(x, y));
    }
#endif
};

struct SignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    }
};
struct UnsignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(x, y);
    }
};
template <typename Q8, typename Dot> struct Sum4 {
    Dot dot;
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[0].qs));
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y[1].qs));
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y[2].qs));
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y[3].qs));
        const __m256i p01 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p0, p1));    // 0,0, 1,1, 0,0, 1,1
        const __m256i p23 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p2, p3));    // 2,2, 3,3, 2,2, 3,3
        return _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p01, p23)); // 0,1,2,3, 0,1,2,3
    }
};

struct Sum4_Q8 {
    SignedDot dot;
    static inline __m256i add1(__m256i a, __m256i b) {
        return _mm256_add_epi32(_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b));
    }
    static inline __m256i add2(__m256i a, __m256i b) {
        return _mm256_add_epi32(_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));
    }
    inline __m256i compute(const __m256i * qx, const block_q8_0 * y) const {
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[0].qs));
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y[1].qs));
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y[2].qs));
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y[3].qs));
        const __m256i p01 = add1(p0, p1);  // 0,1, 0,1, 0,1, 0,1
        const __m256i p23 = add1(p2, p3);  // 2,3, 2,3, 2,3, 2,3
        return add2(p01, p23); // returns 0,1,2,3, 0,1,2,3
    }
};

struct ScaleHelperQ_0 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

struct ScaleHelperQ_1 {
    uint32_t scales8[4];
    const __m128i shuffle = _mm_set_epi16(0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100);

    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) {
            // it is slightly faster to directly dereference (const uint32 *)&y[j].d, but some compilers
            // complain that this breaks strict-aliasing rules.
            memcpy(scales8 + j, &y[j].d, sizeof(uint32_t));
        }
        return _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)scales8), shuffle));
    }

    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }

    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
};

template <int nrc_y> struct MinusType1 {
    __m128 accm[nrc_y];
    MinusType1() { for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm_setzero_ps(); }
    inline __m256 compute(__m256 dm, int iy) {
        const __m128 d = _mm256_castps256_ps128(dm);
        const __m128 m = _mm256_extractf128_ps(dm, 1);
        accm[iy] = _mm_add_ps(accm[iy], m);
        return _mm256_set_m128(d, d);
    }
    inline float compute(const std::pair<float, float>& dm, int iy) {
        accm[iy] = _mm_add_ps(accm[iy], _mm_set1_ps(dm.second*0.25f));
        return dm.first;
    }
    inline float result(__m256 acc, int iy) const {
        const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        return hsum_float_4(_mm_add_ps(sum, accm[iy]));
    }
};

template <typename Minus, int nrc_y, bool is_multiple_of_4> struct AccumT {
    __m256 acc[nrc_y];
    Minus accm;
    AccumT() {  for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = _mm256_setzero_ps(); }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, const DataInfo& info, int ix) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
            //s[iy*bs] = accm.result(acc[iy], iy);
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0, nrc_y, is_multiple_of_4>;

template <int nrc_y, bool is_multiple_of_4>
using AccumType1 = AccumT<MinusType1<nrc_y>, nrc_y, is_multiple_of_4>;

using Sum4Type0 = Sum4<block_q8_0, SignedDot>;
using Sum4Type1 = Sum4<block_q8_1, UnsignedDot>;

template <typename Unpacker, typename Sum4Type, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    Unpacker unp(vx, bx);
    Sum4Type sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ++ix) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type0, AccumType0<nrc_y, true>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type0, AccumType0<nrc_y, false>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_1_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_1> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type1, AccumType1<nrc_y, true>, ScaleHelperQ_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, Sum4Type1, AccumType1<nrc_y, false>, ScaleHelperQ_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

struct Dequantizer4bit {
    const __m256i m4 = _mm256_set1_epi8(0xf);
    inline __m256i dequant(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        return _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), m4);
    }
};

struct Q8_0_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_loadu_si256((const __m256i *)x->qs);
    }
};

struct Q4_0_Dequantizer {
    Dequantizer4bit b4;
    const __m256i m8 = _mm256_set1_epi8(-8);
    inline __m256i dequant(const block_q4_0 * x) const {
        return _mm256_add_epi8(b4.dequant(x->qs), m8);
    }
};

struct Q4_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_1 * x) const {
        return b4.dequant(x->qs);
    }
};

struct HBitDequantizer {
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m256i minus1 = _mm256_set1_epi64x(-1);
    inline __m256i to_bytes(const uint8_t * bits) const {
        // Note: Data in all ggml quants is at least 2-byte aligned.
        // => we can cast to uint16_t and use or on two consecutive entries
        // which is faster than memcpy
        const uint16_t * aux16 = (const uint16_t *)bits;
        const uint32_t aux32 = aux16[0] | (aux16[1] << 16);
        //uint32_t aux32; memcpy(&aux32, bits, sizeof(uint32_t));
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(aux32), shuffle);
        bytes = _mm256_or_si256(bytes, mask);
        return _mm256_cmpeq_epi8(bytes, minus1);
    }
};

struct Q5_0_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8((char)0xF0);
    inline __m256i dequant(const block_q5_0 * x) const {
        const __m256i vqh = _mm256_andnot_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

struct Q5_1_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8(0x10);
    inline __m256i dequant(const block_q5_1 * x) const {
        const __m256i vqh = _mm256_and_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

template <typename Q, typename Scales, typename Dequantizer>
struct Q_Unpacker {
    Q_Unpacker(const void * vx, size_t bx) : cx_0((const char *)vx), x((const Q*)cx_0), bx(bx) {}

    const char * cx_0;
    const Q    * x;
    size_t       bx;

    Scales scales;
    Dequantizer deq;

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const Q*)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        for (int j = 0; j < 4; ++j) {
            qx[j] = deq.dequant(x + 4*i + j);
        }
        return scales.prepare4(x + 4*i);
    }
    inline auto set_block(int i) {
        qx[0] = deq.dequant(x + i);
        return scales.prepare1(x + i);
    }
};

struct Q8_0_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_0; }
};
struct Q4_0_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0, Q4_0_Dequantizer> {
    Q4_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_0; }
};
struct Q5_0_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0, Q5_0_Dequantizer> {
    Q5_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK5_0; }
};
struct Q4_1_Unpacker final : public Q_Unpacker<block_q4_1, ScaleHelperQ_1, Q4_1_Dequantizer> {
    Q4_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_1; }
};
struct Q5_1_Unpacker final : public Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer> {
    Q5_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    inline static int block_size() { return QK4_1; }
};

template <int nrc_y>
void mul_mat_q8_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Q8_0_Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Q8_0_Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Q8_0_Unpacker, Sum4_Q8, AccumType0<nrc_y, true>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Q8_0_Unpacker, Sum4_Q8, AccumType0<nrc_y, false>, ScaleHelperQ_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
        if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker>) {
            m.funcs[0] = mul_mat_qX_0_q8_0_T<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_0_q8_0_T<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_0_q8_0_T<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_0_q8_0_T<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_0_q8_0_T<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_0_q8_0_T<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_0_q8_0_T<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_0_q8_0_T<Dequantizer, 8>;
        }
        else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker>) {
            m.funcs[0] = mul_mat_qX_1_q8_1_T<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_1_q8_1_T<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_1_q8_1_T<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_1_q8_1_T<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_1_q8_1_T<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_1_q8_1_T<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_1_q8_1_T<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_1_q8_1_T<Dequantizer, 8>;
        }
        else {
#ifdef HAVE_FANCY_SIMD
            m.funcs[0] = mul_mat_qX_K_q8_K_T<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_K_q8_K_T<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_K_q8_K_T<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_K_q8_K_T<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_K_q8_K_T<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_K_q8_K_T<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_K_q8_K_T<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_K_q8_K_T<Dequantizer, 8>;
#else
            if constexpr (std::is_same_v<Dequantizer, DequantizerQ2K> ||
                          std::is_same_v<Dequantizer, DequantizerQ3K> ||
                          std::is_same_v<Dequantizer, DequantizerQ6K>) {
                m.funcs[0] = mul_mat_qY_K_q8_K_T<Dequantizer, 1>;
                m.funcs[1] = mul_mat_qY_K_q8_K_T<Dequantizer, 2>;
                m.funcs[2] = mul_mat_qY_K_q8_K_T<Dequantizer, 3>;
                m.funcs[3] = mul_mat_qY_K_q8_K_T<Dequantizer, 4>;
                m.funcs[4] = mul_mat_qY_K_q8_K_T<Dequantizer, 5>;
                m.funcs[5] = mul_mat_qY_K_q8_K_T<Dequantizer, 6>;
                m.funcs[6] = mul_mat_qY_K_q8_K_T<Dequantizer, 7>;
                m.funcs[7] = mul_mat_qY_K_q8_K_T<Dequantizer, 8>;
            } else {
                m.funcs[0] = mul_mat_qX_K_q8_K_T<Dequantizer, 1>;
                m.funcs[1] = mul_mat_qX_K_q8_K_T<Dequantizer, 2>;
                m.funcs[2] = mul_mat_qX_K_q8_K_T<Dequantizer, 3>;
                m.funcs[3] = mul_mat_qX_K_q8_K_T<Dequantizer, 4>;
                m.funcs[4] = mul_mat_qX_K_q8_K_T<Dequantizer, 5>;
                m.funcs[5] = mul_mat_qX_K_q8_K_T<Dequantizer, 6>;
                m.funcs[6] = mul_mat_qX_K_q8_K_T<Dequantizer, 7>;
                m.funcs[7] = mul_mat_qX_K_q8_K_T<Dequantizer, 8>;
            }
#endif
        }
}

bool MulMat::set_mul_mat(int typeA, int ne00, MulMat& mm, int& row_size_q8, int) {

    if (ne00 % ggml_blck_size(GGML_TYPE_Q8_K) == 0)
        row_size_q8 = ggml_row_size(GGML_TYPE_Q8_K, ne00);

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ2K>(mm);
            break;
        case GGML_TYPE_Q3_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ3K>(mm);
            break;
        case GGML_TYPE_Q4_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ4K>(mm);
            break;
        case GGML_TYPE_Q5_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ5K>(mm);
            break;
        case GGML_TYPE_Q6_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ6K>(mm);
            break;
        case GGML_TYPE_IQ4_XS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ4XS>(mm);
            break;
        case GGML_TYPE_Q4_0:
            assert (ne00 % QK4_0 == 0);
            MulMat::set_functions<Q4_0_Unpacker>(mm);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q4_1:
            assert (ne00 % QK4_1 == 0);
            MulMat::set_functions<Q4_1_Unpacker>(mm);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;
        case GGML_TYPE_Q5_0:
            assert (ne00 % QK5_0 == 0);
            MulMat::set_functions<Q5_0_Unpacker>(mm);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q5_1:
            assert (ne00 % QK5_1 == 0);
            MulMat::set_functions<Q5_1_Unpacker>(mm);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;

        default:
            return false;
    }

    return true;
}

} // namespace


#else   // __aarch64__

namespace {

template <int nrc, typename block_q8 = block_q8_K> struct Q8 {

    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

    inline int8x16_t load_quants_16(int iy, int i, int j) const { return vld1q_s8(y[iy][i].qs + 16*j); }
    inline int8x16x2_t load_quants(int iy, int i, int j) const { return vld1q_s8_x2(y[iy][i].qs + 32*j); }
    inline int8x16x4_t load_quants_64(int iy, int i, int j) const { return vld1q_s8_x4(y[iy][i].qs + 64*j); }
    inline int16x8x2_t load_bsums(int iy, int i) const { return vld1q_s16_x2(y[iy][i].bsums); }
    inline int16x8_t load_bsums8(int iy, int i) const {
        auto q8s = vld1q_s16_x2(y[iy][i].bsums);
        return vpaddq_s16(q8s.val[0], q8s.val[1]);
    }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

template <int nrc_y, typename Dequantizer>
IQK_NOINLINE void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_K> q8(info);

    Dequantizer deq(vx, bx, nrc_y);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        float32x4_t acc[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

//#pragma GCC unroll 4
        for (int i = 0; i < nb; ++i) {

            int32x4_t sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = vdupq_n_s32(0);

            if constexpr (nrc_y > 1 && Dequantizer::should_scale_quants()) {
                deq.process_scales(i, q8, acc);
                deq.prepare(i, 0);
                deq.compute(q8, i, 0, sumi);
                deq.prepare(i, 1);
                deq.compute(q8, i, 1, sumi);
            } else {
                if constexpr (Dequantizer::num_blocks() == 8) {
                    auto scales = deq.new_block(i, q8, acc);
                    deq.prepare(i, 0);
#pragma GCC unroll 8
                    for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                    deq.prepare(i, 1);
#pragma GCC unroll 8
                    for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
                }
                else if constexpr (Dequantizer::num_blocks() == 16) {
                    auto scales = deq.new_block(i, q8, acc);
                    deq.prepare(i, 0);
#pragma GCC unroll 8
                    for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                    deq.prepare(i, 1);
#pragma GCC unroll 8
                    for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
                }
                else {
                    GGML_ASSERT(false);
                }
            }

#pragma GCC unroll 8
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vmlaq_f32(acc[iy], vcvtq_f32_s32(sumi[iy]), vdupq_n_f32(deq.d*q8.scale(iy, i)));
            }
        }

#pragma GCC unroll 8
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}
template <int nrc_y, typename Dequantizer>
IQK_NOINLINE void mul_mat_qX_K_q8_K_IQ(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_K> q8(info);

    Dequantizer deq(vx, bx, nrc_y);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        float32x4_t acc[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb; ++i) {

            int32x4_t sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = vdupq_n_s32(0);

            if constexpr (Dequantizer::num_blocks() == 8) {
                auto scales = deq.new_block(i);
                deq.prepare(i, 0);
#pragma GCC unroll 8
                for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                deq.prepare(i, 1);
#pragma GCC unroll 8
                for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
            }
            else if constexpr (Dequantizer::num_blocks() == 16) {
                auto scales = deq.new_block(i);
                deq.prepare(i, 0);
#pragma GCC unroll 8
                for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                deq.prepare(i, 1);
#pragma GCC unroll 8
                for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
            }
            else {
                GGML_ASSERT(false);
            }
#pragma GCC unroll 8
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vmlaq_f32(acc[iy], vcvtq_f32_s32(sumi[iy]), vdupq_n_f32(deq.d*q8.scale(iy, i)));
            }
        }
#pragma GCC unroll 8
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}

template <typename Q8>
IQK_ALWAYS_INLINE void compute_8_blocks(const uint8x16x4_t& qx_1, const uint8x16x4_t& qx_2, const Q8& q8,
        const int32x4x2_t& scales, int iy, int i, int j, int32x4_t& sumi) {
    auto mzero = vdupq_n_s32(0);
    const int8x16_t * qs_1 = (const int8x16_t *)qx_1.val;
    const int8x16_t * qs_2 = (const int8x16_t *)qx_2.val;

    auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
    auto p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qs_1[0], q8b_1.val[0]), qs_1[1], q8b_1.val[1]); // block 1
    auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
    auto p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qs_1[2], q8b_2.val[0]), qs_1[3], q8b_2.val[1]); // block 2
    auto p12 = vpaddq_s32(p1, p2);

    auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
    auto p3 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qs_2[0], q8b_3.val[0]), qs_2[1], q8b_3.val[1]); // block 3
    auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
    auto p4 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qs_2[2], q8b_4.val[0]), qs_2[3], q8b_4.val[1]); // block 4
    auto p34 = vpaddq_s32(p3, p4);

    auto pall = vpaddq_s32(p12, p34);
    sumi = vmlaq_s32(sumi, scales.val[j], pall);
}
template <typename Q8>
IQK_ALWAYS_INLINE void compute_8_blocks(const int8x16_t * qx, const Q8& q8,
        const int32x4_t& scales, int iy, int i, int j, int32x4_t& sumi) {
    auto mzero = vdupq_n_s32(0);

    auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
    auto p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qx[0], q8b_1.val[0]), qx[1], q8b_1.val[1]); // block 1
    auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
    auto p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qx[2], q8b_2.val[0]), qx[3], q8b_2.val[1]); // block 2
    auto p12 = vpaddq_s32(p1, p2);

    auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
    auto p3 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qx[4], q8b_3.val[0]), qx[5], q8b_3.val[1]); // block 3
    auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
    auto p4 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, qx[6], q8b_4.val[0]), qx[7], q8b_4.val[1]); // block 4
    auto p34 = vpaddq_s32(p3, p4);

    auto pall = vpaddq_s32(p12, p34);
    sumi = vmlaq_s32(sumi, scales, pall);
}

template <typename Q8>
IQK_ALWAYS_INLINE void compute_16_blocks(const uint8x16x4_t& qx_1, const uint8x16x4_t& qx_2, const Q8& q8,
        const int32x4x4_t& scales, int iy, int i, int j, int32x4_t& sumi) {

    auto mzero = vdupq_n_s32(0);
    auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
    auto p1 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[0]), q8b_1.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[1]), q8b_1.val[1])); // blocks 0, 0, 1, 1,
    auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
    auto p2 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[2]), q8b_2.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[3]), q8b_2.val[1])); // blocks 3, 3, 4, 4,
    auto p12 = vpaddq_s32(p1, p2); // blocks 0, 1, 2, 3
    sumi = vmlaq_s32(sumi, scales.val[2*j+0], p12);

    auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
    auto p3 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[0]), q8b_3.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[1]), q8b_3.val[1])); // block 4, 4, 5, 5,
    auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
    auto p4 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[2]), q8b_4.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[3]), q8b_4.val[1])); // block 6, 6, 7, 7,
    auto p34 = vpaddq_s32(p3, p4); // blocks 4, 5, 6, 7
    sumi = vmlaq_s32(sumi, scales.val[2*j+1], p34);
}

template <typename Q8>
inline void accum_mins_8(const int16x8_t& mins, const Q8& q8, float32x4_t * acc, int i, float c) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto q8s = q8.load_bsums8(iy, i);
        int32x4_t b1 = vmull_s16(vget_low_s16(mins), vget_low_s16(q8s));
        int32x4_t b2 = vmull_s16(vget_high_s16(mins), vget_high_s16(q8s));
        float32x4_t prod = vcvtq_f32_s32(vaddq_s32(b1, b2));
        acc[iy] = vmlaq_f32(acc[iy], prod, vdupq_n_f32(c*q8.scale(iy, i)));
    }
}
template <typename Q8>
inline void accum_mins_16(const int16x8x2_t& mins, const Q8& q8, float32x4_t * acc, int i, float c) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto q8s = q8.load_bsums(iy, i);
        int32x4_t b1 = vmull_s16(vget_low_s16 (mins.val[0]), vget_low_s16 (q8s.val[0]));
        int32x4_t b2 = vmull_s16(vget_high_s16(mins.val[0]), vget_high_s16(q8s.val[0]));
        int32x4_t b3 = vmull_s16(vget_low_s16 (mins.val[1]), vget_low_s16 (q8s.val[1]));
        int32x4_t b4 = vmull_s16(vget_high_s16(mins.val[1]), vget_high_s16(q8s.val[1]));
        float32x4_t prod = vcvtq_f32_s32(vaddq_s32(vaddq_s32(b1, b2), vaddq_s32(b3, b4)));
        acc[iy] = vmlaq_f32(acc[iy], prod, vdupq_n_f32(c*q8.scale(iy, i)));
    }
}

struct Scales8 {
    uint32_t utmp[4];
    const uint8_t * sc8 = (const uint8_t *)utmp;
    template <typename Q8, typename Qx>
    inline int32x4x2_t process_scales_mins(const Qx& x, const Q8& q8, int i, float32x4_t * acc) {
        make_q4_scales(x.scales, utmp);
        int16x8_t mins = vmovl_s8(vld1_s8((const int8_t *)sc8 + 8));
        accum_mins_8(mins, q8, acc, i, -GGML_FP16_TO_FP32(x.dmin));

        uint8x8_t scales8 = vld1_u8(sc8);
        uint16x8_t scales16 = vmovl_u8(scales8);
        int32x4x2_t scales = {vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales16))),
                              vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales16)))};
        return scales;
    }
};

struct Q4bits {
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    uint8x16x4_t b1, b2;
    inline void prepare4(uint8x16x4_t& b, const uint8x16_t * val) const {
        b.val[0] = vandq_u8(val[0], m4b);
        b.val[2] = vshrq_n_u8(val[0], 4);
        b.val[1] = vandq_u8(val[1], m4b);
        b.val[3] = vshrq_n_u8(val[1], 4);
    }
    inline void prepare4_16(uint8x16x4_t& b, const uint8x16_t * val) const {
        b.val[0] = vandq_u8(val[0], m4b);
        b.val[1] = vshrq_n_u8(val[0], 4);
        b.val[2] = vandq_u8(val[1], m4b);
        b.val[3] = vshrq_n_u8(val[1], 4);
    }
    inline void prepare(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x2(qs);
        prepare4(b1, q4bits.val);
        q4bits = vld1q_u8_x2(qs+32);
        prepare4(b2, q4bits.val);
    }
    inline void prepare_v2(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        prepare4(b1, q4bits.val+0);
        prepare4(b2, q4bits.val+2);
    }
    inline void prepare64(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        b1.val[0] = vandq_u8(q4bits.val[0], m4b);
        b1.val[1] = vandq_u8(q4bits.val[1], m4b);
        b1.val[2] = vandq_u8(q4bits.val[2], m4b);
        b1.val[3] = vandq_u8(q4bits.val[3], m4b);
        b2.val[0] = vshrq_n_u8(q4bits.val[0], 4);
        b2.val[1] = vshrq_n_u8(q4bits.val[1], 4);
        b2.val[2] = vshrq_n_u8(q4bits.val[2], 4);
        b2.val[3] = vshrq_n_u8(q4bits.val[3], 4);
    }
    inline void prepare16(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x2(qs);
        prepare4_16(b1, q4bits.val);
        q4bits = vld1q_u8_x2(qs+32);
        prepare4_16(b2, q4bits.val);
    }
    inline void prepare16_v2(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        prepare4_16(b1, q4bits.val+0);
        prepare4_16(b2, q4bits.val+2);
    }
};

struct Q2bits {
    const uint8x16_t m4b = vdupq_n_u8(0x03);
    uint8x16x4_t b1, b2;
    inline void prepare(const uint8_t * qs) {
        auto q2bits = vld1q_u8_x2(qs);
        b1.val[0] = vandq_u8(q2bits.val[0], m4b);
        b1.val[1] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b1.val[2] = vandq_u8(q2bits.val[0], m4b);
        b1.val[3] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b2.val[0] = vandq_u8(q2bits.val[0], m4b);
        b2.val[1] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b2.val[2] = vandq_u8(q2bits.val[0], m4b);
        b2.val[3] = vandq_u8(q2bits.val[1], m4b);
    }
};

template <typename block_q>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx, int nrc) : vx(vx), x(nullptr), bx(bx), nrc(nrc) {}
    inline void new_row(int ix) { x = (const block_q *)((const char *)vx + ix*bx); }
    const void * vx;
    const block_q * x;
    const size_t bx;
    const int nrc;
};

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return s8.process_scales_mins(x[i], q8, i, acc);
    }
    inline void prepare(int i, int j) {
        if (nrc == 1) bits.prepare_v2(x[i].qs+64*j);
        else bits.prepare(x[i].qs+64*j);
    }

    Q4bits bits;
    Scales8 s8;

    float d;
};

struct HighBit5 {
    const uint8x16_t mhb = vdupq_n_u8(0x10);
    uint8x16x2_t bits;
    inline void apply(uint8x16x4_t& b1, uint8x16x4_t& b2, bool do_shift) {
        b1.val[0] = vorrq_u8(b1.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 4), mhb));
        b1.val[1] = vorrq_u8(b1.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 4), mhb));
        b1.val[2] = vorrq_u8(b1.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 3), mhb));
        b1.val[3] = vorrq_u8(b1.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 3), mhb));

        b2.val[0] = vorrq_u8(b2.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 2), mhb));
        b2.val[1] = vorrq_u8(b2.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 2), mhb));
        b2.val[2] = vorrq_u8(b2.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 1), mhb));
        b2.val[3] = vorrq_u8(b2.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 1), mhb));

        if (do_shift) {
            bits.val[0] = vshrq_n_u8(bits.val[0], 4);
            bits.val[1] = vshrq_n_u8(bits.val[1], 4);
        }
    }
};

struct HighBit3 {
    const uint8x16_t mhb = vdupq_n_u8(0x04);
    uint8x16x2_t bits;
    inline void apply(uint8x16x4_t& b1, uint8x16x4_t& b2, bool do_shift) {
        b1.val[0] = vorrq_u8(b1.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 2), mhb));
        b1.val[1] = vorrq_u8(b1.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 2), mhb));
        b1.val[2] = vorrq_u8(b1.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 1), mhb));
        b1.val[3] = vorrq_u8(b1.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 1), mhb));

        b2.val[0] = vorrq_u8(b2.val[0], vandq_u8(bits.val[0], mhb));
        b2.val[1] = vorrq_u8(b2.val[1], vandq_u8(bits.val[1], mhb));
        b2.val[2] = vorrq_u8(b2.val[2], vandq_u8(vshrq_n_u8(bits.val[0], 1), mhb));
        b2.val[3] = vorrq_u8(b2.val[3], vandq_u8(vshrq_n_u8(bits.val[1], 1), mhb));

        if (do_shift) {
            bits.val[0] = vshrq_n_u8(bits.val[0], 4);
            bits.val[1] = vshrq_n_u8(bits.val[1], 4);
        }
    }
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        h.bits = vld1q_u8_x2(x[i].qh);
        return s8.process_scales_mins(x[i], q8, i, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+64*j);
        h.apply(bits.b1, bits.b2, j == 0);
    }

    Q4bits bits;
    HighBit5 h;
    Scales8 s8;

    uint8x16x2_t hbits;

    float d;
};

inline int32x4x4_t make_wider(const int16x8x2_t& scales16) {
    int32x4x4_t scales = {
        vmovl_s16(vget_low_s16 (scales16.val[0])),
        vmovl_s16(vget_high_s16(scales16.val[0])),
        vmovl_s16(vget_low_s16 (scales16.val[1])),
        vmovl_s16(vget_high_s16(scales16.val[1])),
    };
    return scales;
}

template <typename Q8>
inline int32x4x4_t process_scales_mins_16(const int8x16_t& scales8, const Q8& q8, float32x4_t * acc, int i, float c) {
    int16x8x2_t scales16;
    scales16.val[0] = vmovl_s8(vget_low_s8(scales8));
    scales16.val[1] = vmovl_s8(vget_high_s8(scales8));
    accum_mins_16(scales16, q8, acc, i, c);
    return make_wider(scales16);
}

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return process_scales_mins_16(vld1q_s8(x[i].scales), q8, acc, i, -32.f*d);
    }
    inline void prepare(int i, int j) {

        auto hbits = vld1q_u8_x2(x[i].qh + 32*j);

        bits.prepare64(x[i].ql+64*j);
        bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 4), mhb));
        bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 4), mhb));
        bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 2), mhb));
        bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 2), mhb));

        bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], mhb));
        bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], mhb));
        bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 2), mhb));
        bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 2), mhb));

    }

    Q4bits bits;

    const uint8x16_t mhb = vdupq_n_u8(0x30);

    float d;
};

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        h.bits = vld1q_u8_x2(x[i].hmask);
        const uint16_t * sc16 = (const uint16_t *)x[i].scales;
        uint32_t aux0 = sc16[0] | (sc16[1] << 16);
        uint32_t aux1 = sc16[2] | (sc16[3] << 16);
        uint32_t aux2 = sc16[4] | (sc16[5] << 16);
        aux32[0] =  (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030);
        aux32[1] =  (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030);
        aux32[2] = ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030);
        aux32[3] = ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030);
        return process_scales_mins_16(vaddq_s8(vld1q_s8((const int8_t *)aux32), vdupq_n_s8(-32)), q8, acc, i, -4.f*d);
    }

    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        h.apply(bits.b1, bits.b2, j == 0);
    }

    uint32_t aux32[4];

    Q2bits bits;

    HighBit3 h;

    float d;
};

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return true; }

    template <typename Q8>
    inline void process_scales(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales_and_mins = vld1q_u8(x[i].scales);
        auto mins8 = vreinterpretq_s8_u8(vshrq_n_u8(scales_and_mins, 4));
        int16x8x2_t scales16;
        scales16.val[0] = vmovl_s8(vget_low_s8(mins8));
        scales16.val[1] = vmovl_s8(vget_high_s8(mins8));
        accum_mins_16(scales16, q8, acc, i, -GGML_FP16_TO_FP32(x[i].dmin));

        scales8 = vandq_u8(scales_and_mins, vdupq_n_u8(0xf));
    }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        process_scales(i, q8, acc);
        int16x8x2_t scales16;
        scales16.val[0] = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(scales8)));
        scales16.val[1] = vmovl_s8(vget_high_s8(vreinterpretq_s8_u8(scales8)));
        return make_wider(scales16);
    }

    template <typename Q8>
    inline void compute(const Q8& q8, int i, int j, int32x4_t * sumi) {
        auto m1 = vdupq_n_u8(1);
        auto shuffle = vdupq_n_u8(8*j);
        bits.b1.val[0] = vmulq_u8(bits.b1.val[0], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[1] = vmulq_u8(bits.b1.val[1], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[2] = vmulq_u8(bits.b1.val[2], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[3] = vmulq_u8(bits.b1.val[3], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[0] = vmulq_u8(bits.b2.val[0], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[1] = vmulq_u8(bits.b2.val[1], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[2] = vmulq_u8(bits.b2.val[2], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[3] = vmulq_u8(bits.b2.val[3], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b1.val[0]), q8b_1.val[0]),
                    vreinterpretq_s8_u8(bits.b1.val[1]), q8b_1.val[1]);

            auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b1.val[2]), q8b_2.val[0]),
                    vreinterpretq_s8_u8(bits.b1.val[3]), q8b_2.val[1]);

            auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b2.val[0]), q8b_3.val[0]),
                    vreinterpretq_s8_u8(bits.b2.val[1]), q8b_3.val[1]);

            auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b2.val[2]), q8b_4.val[0]),
                    vreinterpretq_s8_u8(bits.b2.val[3]), q8b_4.val[1]);
        }
    }

    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
    }

    uint32_t aux32[4];

    uint8x16_t scales8;

    Q2bits bits;

    float d;
};

// ============================= i-quants

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {

    static int8x16_t load_values() {
        static const int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        return vld1q_s8(iq4nl_values);
    }

    DequantizerIQ4XS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(load_values()) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    inline void new_row(int ix) { x = (const block_iq4_xs *)((const char *)vx + bx*ix); }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        d = GGML_FP16_TO_FP32(x[i].d);
        const uint16_t scales_h = x[i].scales_h;
        const uint16_t * scales_l = (const uint16_t *)x[i].scales_l;
        aux32[0] = scales_l[0] | (scales_l[1] << 16);
        aux32[1] = aux32[0] >> 4;
        // scl is ordered as 0, 2, 4, 6, 1, 3, 5, 7
        uint8x8_t scl8 = vand_u8(vld1_u8((const uint8_t *)aux32), vdup_n_u8(0xf));
        uint16_t * aux16 = (uint16_t *)aux32;
        aux16[0] = scales_h << 4; aux16[1] = scales_h << 2; aux16[2] = scales_h; aux16[3] = scales_h >> 2;
        // sch is ordered as 0, 4, 1, 5, 2, 6, 3, 7
        uint8x8_t sch8 = vand_u8(vld1_u8((const uint8_t *)aux16), vdup_n_u8(0x30));
        int8x8_t scales8 = vadd_s8(vreinterpret_s8_u8(vorr_u8(scl8, vtbl1_u8(sch8, vreinterpret_u8_u32(hshuff)))), vdup_n_s8(-32));
        // shuffle 0, 2, 4, 6, 1, 3, 5, 7 -> 0, 1, 2, 3, 4, 5, 6, 7
        scales8 = vtbl1_s8(scales8, vreinterpret_s8_u32(hshuff));
        int16x8_t scales16 = vmovl_s8(scales8);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs+64*j);
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values, bits.b1.val[k]));
            bits.b2.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values, bits.b2.val[k]));
        }
    }

    Q4bits bits;
    const int8x16_t values;
    uint32_t aux32[2];

    constexpr static uint32x2_t hshuff = {0x05010400, 0x07030602};

    float d;
};

struct SimpleBits {
    uint8x16x4_t b1;
    uint8x16x4_t b2;
};

IQK_ALWAYS_INLINE int32x4x2_t prepare_scales_8(const uint32x4_t& v1, const uint32x4_t& v2) {
    int32x4x2_t scales;
    auto one = vdupq_n_u32(1);
    scales.val[0] = vreinterpretq_s32_u32(vsliq_n_u32(one, vshrq_n_u32(v1, 28), 1));
    scales.val[1] = vreinterpretq_s32_u32(vsliq_n_u32(one, vshrq_n_u32(v2, 28), 1));
    return scales;
}

inline void apply_signs_2(uint8x16_t * b, const uint64_t * signs, uint32_t sidx) {
    auto s1 = vcombine_s8(vld1_s8((const int8_t *)(signs + ((sidx >> 0) & 127))), vld1_s8((const int8_t *)(signs + ((sidx >> 7) & 127))));
    auto s2 = vcombine_s8(vld1_s8((const int8_t *)(signs + ((sidx >>14) & 127))), vld1_s8((const int8_t *)(signs + ((sidx >>21) & 127))));
    b[0] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[0]), s1));
    b[1] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[1]), s2));
}

IQK_ALWAYS_INLINE int32x4_t prepare_scales_8(const uint32x4_t& v1) {
    return vreinterpretq_s32_u32(vsliq_n_u32(vdupq_n_u32(1), vshrq_n_u32(v1, 28), 1));
}

struct DequantizerIQ2XXS final : public BaseDequantizer<block_iq2_xxs> {
    DequantizerIQ2XXS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    IQK_ALWAYS_INLINE float new_block(int i) const { return 0.125f * GGML_FP16_TO_FP32(x[i].d); }

    inline int32x4_t unpack(int i, int j, uint8x16_t * q) const {
        auto data = vld1q_u32_x2((const uint32_t *)(x[i].qs + 16*j));
        prepare_all(data, q);
        return prepare_scales_8(vuzp2q_u32(data.val[0], data.val[1]));
    }

private:

    static inline void prepare2(uint8x16_t * b, const uint32_t * bits, const uint64_t * signs) {
        const uint8_t * idx = (const uint8_t *)bits;
        b[0] = vreinterpretq_u8_u64(uint64x2_t{iq2xxs_grid[idx[0]], iq2xxs_grid[idx[1]]});
        b[1] = vreinterpretq_u8_u64(uint64x2_t{iq2xxs_grid[idx[2]], iq2xxs_grid[idx[3]]});
        apply_signs_2(b, signs, bits[1]);
    }

    inline static void prepare_all(const uint32x4x2_t& data, uint8x16_t * quants) {
        const uint32_t * q2 = (const uint32_t *)data.val;
        prepare2(quants+0, q2+0, keven_signs);
        prepare2(quants+2, q2+2, keven_signs);
        prepare2(quants+4, q2+4, keven_signs);
        prepare2(quants+6, q2+6, keven_signs);
    }
};

inline int32x4x4_t prepare_4bit_scales16(const uint8_t * sc) {
    auto aux = vld1_u8(sc);
    auto scales_l = vand_u8(aux, vdup_n_u8(0xf));
    auto scales_h = vshr_n_u8(aux, 4);
    auto aux1 = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));

    auto scales8 = vreinterpretq_s8_u8(vorrq_u8(vshlq_n_u8(aux1, 1), vdupq_n_u8(1)));
    int16x8x2_t scales16 = { vmovl_s8(vget_low_s8(scales8)), vmovl_s8(vget_high_s8(scales8)) };
    return make_wider(scales16);
}

struct DequantizerIQ2XS final : public BaseDequantizer<block_iq2_xs> {
    DequantizerIQ2XS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    SimpleBits bits;
    float d;

    inline int32x4x4_t new_block(int i) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        prepare_internal(i, 0);
        return prepare_4bit_scales16(x[i].scales);
    }

    inline void prepare(int i, int j) {
        if (j == 1) prepare_internal(i, 1);
    }

private:

    static void make2(const uint16_t * qs, uint8x16_t * b) {
        auto v1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xs_grid + (qs[0] & 511))), vld1_s8((const int8_t *)(iq2xs_grid + (qs[1] & 511))));
        auto v2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xs_grid + (qs[2] & 511))), vld1_s8((const int8_t *)(iq2xs_grid + (qs[3] & 511))));
        auto s1 = vcombine_s8(vld1_s8((const int8_t *)(keven_signs + (qs[0] >> 9))), vld1_s8((const int8_t *)(keven_signs + (qs[1] >> 9))));
        auto s2 = vcombine_s8(vld1_s8((const int8_t *)(keven_signs + (qs[2] >> 9))), vld1_s8((const int8_t *)(keven_signs + (qs[3] >> 9))));
        b[0] = vreinterpretq_u8_s8(vmulq_s8(v1, s1));
        b[1] = vreinterpretq_u8_s8(vmulq_s8(v2, s2));
    }

    inline static void make4(const uint16_t * qs, uint8x16_t * b) {
        make2(qs + 0, b + 0);
        make2(qs + 4, b + 2);
    }

    IQK_ALWAYS_INLINE void prepare_internal(int i, int j) {
        make4(x[i].qs + 16*j + 0, bits.b1.val);
        make4(x[i].qs + 16*j + 8, bits.b2.val);
    }

};

// So, I hate to include this table, but with the GCC 12.3 compiler
// bundled in the Cosmopolitan tools, loading the unpacked sign bytes
// from this table using the packed 8 sign bits as index is faster than
// using the standard trick of vceqq_u8(vandq_u8(bits, mask), mask) to
// expand the bits to bytes.
static const uint64_t kall_signs[256] = {
    0x0101010101010101, 0x01010101010101ff, 0x010101010101ff01, 0x010101010101ffff,
    0x0101010101ff0101, 0x0101010101ff01ff, 0x0101010101ffff01, 0x0101010101ffffff,
    0x01010101ff010101, 0x01010101ff0101ff, 0x01010101ff01ff01, 0x01010101ff01ffff,
    0x01010101ffff0101, 0x01010101ffff01ff, 0x01010101ffffff01, 0x01010101ffffffff,
    0x010101ff01010101, 0x010101ff010101ff, 0x010101ff0101ff01, 0x010101ff0101ffff,
    0x010101ff01ff0101, 0x010101ff01ff01ff, 0x010101ff01ffff01, 0x010101ff01ffffff,
    0x010101ffff010101, 0x010101ffff0101ff, 0x010101ffff01ff01, 0x010101ffff01ffff,
    0x010101ffffff0101, 0x010101ffffff01ff, 0x010101ffffffff01, 0x010101ffffffffff,
    0x0101ff0101010101, 0x0101ff01010101ff, 0x0101ff010101ff01, 0x0101ff010101ffff,
    0x0101ff0101ff0101, 0x0101ff0101ff01ff, 0x0101ff0101ffff01, 0x0101ff0101ffffff,
    0x0101ff01ff010101, 0x0101ff01ff0101ff, 0x0101ff01ff01ff01, 0x0101ff01ff01ffff,
    0x0101ff01ffff0101, 0x0101ff01ffff01ff, 0x0101ff01ffffff01, 0x0101ff01ffffffff,
    0x0101ffff01010101, 0x0101ffff010101ff, 0x0101ffff0101ff01, 0x0101ffff0101ffff,
    0x0101ffff01ff0101, 0x0101ffff01ff01ff, 0x0101ffff01ffff01, 0x0101ffff01ffffff,
    0x0101ffffff010101, 0x0101ffffff0101ff, 0x0101ffffff01ff01, 0x0101ffffff01ffff,
    0x0101ffffffff0101, 0x0101ffffffff01ff, 0x0101ffffffffff01, 0x0101ffffffffffff,
    0x01ff010101010101, 0x01ff0101010101ff, 0x01ff01010101ff01, 0x01ff01010101ffff,
    0x01ff010101ff0101, 0x01ff010101ff01ff, 0x01ff010101ffff01, 0x01ff010101ffffff,
    0x01ff0101ff010101, 0x01ff0101ff0101ff, 0x01ff0101ff01ff01, 0x01ff0101ff01ffff,
    0x01ff0101ffff0101, 0x01ff0101ffff01ff, 0x01ff0101ffffff01, 0x01ff0101ffffffff,
    0x01ff01ff01010101, 0x01ff01ff010101ff, 0x01ff01ff0101ff01, 0x01ff01ff0101ffff,
    0x01ff01ff01ff0101, 0x01ff01ff01ff01ff, 0x01ff01ff01ffff01, 0x01ff01ff01ffffff,
    0x01ff01ffff010101, 0x01ff01ffff0101ff, 0x01ff01ffff01ff01, 0x01ff01ffff01ffff,
    0x01ff01ffffff0101, 0x01ff01ffffff01ff, 0x01ff01ffffffff01, 0x01ff01ffffffffff,
    0x01ffff0101010101, 0x01ffff01010101ff, 0x01ffff010101ff01, 0x01ffff010101ffff,
    0x01ffff0101ff0101, 0x01ffff0101ff01ff, 0x01ffff0101ffff01, 0x01ffff0101ffffff,
    0x01ffff01ff010101, 0x01ffff01ff0101ff, 0x01ffff01ff01ff01, 0x01ffff01ff01ffff,
    0x01ffff01ffff0101, 0x01ffff01ffff01ff, 0x01ffff01ffffff01, 0x01ffff01ffffffff,
    0x01ffffff01010101, 0x01ffffff010101ff, 0x01ffffff0101ff01, 0x01ffffff0101ffff,
    0x01ffffff01ff0101, 0x01ffffff01ff01ff, 0x01ffffff01ffff01, 0x01ffffff01ffffff,
    0x01ffffffff010101, 0x01ffffffff0101ff, 0x01ffffffff01ff01, 0x01ffffffff01ffff,
    0x01ffffffffff0101, 0x01ffffffffff01ff, 0x01ffffffffffff01, 0x01ffffffffffffff,
    0xff01010101010101, 0xff010101010101ff, 0xff0101010101ff01, 0xff0101010101ffff,
    0xff01010101ff0101, 0xff01010101ff01ff, 0xff01010101ffff01, 0xff01010101ffffff,
    0xff010101ff010101, 0xff010101ff0101ff, 0xff010101ff01ff01, 0xff010101ff01ffff,
    0xff010101ffff0101, 0xff010101ffff01ff, 0xff010101ffffff01, 0xff010101ffffffff,
    0xff0101ff01010101, 0xff0101ff010101ff, 0xff0101ff0101ff01, 0xff0101ff0101ffff,
    0xff0101ff01ff0101, 0xff0101ff01ff01ff, 0xff0101ff01ffff01, 0xff0101ff01ffffff,
    0xff0101ffff010101, 0xff0101ffff0101ff, 0xff0101ffff01ff01, 0xff0101ffff01ffff,
    0xff0101ffffff0101, 0xff0101ffffff01ff, 0xff0101ffffffff01, 0xff0101ffffffffff,
    0xff01ff0101010101, 0xff01ff01010101ff, 0xff01ff010101ff01, 0xff01ff010101ffff,
    0xff01ff0101ff0101, 0xff01ff0101ff01ff, 0xff01ff0101ffff01, 0xff01ff0101ffffff,
    0xff01ff01ff010101, 0xff01ff01ff0101ff, 0xff01ff01ff01ff01, 0xff01ff01ff01ffff,
    0xff01ff01ffff0101, 0xff01ff01ffff01ff, 0xff01ff01ffffff01, 0xff01ff01ffffffff,
    0xff01ffff01010101, 0xff01ffff010101ff, 0xff01ffff0101ff01, 0xff01ffff0101ffff,
    0xff01ffff01ff0101, 0xff01ffff01ff01ff, 0xff01ffff01ffff01, 0xff01ffff01ffffff,
    0xff01ffffff010101, 0xff01ffffff0101ff, 0xff01ffffff01ff01, 0xff01ffffff01ffff,
    0xff01ffffffff0101, 0xff01ffffffff01ff, 0xff01ffffffffff01, 0xff01ffffffffffff,
    0xffff010101010101, 0xffff0101010101ff, 0xffff01010101ff01, 0xffff01010101ffff,
    0xffff010101ff0101, 0xffff010101ff01ff, 0xffff010101ffff01, 0xffff010101ffffff,
    0xffff0101ff010101, 0xffff0101ff0101ff, 0xffff0101ff01ff01, 0xffff0101ff01ffff,
    0xffff0101ffff0101, 0xffff0101ffff01ff, 0xffff0101ffffff01, 0xffff0101ffffffff,
    0xffff01ff01010101, 0xffff01ff010101ff, 0xffff01ff0101ff01, 0xffff01ff0101ffff,
    0xffff01ff01ff0101, 0xffff01ff01ff01ff, 0xffff01ff01ffff01, 0xffff01ff01ffffff,
    0xffff01ffff010101, 0xffff01ffff0101ff, 0xffff01ffff01ff01, 0xffff01ffff01ffff,
    0xffff01ffffff0101, 0xffff01ffffff01ff, 0xffff01ffffffff01, 0xffff01ffffffffff,
    0xffffff0101010101, 0xffffff01010101ff, 0xffffff010101ff01, 0xffffff010101ffff,
    0xffffff0101ff0101, 0xffffff0101ff01ff, 0xffffff0101ffff01, 0xffffff0101ffffff,
    0xffffff01ff010101, 0xffffff01ff0101ff, 0xffffff01ff01ff01, 0xffffff01ff01ffff,
    0xffffff01ffff0101, 0xffffff01ffff01ff, 0xffffff01ffffff01, 0xffffff01ffffffff,
    0xffffffff01010101, 0xffffffff010101ff, 0xffffffff0101ff01, 0xffffffff0101ffff,
    0xffffffff01ff0101, 0xffffffff01ff01ff, 0xffffffff01ffff01, 0xffffffff01ffffff,
    0xffffffffff010101, 0xffffffffff0101ff, 0xffffffffff01ff01, 0xffffffffff01ffff,
    0xffffffffffff0101, 0xffffffffffff01ff, 0xffffffffffffff01, 0xffffffffffffffff,
};

struct SignHelper {

    IQK_ALWAYS_INLINE void apply_signs_1x(uint8x16_t * b, const uint8_t * sign_bits) const {
        auto s = vreinterpretq_s8_u64(uint64x2_t{kall_signs[sign_bits[0]], kall_signs[sign_bits[1]]});
        // Normally we would expect this to be faster, but it isn't.
        // auto aux = vcombine_u8(vdup_n_u8(sign_bits[0]), vdup_n_u8(sign_bits[1]));
        // auto s = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(aux, smask), smask), m1));
        b[0] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[0]), s));
    }

    // We would need these two if we weren't loading from the unpacked sign table.
    //const uint8x16_t smask = vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201));
    //const uint8x16_t m1    = vdupq_n_u8(1);
};

struct DequantizerIQ2S final : public BaseDequantizer<block_iq2_s> {
    DequantizerIQ2S(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    SimpleBits bits;
    float d;

    inline int32x4x4_t new_block(int i) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        prepare_internal(i, 0, bits);
        return prepare_4bit_scales16(x[i].scales);
    }

    inline void prepare(int i, int j) {
        if (j == 1) prepare_internal(i, 1, bits);
    }

private:

    static void make4(const SignHelper& sh, const uint8_t * sign_bits, const uint8_t * qs, const uint8_t * qh, uint8x16_t * b) {
        uint32_t aux32[2];
        const uint16_t * aux16 = (const uint16_t *)aux32;
        for (int k = 0; k < 2; ++k) {
            aux32[1] = (qh[k] << 4) | (qh[k] << 18);
            aux32[0] = (aux32[1] << 4) & 0x03000300;
            aux32[1] &= 0x03000300;
            b[2*k+0] = vcombine_u8(vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+0] | aux16[0]))),
                                   vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+1] | aux16[1]))));
            b[2*k+1] = vcombine_u8(vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+2] | aux16[2]))),
                                   vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+3] | aux16[3]))));
            sh.apply_signs_1x(b+2*k+0, sign_bits); sign_bits += 2;
            sh.apply_signs_1x(b+2*k+1, sign_bits); sign_bits += 2;
        }
    }

    void prepare_internal(int i, int j, SimpleBits& sb) {

        const auto * qs = x[i].qs + 16*j;
        const auto * qh = x[i].qh + 4*j;
        const auto * sign_bits = qs + QK_K/8;

        make4(sh, sign_bits+0, qs+0, qh+0, sb.b1.val);
        make4(sh, sign_bits+8, qs+8, qh+2, sb.b2.val);
    }

    SignHelper sh;
};

struct DequantizerIQ3XXS final : public BaseDequantizer<block_iq3_xxs> {
    DequantizerIQ3XXS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    IQK_ALWAYS_INLINE float new_block(int i) const { return 0.25f * GGML_FP16_TO_FP32(x[i].d); }

    inline int32x4_t unpack(int i, int j, uint8x16_t * q) const {
        auto q3data = vld1q_u8_x2(x[i].qs + 32*j);
        auto gas = vld1q_u32((const uint32_t *)(x[i].qs + QK_K/4 + 16*j));
        prepare_block((const uint8_t *)q3data.val, (const uint32_t *)&gas, q);
        return prepare_scales_8(gas);
    }

private:

    inline static void make2(const uint8_t * q3, const uint32_t sidx, uint8x16_t * b) {
        b[0] = vreinterpretq_u8_u32(uint32x4_t{iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]]});
        b[1] = vreinterpretq_u8_u32(uint32x4_t{iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]]});
        apply_signs_2(b, keven_signs, sidx);
    }
    inline static void prepare_block(const uint8_t * q3, const uint32_t * signs, uint8x16_t * quants) {
        make2(q3+ 0, signs[0], quants + 0);
        make2(q3+ 8, signs[1], quants + 2);
        make2(q3+16, signs[2], quants + 4);
        make2(q3+24, signs[3], quants + 6);
    }
};

struct DequantizerIQ3S final : public BaseDequantizer<block_iq3_s> {
    DequantizerIQ3S(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    SimpleBits bits;
    float d;

    inline int32x4x2_t new_block(int i) {
        d = GGML_FP16_TO_FP32(x[i].d);
        uint32_t scales32[2];
        auto qs = vld1q_u8_x2(x[i].qs);
        auto signs = vld1q_u8(x[i].signs);

        prepare_block((const uint8_t *)qs.val, x[i].qh, (const uint8_t *)&signs);

        std::memcpy(scales32, x[i].scales, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
        auto scales8 = vld1_u8((const uint8_t *)scales32); // 0, 2, 4, 6, 1, 3, 5, 7
        scales8 = vtbl1_u8(scales8, vreinterpret_u8_u64(vdup_n_u64(0x0703060205010400)));
        auto scales16 = vreinterpretq_s16_u16(vmovl_u8(scales8));
        int32x4x2_t scales;
        scales.val[0] = vmovl_s16(vget_low_s16(scales16));
        scales.val[1] = vmovl_s16(vget_high_s16(scales16));
        return scales;
    }

    inline void prepare(int i, int j) {
        if (j == 1) {
            auto qs = vld1q_u8_x2(x[i].qs + 32);
            auto signs = vld1q_u8(x[i].signs + 16);
            prepare_block((const uint8_t *)qs.val, x[i].qh + 4, (const uint8_t *)&signs);
        }
    }

private:

    static inline void make2(const SignHelper& sh, const uint8_t * sign_bits, const uint16x8_t& idx_l, uint8_t qh,
            const int16x8_t& hshift, uint8x16_t * b) {
        auto vindex = vorrq_u16(idx_l, vandq_u16(vshlq_u16(vdupq_n_u16(qh), hshift), vdupq_n_u16(256)));
        const uint16_t * idx = (const uint16_t *)&vindex;
        b[0] = vreinterpretq_u8_u32(uint32x4_t{iq3s_grid[idx[0]], iq3s_grid[idx[1]], iq3s_grid[idx[2]], iq3s_grid[idx[3]]});
        sh.apply_signs_1x(b+0, sign_bits+0);
        b[1] = vreinterpretq_u8_u32(uint32x4_t{iq3s_grid[idx[4]], iq3s_grid[idx[5]], iq3s_grid[idx[6]], iq3s_grid[idx[7]]});
        sh.apply_signs_1x(b+1, sign_bits+2);
    }
    static inline void make4(const SignHelper& sh, const uint8_t * sign_bits, const uint8_t * qs, const uint8_t * qh,
            const int16x8_t& hshift, uint8x16_t * b) {
        auto idx_l = vld1q_u8(qs);
        make2(sh, sign_bits+0, vmovl_u8(vget_low_u8 (idx_l)), qh[0], hshift, b+0);
        make2(sh, sign_bits+4, vmovl_u8(vget_high_u8(idx_l)), qh[1], hshift, b+2);
    }

    static int16x8_t load_shift() {
        static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        return vld1q_s16(k_shift);
    }

    inline void prepare_block(const uint8_t * qs, const uint8_t * qh, const uint8_t * sign_bits) {
        auto signs = vld1q_u8(sign_bits);
        auto s = (const uint8_t *)&signs;
        make4(sh, s + 0, qs+ 0, qh+0, hshift, bits.b1.val);
        make4(sh, s + 8, qs+16, qh+2, hshift, bits.b2.val);
    }

    SignHelper sh;
    const int16x8_t hshift = load_shift();

};

template <int nrc_y, typename Dequantizer>
IQK_NOINLINE void mul_mat_qX_K_q8_K_IQXXS(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_K> q8(info);
    Dequantizer deq(vx, bx, nrc_y);
    uint8x16_t  qx[8];
    int32x4_t   sumi[nrc_y];
    float32x4_t acc[nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);
        for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb; ++i) {
            float d = deq.new_block(i);
            auto scales = deq.unpack(i, 0, qx);
#pragma GCC unroll 8
            for (int iy = 0; iy < nrc_y; ++iy) {
                sumi[iy] = vdupq_n_s32(0);
                compute_8_blocks((const int8x16_t *)qx, q8, scales, iy, i, 0, sumi[iy]);
            }
            scales = deq.unpack(i, 1, qx);
#pragma GCC unroll 8
            for (int iy = 0; iy < nrc_y; ++iy) {
                compute_8_blocks((const int8x16_t *)qx, q8, scales, iy, i, 1, sumi[iy]);
                acc[iy] = vmlaq_f32(acc[iy], vdupq_n_f32(d*q8.scale(iy, i)), vcvtq_f32_s32(sumi[iy]));
            }
        }
#pragma GCC unroll 8
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}

// =========================================== Legacy quants

template <typename Block>
inline float16x4_t load_scales_q0(const Block * x, ggml_half * aux) {
    for (int k = 0; k < 4; ++k) aux[k] = x[k].d;
    return vld1_f16((const float16_t *)aux);
}

template <typename Block>
inline float16x8_t load_scales_q1(const Block * x, ggml_half * aux) {
    if constexpr (std::is_same_v<Block, block_q8_1>) {
        for (int k = 0; k < 4; ++k) { aux[k] = x[k].d; aux[k+4] = x[k].s; }
    } else {
        for (int k = 0; k < 4; ++k) { aux[k] = x[k].d; aux[k+4] = x[k].m; }
    }
    return vld1q_f16((const float16_t *)aux);
}

struct Q4LegacyBits {
    template <typename Block>
    inline void prepare(const Block * x) {
        for (int i = 0; i < 4; ++i) {
            auto q4bits = vld1q_u8(x[i].qs);
            b[2*i+0] = vreinterpretq_s8_u8(vandq_u8(q4bits, m4b));
            b[2*i+1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits, 4));
        }
    }
    inline void prepare1(const uint8_t * qs, int8x16_t * q) const {
        auto q4bits = vld1q_u8(qs);
        q[0] = vreinterpretq_s8_u8(vandq_u8(q4bits, m4b));
        q[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits, 4));
    }
    inline void prepare1(const uint8_t * qs) {
        prepare1(qs, b);
    }
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    int8x16_t b[8];
};

// One would think this commented out version would do better than the one below
// because it offers more opportunities to execute instructions in parallel.
// Instead, it runs significantly slower. Why? If the compiler is running out of vector registers
// cannot it just do the sequential version below on its own?
//inline int32x4_t sum_4_blocks(const int8x16_t * b, const int8_t * qs) {
//    const auto q8b_1 = vld1q_s8_x2(qs + 0);
//    auto p12 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[0], q8b_1.val[0]), b[1], q8b_1.val[1]);
//    const auto q8b_2 = vld1q_s8_x2(qs + 32);
//    auto p34 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[2], q8b_2.val[0]), b[3], q8b_2.val[1]);
//    auto p1234 = vpaddq_s32(p12, p34);
//    const auto q8b_3 = vld1q_s8_x2(qs + 64);
//    auto p56 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[4], q8b_3.val[0]), b[5], q8b_3.val[1]);
//    const auto q8b_4 = vld1q_s8_x2(qs + 96);
//    auto p78 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[6], q8b_4.val[0]), b[7], q8b_4.val[1]);
//    return vpaddq_s32(p1234, vpaddq_s32(p56, p78));
//}

inline int32x4_t sum_4_blocks(const int8x16_t * b, const int8_t * qs) {
    auto q8b = vld1q_s8_x2(qs + 0);
    auto p12 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[0], q8b.val[0]), b[1], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 32);
    auto p34 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[2], q8b.val[0]), b[3], q8b.val[1]);
    auto p1234 = vpaddq_s32(p12, p34);
    q8b = vld1q_s8_x2(qs + 64);
    auto p56 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[4], q8b.val[0]), b[5], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 96);
    auto p78 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[6], q8b.val[0]), b[7], q8b.val[1]);
    return vpaddq_s32(p1234, vpaddq_s32(p56, p78));
}

template <int nrc> struct Q80 {

    constexpr static int nrc_y = nrc;

    Q80(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_0 *)info.src1_row(iy);
    }

    inline const int8_t * quant_data(int iy, int i) const {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y[iy] + i;
        return y4->qs;
    }

    inline float16x4_t load_scales(int iy, int i) const {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y[iy] + i;
        return vld1_f16((const float16_t *)y4->d);
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq, float16x4_t * sc16, float32x4_t * /*acc*/) const {
        auto qx_scales = deq.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            sc16[iy] = vmul_f16(qx_scales, q8_scales);
        }
    }

    template <typename Dequantizer>
    inline void process_1_block(int i, Dequantizer& deq, float32x4_t * acc) const {
        deq.prepare1(i);
        float d = GGML_FP16_TO_FP32(deq.x[i].d);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8b = vld1q_s8_x2(y[iy][i].qs);
            auto p = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), deq.bits.b[0], q8b.val[0]), deq.bits.b[1], q8b.val[1]);
            acc[iy] = vmlaq_f32(acc[iy], vdupq_n_f32(d*GGML_FP16_TO_FP32(y[iy][i].d)), vcvtq_f32_s32(p));
        }
    }

    const block_q8_0 * y[nrc_y];
};

template <int nrc> struct Q81 {

    constexpr static int nrc_y = nrc;

    Q81(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_1 *)info.src1_row(iy);
    }

    inline const int8_t * quant_data(int iy, int i) const {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y[iy] + i;
        return y4->qs;
    }

    inline float16x8_t load_scales(int iy, int i) const {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y[iy] + i;
        return vld1q_f16((const float16_t *)y4->d);
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq, float16x4_t * sc16, float32x4_t * acc) const {
        auto qx_scales = deq.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            auto m = vmul_f16(vget_high_f16(qx_scales), vget_high_f16(q8_scales));
            acc[iy] = vaddq_f32(acc[iy], vcvt_f32_f16(m));
            sc16[iy] = vmul_f16(vget_low_f16(qx_scales), vget_low_f16(q8_scales));
        }
    }

    template <typename Dequantizer>
    inline void process_1_block(int i, Dequantizer& deq, float32x4_t * acc) const {
        deq.prepare1(i);
        float d = GGML_FP16_TO_FP32(deq.x[i].d), m = 0.25f*GGML_FP16_TO_FP32(deq.x[i].m);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8b = vld1q_s8_x2(y[iy][i].qs);
            auto p = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), deq.bits.b[0], q8b.val[0]), deq.bits.b[1], q8b.val[1]);
            acc[iy] = vmlaq_f32(acc[iy], vdupq_n_f32(d*GGML_FP16_TO_FP32(y[iy][i].d)), vcvtq_f32_s32(p));
            acc[iy] = vaddq_f32(acc[iy], vdupq_n_f32(m*GGML_FP16_TO_FP32(y[iy][i].s)));
        }
    }

    const block_q8_1 * y[nrc_y];
};

template <typename block_q>
struct BaseLegacyDequantizer {

    BaseLegacyDequantizer(const void * vx, size_t bx) : vx(vx), x(nullptr), bx(bx) {}

    inline void new_row(int ix) { x = (const block_q *)((const char *)vx + bx*ix); }

    Q4LegacyBits bits;

    const void * vx;
    const block_q * x;
    size_t bx;
};

struct DequantizerQ40 final : public BaseLegacyDequantizer<block_q4_0> {

    DequantizerQ40(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        q[0] = vaddq_s8(q[0], m8);
        q[1] = vaddq_s8(q[1], m8);
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }

    const int8x16_t m8 = vdupq_n_s8(-8);
    //ggml_half aux[4];
};

struct DequantizerQ41 : public BaseLegacyDequantizer<block_q4_1> {

    DequantizerQ41(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i) {
        bits.prepare1(x[i].qs);
    }

    inline float16x8_t new_block(int i) {
        uint32_t aux32[4];
        const uint32_t * s32 = (const uint32_t *)&x[4*i].d;
        for (int k = 0; k < 4; ++k) {
            aux32[k] = *s32; s32 += sizeof(block_q4_1)/4;
            bits.prepare1(x[4*i+k].qs, bits.b + 2*k);
        }
        return vreinterpretq_f16_u8(vqtbl1q_u8(vld1q_u8((const uint8_t *)aux32), vreinterpretq_u8_u64(shuffle)));
    }
    // Leaving this commented out attempt to be reminded that I already tried this.
    // It has basically the same performance as the version above.
    //inline float16x8_t new_block(int i) {
    //    uint32x4_t scales = {};
    //    const block_q4_1 * xi = x + 4*i;
    //    const uint32_t * s32 = (const uint32_t *)&xi->d;
    //    scales = vsetq_lane_u32(*s32, scales, 0); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[0].qs, bits.b + 0);
    //    scales = vsetq_lane_u32(*s32, scales, 1); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[1].qs, bits.b + 2);
    //    scales = vsetq_lane_u32(*s32, scales, 2); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[2].qs, bits.b + 4);
    //    scales = vsetq_lane_u32(*s32, scales, 3);
    //    bits.prepare1(xi[3].qs, bits.b + 6);
    //    return vreinterpretq_f16_u8(vqtbl1q_u8(vreinterpretq_u8_u32(scales), vreinterpretq_u8_u64(shuffle)));
    //}

    const uint64x2_t shuffle = {0x0d0c090805040100, 0x0f0e0b0a07060302};
};

struct HighBit5Legacy {
    inline uint8x16_t to_bytes(const uint8_t * qh) const {
        uint8x16_t h = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(*(const uint16_t *)qh)), shuffle);
        return vceqq_u8(vandq_u8(h, vreinterpretq_u8_u64(mask)), vreinterpretq_u8_u64(mask));
    }
    inline uint8x16_t to_negated_bytes(const uint8_t * qh) const {
        uint8x16_t h = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(*(const uint16_t *)qh)), shuffle);
        return vceqq_u8(vandq_u8(h, vreinterpretq_u8_u64(mask)), vdupq_n_u8(0));
    }
    const uint64x2_t mask = vdupq_n_u64(0x8040201008040201);
    const uint8x16_t shuffle = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
};

struct DequantizerQ50 final : public BaseLegacyDequantizer<block_q5_0> {

    DequantizerQ50(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        auto qh = x[i].qh;
        q[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[0]), vandq_u8(mh, hbits.to_negated_bytes(qh+0))));
        q[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[1]), vandq_u8(mh, hbits.to_negated_bytes(qh+2))));
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }

    HighBit5Legacy hbits;

    const uint8x16_t mh = vdupq_n_u8(0xf0);

};

struct DequantizerQ80 final : public BaseLegacyDequantizer<block_q8_0> {

    DequantizerQ80(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i) {
        bits.b[0] = vld1q_s8(x[i].qs);
        bits.b[1] = vld1q_s8(x[i].qs+16);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            bits.b[2*k+0] = vld1q_s8(x[4*i+k].qs);
            bits.b[2*k+1] = vld1q_s8(x[4*i+k].qs+16);
        }
        return vld1_f16((const float16_t *)aux);
    }

};

struct DequantizerQ51 final : public BaseLegacyDequantizer<block_q5_1> {

    DequantizerQ51(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        auto qh = x[i].qh;
        q[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[0]), vandq_u8(mh, hbits.to_bytes(qh+0))));
        q[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[1]), vandq_u8(mh, hbits.to_bytes(qh+2))));
    }
    inline void prepare1(int i) {
        bits.prepare1(x[i].qs, bits.b);
    }

    inline float16x8_t new_block(int i) {
        uint32_t aux32[4];
        const uint32_t * s32 = (const uint32_t *)&x[4*i].d;
        for (int k = 0; k < 4; ++k) {
            aux32[k] = *s32; s32 += sizeof(block_q5_1)/4;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vreinterpretq_f16_u8(vqtbl1q_u8(vld1q_u8((const uint8_t *)aux32), vreinterpretq_u8_u64(shuffle)));
    }

    HighBit5Legacy hbits;

    const uint8x16_t mh = vdupq_n_u8(0x10);
    const uint64x2_t shuffle = {0x0d0c090805040100, 0x0f0e0b0a07060302};

};

template <typename Dequantizer, typename Q8>
inline void sum_4(int i, Dequantizer& deq, const Q8& q8, const float16x4_t * sc16, float32x4_t * acc) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto pall = sum_4_blocks(deq.bits.b, q8.quant_data(iy, i));
        auto scale = vcvt_f32_f16(sc16[iy]);
        acc[iy] = vmlaq_f32(acc[iy], scale, vcvtq_f32_s32(pall));
    }
}

template <typename Dequantizer, typename Q8>
inline void mul_mat_qX_Y_q8_Y(int n, Dequantizer& deq, Q8& q8, const DataInfo& info, int nrc_x) {
    const int nb = n / QK4_1;

    float16x4_t sc16[Q8::nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        float32x4_t acc[Q8::nrc_y];
        for (int iy = 0; iy < Q8::nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb/4; ++i) {
            q8.process_scales(i, deq, sc16, acc);
            sum_4(i, deq, q8, sc16, acc);
        }
        for (int i = 4*(nb/4); i < nb; ++i) {
            q8.process_1_block(i, deq, acc);
        }

        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}

template <typename Dequantizer, typename Q8>
inline void mul_mat_qX_Y_q8_Y_1(int n, Dequantizer& deq1, Dequantizer& deq2, Q8& q8, const DataInfo& info, int nrc_x) {
    const int nb = n / QK4_1;

    float16x4_t sc16[2];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq1.new_row(ix);
        deq2.new_row(ix);

        float32x4_t acc[2] = { vdupq_n_f32(0.f), vdupq_n_f32(0.f) };

        for (int i = 0; i < nb/8; ++i) {
            q8.process_scales(2*i+0, deq1, sc16+0, acc+0);
            q8.process_scales(2*i+1, deq2, sc16+1, acc+1);
            sum_4(2*i+0, deq1, q8, sc16+0, acc+0);
            sum_4(2*i+1, deq2, q8, sc16+1, acc+1);
        }
        for (int i = 2*(nb/8); i < nb/4; ++i) {
            q8.process_scales(i, deq1, sc16, acc);
            sum_4(i, deq1, q8, sc16, acc);
        }
        for (int i = 4*(nb/4); i < nb; ++i) {
            q8.process_1_block(i, deq1, acc);
        }

        info.store(ix, 0, vaddvq_f32(vaddq_f32(acc[0], acc[1])));
    }
}

template <typename Dequantizer, int nrc_y>
static void IQK_NOINLINE mul_mat_qX_1_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q81<nrc_y> q8(info);
    if constexpr (nrc_y == 1) {
        Dequantizer deq1(vx, bx), deq2(vx, bx);
        mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
    } else {
        Dequantizer deq(vx, bx);
        mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
    }
}

template <typename Dequantizer, int nrc_y>
static void IQK_NOINLINE mul_mat_qX_0_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q80<nrc_y> q8(info);
    if constexpr (nrc_y == 1) {
        Dequantizer deq1(vx, bx), deq2(vx, bx);
        mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
    } else {
        Dequantizer deq(vx, bx);
        mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
    }
}

template <typename Dequantizer>
static void IQK_NOINLINE mul_mat_qX_1_q8_1_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Dequantizer deq1(vx, bx), deq2(vx, bx);
    Q81<1> q8(info);
    mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
}

template <typename Dequantizer>
static void IQK_NOINLINE mul_mat_qX_0_q8_0_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Dequantizer deq1(vx, bx), deq2(vx, bx);
    Q80<1> q8(info);
    mul_mat_qX_Y_q8_Y(n, deq1, deq2, q8, info, nrc_x);
}

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
    if constexpr (std::is_same_v<Dequantizer, DequantizerQ40> || std::is_same_v<Dequantizer, DequantizerQ50> ||
                  std::is_same_v<Dequantizer, DequantizerQ80>) {
        m.funcs[0] = mul_mat_qX_0_q8_0<Dequantizer, 1>;
        m.funcs[1] = mul_mat_qX_0_q8_0<Dequantizer, 2>;
        m.funcs[2] = mul_mat_qX_0_q8_0<Dequantizer, 3>;
        m.funcs[3] = mul_mat_qX_0_q8_0<Dequantizer, 4>;
        m.funcs[4] = mul_mat_qX_0_q8_0<Dequantizer, 5>;
        m.funcs[5] = mul_mat_qX_0_q8_0<Dequantizer, 6>;
        m.funcs[6] = mul_mat_qX_0_q8_0<Dequantizer, 7>;
        m.funcs[7] = mul_mat_qX_0_q8_0<Dequantizer, 8>;
    }
    else if constexpr (std::is_same_v<Dequantizer, DequantizerQ41> || std::is_same_v<Dequantizer, DequantizerQ51>) {
        m.funcs[0] = mul_mat_qX_1_q8_1<Dequantizer, 1>;
        m.funcs[1] = mul_mat_qX_1_q8_1<Dequantizer, 2>;
        m.funcs[2] = mul_mat_qX_1_q8_1<Dequantizer, 3>;
        m.funcs[3] = mul_mat_qX_1_q8_1<Dequantizer, 4>;
        m.funcs[4] = mul_mat_qX_1_q8_1<Dequantizer, 5>;
        m.funcs[5] = mul_mat_qX_1_q8_1<Dequantizer, 6>;
        m.funcs[6] = mul_mat_qX_1_q8_1<Dequantizer, 7>;
        m.funcs[7] = mul_mat_qX_1_q8_1<Dequantizer, 8>;
    }
    else if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2XXS> || std::is_same_v<Dequantizer, DequantizerIQ3XXS>) {
        m.funcs[0] = mul_mat_qX_K_q8_K_IQXXS<1, Dequantizer>;
        m.funcs[1] = mul_mat_qX_K_q8_K_IQXXS<2, Dequantizer>;
        m.funcs[2] = mul_mat_qX_K_q8_K_IQXXS<3, Dequantizer>;
        m.funcs[3] = mul_mat_qX_K_q8_K_IQXXS<4, Dequantizer>;
        m.funcs[4] = mul_mat_qX_K_q8_K_IQXXS<5, Dequantizer>;
        m.funcs[5] = mul_mat_qX_K_q8_K_IQXXS<6, Dequantizer>;
        m.funcs[6] = mul_mat_qX_K_q8_K_IQXXS<7, Dequantizer>;
        m.funcs[7] = mul_mat_qX_K_q8_K_IQXXS<8, Dequantizer>;
    }
    else if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2S> ||
                       std::is_same_v<Dequantizer, DequantizerIQ3S> ||
                       std::is_same_v<Dequantizer, DequantizerIQ2XS>) {
        m.funcs[0] = mul_mat_qX_K_q8_K_IQ<1, Dequantizer>;
        m.funcs[1] = mul_mat_qX_K_q8_K_IQ<2, Dequantizer>;
        m.funcs[2] = mul_mat_qX_K_q8_K_IQ<3, Dequantizer>;
        m.funcs[3] = mul_mat_qX_K_q8_K_IQ<4, Dequantizer>;
        m.funcs[4] = mul_mat_qX_K_q8_K_IQ<5, Dequantizer>;
        m.funcs[5] = mul_mat_qX_K_q8_K_IQ<6, Dequantizer>;
        m.funcs[6] = mul_mat_qX_K_q8_K_IQ<7, Dequantizer>;
        m.funcs[7] = mul_mat_qX_K_q8_K_IQ<8, Dequantizer>;
    }
    else {
        m.funcs[0] = mul_mat_qX_K_q8_K_T<1, Dequantizer>;
        m.funcs[1] = mul_mat_qX_K_q8_K_T<2, Dequantizer>;
        m.funcs[2] = mul_mat_qX_K_q8_K_T<3, Dequantizer>;
        m.funcs[3] = mul_mat_qX_K_q8_K_T<4, Dequantizer>;
        m.funcs[4] = mul_mat_qX_K_q8_K_T<5, Dequantizer>;
        m.funcs[5] = mul_mat_qX_K_q8_K_T<6, Dequantizer>;
        m.funcs[6] = mul_mat_qX_K_q8_K_T<7, Dequantizer>;
        m.funcs[7] = mul_mat_qX_K_q8_K_T<8, Dequantizer>;
    }
}

bool MulMat::set_mul_mat(int typeA, int ne00, MulMat& m, int& row_size_q8, int Ny) {
    row_size_q8 = ggml_row_size(GGML_TYPE_Q8_K, ne00);

    (void)Ny;
    // Uncommenting out this would disable iqk_mul_mat for matrix x vector multiplications.
    //if (Ny == 1 && (typeA == GGML_TYPE_IQ2_XXS || typeA == GGML_TYPE_IQ2_XS || typeA == GGML_TYPE_IQ2_S ||
    //                typeA == GGML_TYPE_IQ3_XXS || typeA == GGML_TYPE_IQ3_S)) return false;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            MulMat::set_functions<DequantizerQ2K>(m);
            break;
        case GGML_TYPE_Q3_K:
            MulMat::set_functions<DequantizerQ3K>(m);
            break;
        case GGML_TYPE_Q4_K:
            MulMat::set_functions<DequantizerQ4K>(m);
            break;
        case GGML_TYPE_Q5_K:
            MulMat::set_functions<DequantizerQ5K>(m);
            break;
        case GGML_TYPE_Q6_K:
            MulMat::set_functions<DequantizerQ6K>(m);
            break;
        case GGML_TYPE_IQ4_XS:
            MulMat::set_functions<DequantizerIQ4XS>(m);
            break;
        case GGML_TYPE_IQ3_S:
            MulMat::set_functions<DequantizerIQ3S>(m);
            break;
        case GGML_TYPE_IQ3_XXS:
            MulMat::set_functions<DequantizerIQ3XXS>(m);
            break;
        case GGML_TYPE_IQ2_S:
            MulMat::set_functions<DequantizerIQ2S>(m);
            break;
        case GGML_TYPE_IQ2_XS:
            MulMat::set_functions<DequantizerIQ2XS>(m);
            break;
        case GGML_TYPE_IQ2_XXS:
            MulMat::set_functions<DequantizerIQ2XXS>(m);
            break;
        case GGML_TYPE_Q4_0:
            MulMat::set_functions<DequantizerQ40>(m);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q4_1:
            MulMat::set_functions<DequantizerQ41>(m);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;
        case GGML_TYPE_Q5_0:
            MulMat::set_functions<DequantizerQ50>(m);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        case GGML_TYPE_Q5_1:
            MulMat::set_functions<DequantizerQ51>(m);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_1, ne00);
            break;
        case GGML_TYPE_Q8_0:
            MulMat::set_functions<DequantizerQ80>(m);
            row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne00);
            break;
        default:
            return false;
    }
    return true;
}

}

#endif // __x86_64__ or __aarch64__
