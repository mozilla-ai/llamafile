// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Utility functions for sgemm tests
//
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// For ggml_vec_dot_f32 - the actual function llama.cpp uses in its fallback
#include "vec.h"

// Timing
static inline long long micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
}

// Benchmarking macro
#define BENCH(x, iterations) \
    do { \
        x; \
        __asm__ volatile("" ::: "memory"); \
        long long start = micros(); \
        for (int _i = 0; _i < (iterations); ++_i) { \
            __asm__ volatile("" ::: "memory"); \
            x; \
            __asm__ volatile("" ::: "memory"); \
        } \
        printf("%12lld us %s\n", (micros() - start + (iterations) - 1) / (iterations), #x); \
    } while (0)

// Macros
#define ROUNDUP(X, K) (((X) + (K) - 1) & -(K))

// Random number generation
static inline int rand32(void) {
    static unsigned long long lcg = 1;
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

static inline float float01(unsigned x) {
    return 1.f / 8388608 * ((x >> 9) + .5f);
}

static inline float numba(void) {
    return float01(rand32()) * 2.f - 1.f;
}

template <typename T>
void randomize(int m, int n, T *A, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[lda * j + i] = numba();
}

template <typename T, typename U>
void broadcast(T *A, int n, U x) {
    for (int i = 0; i < n; ++i)
        A[i] = x;
}

// Float utilities
namespace flt {

inline unsigned toint(float f) {
    union {
        float f;
        unsigned i;
    } u = {f};
    return u.i;
}

inline bool isnan(float f) {
    return (toint(f) & 0x7fffffff) > 0x7f800000;
}

} // namespace flt

// Reference BLAS implementation (ANSI C, double precision accumulation)
namespace ansiBLAS {

static constexpr int KN = 8;

union Vector {
    double v[KN];
};

inline Vector load(const float *p) {
    Vector x;
    for (int i = 0; i < KN; ++i)
        x.v[i] = p[i];
    return x;
}

inline Vector madd(Vector x, Vector y, Vector s) {
    for (int i = 0; i < KN; ++i)
        s.v[i] = fma(x.v[i], y.v[i], s.v[i]);
    return s;
}

inline float hsum(Vector x) {
    double s = 0;
    for (int i = 0; i < KN; ++i)
        s += x.v[i];
    return s;
}

struct ansiBLAS {
    ansiBLAS(int k, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int ith,
             int nth)
        : k(k), A(A), lda(lda), B(B), ldb(ldb), C(C), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(int m0, int m, int n0, int n) {
        int mc, nc, mp, np;
        if (m - m0 <= 0 || n - n0 <= 0)
            return;
        if (m - m0 >= 4 && n - n0 >= 3) {
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
        } else if (n - n0 >= 4) {
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
        } else if (m - m0 >= 4) {
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
        } else {
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    void gemm(int m0, int m, int n0, int n) {
        int ytiles = (m - m0) / RM;
        int xtiles = (n - n0) / RN;
        int tiles = xtiles * ytiles;
        int duty = (tiles + nth - 1) / nth;
        int start = duty * ith;
        int end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int job = start; job < end; ++job) {
            int ii = m0 + job / xtiles * RM;
            int jj = n0 + job % xtiles * RN;
            Vector Cv[RN][RM] = {};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load(A + lda * (ii + i) + l),
                                        load(B + ldb * (jj + j) + l),
                                        Cv[j][i]);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    const int k;
    const float *const A;
    const int lda;
    const float *const B;
    const int ldb;
    float *const C;
    const int ldc;
    const int ith;
    const int nth;
};

inline void sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C,
                  int ldc, int nth) {
#pragma omp parallel for
    for (int ith = 0; ith < nth; ++ith) {
        ansiBLAS tb{k, A, lda, B, ldb, C, ldc, ith, nth};
        tb.matmul(m, n);
    }
}

} // namespace ansiBLAS

// ==============================================================================
// ggmlBLAS: Uses llama.cpp's actual ggml_vec_dot_f32 function
// ==============================================================================
// This is what llama.cpp's fallback actually uses when llamafile_sgemm returns
// false. It calls ggml_vec_dot_f32 for each row-column pair.
//
// Note: ggml_vec_dot_f32 is a SIMD-optimized dot product implementation that
// uses SSE/AVX on x86 or NEON/SVE on ARM.
namespace ggmlBLAS {

// Matrix multiplication using ggml_vec_dot_f32
// C[m,n] = A^T[m,k] * B[k,n]
// A is stored as k x m (column-major), B as k x n (column-major)
// C is stored as m x n (column-major)
inline void sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb, float *C,
                  int ldc, int nth) {
#pragma omp parallel for collapse(2)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            // Compute C[i,j] = dot(A[:,i], B[:,j])
            // A[:,i] starts at A + i*lda
            // B[:,j] starts at B + j*ldb
            ggml_vec_dot_f32(k, &C[ldc * j + i], 0, A + i * lda, 0, B + j * ldb, 0, 1);
        }
    }
}

} // namespace ggmlBLAS
