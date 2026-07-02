// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// AVX-512F-optimized inline GEMM for CPU flash-attention tiled path
// (llamafile issue #975).
//
// Wraps ggml-cpu's static-inline simd_gemm (in simd-gemm.h) compiled
// with AVX-512 flags so the FMA inner loop uses 16-lane zmm registers
// and the wider 4x4 tile config (GEMM_RM=4, GEMM_RN=4 — 16 register-
// width FMAs per inner iteration) instead of the AVX2 6x2 tile that
// ops.cpp's default-ISA build gets. Roughly 2.7x peak FMA throughput
// on Skylake-X+ CPUs (4x16 vs 6x8 lane-FMAs).
//
// Called from the tiled FA function at ops.cpp:8660 (KQ = Q @ K) and
// :8740 (VKQ += KQ @ V). Runtime-dispatched via sgemm.cpp; falls back
// to upstream's simd_gemm (already inlined in ops.cpp) on CPUs without
// AVX-512F by returning false.

#ifdef __x86_64__

#include "sgemm.h"
#include "simd-gemm.h"

extern "C" bool llamafile_fa_simd_gemm_amd_avx512f(float *C, const float *A,
                                                    const float *B,
                                                    int M, int K, int N) {
    // Computes C[M x N] += A[M x K] * B[K x N]. simd_gemm in
    // simd-gemm.h is a header-static template; compiling this TU with
    // -Xx86_64-mavx512f gives the included simd_gemm the AVX-512
    // codegen we want.
    simd_gemm(C, A, B, M, K, N);
    return true;
}

#endif // __x86_64__
