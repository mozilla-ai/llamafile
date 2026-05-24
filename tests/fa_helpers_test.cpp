// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla Foundation
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
// Numerical-equivalence tests for the llamafile flash-attention helpers
// (issue #975).
//
// These tests compare our AVX-512-optimized llamafile_fa_* helpers
// against the upstream ggml-cpu reference implementations. Goal: ensure
// the alternative low-level math we ship via llamafile/sgemm.cpp's
// dispatch produces output that's numerically equivalent to the
// reference (bit-identical for the conversion helper, within ULP
// tolerance for the reduction helper).
//
// Run via `make check`. On CPUs without AVX-512F the helpers report
// "not supported" and the corresponding assertions are skipped (the
// test reports skipped count); the test still validates the dispatch
// table wiring.

#include "ggml.h"
#include "sgemm.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <random>
#include <vector>
#include <algorithm>

// ggml_vec_dot_f16 is defined in ggml-cpu/vec.cpp but not declared in a
// public header (it's reached through type_traits dispatch). Declare it
// here for the test.
extern "C" void ggml_vec_dot_f16(int n, float *s, size_t bs,
                                  ggml_fp16_t *x, size_t bx,
                                  ggml_fp16_t *y, size_t by, int nrc);

static int test_count = 0;
static int fail_count = 0;
static int skip_count = 0;

// Each TEST() body runs from main(), NOT at static-init time, because
// the upstream ggml helpers we compare against (ggml_fp16_to_fp32_row
// in particular) depend on a lookup table populated by ggml's graph-
// compute first-call init. We init that via a tiny graph-compute in
// main() before invoking any test. Tests register themselves into a
// list at static-init time, then main() iterates and runs them.
struct TestEntry {
    const char *name;
    void (*fn)();
};
static TestEntry g_tests[64];
static int g_test_count_registered = 0;

#define TEST(name) \
    static void test_##name(); \
    static struct TestRegister_##name { \
        TestRegister_##name() { \
            g_tests[g_test_count_registered++] = { #name, test_##name }; \
        } \
    } test_register_##name; \
    static void test_##name()

#define ASSERT_TRUE(cond, msg) \
    do { \
        test_count++; \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n  condition: %s\n", \
                    msg, __FILE__, __LINE__, #cond); \
            fail_count++; \
        } \
    } while (0)

#define ASSERT_BITEQ_F32(expected, actual, msg) \
    do { \
        test_count++; \
        uint32_t e_bits, a_bits; \
        memcpy(&e_bits, &(expected), sizeof(uint32_t)); \
        memcpy(&a_bits, &(actual), sizeof(uint32_t)); \
        if (e_bits != a_bits) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n" \
                    "  expected: %.9g (0x%08x)\n" \
                    "  actual:   %.9g (0x%08x)\n", \
                    msg, __FILE__, __LINE__, \
                    (double)(expected), e_bits, \
                    (double)(actual),   a_bits); \
            fail_count++; \
        } \
    } while (0)

#define ASSERT_NEAR_F32(expected, actual, tol, msg) \
    do { \
        test_count++; \
        float diff = fabsf((expected) - (actual)); \
        if (!(diff <= (tol))) { /* using <= to fail on NaN */ \
            fprintf(stderr, "FAIL: %s (%s:%d)\n" \
                    "  expected: %.9g\n" \
                    "  actual:   %.9g\n" \
                    "  diff:     %.9g (tol %.9g)\n", \
                    msg, __FILE__, __LINE__, \
                    (double)(expected), (double)(actual), \
                    (double)diff, (double)(tol)); \
            fail_count++; \
        } \
    } while (0)

// Test sizes covering typical attention head dims plus odd tail sizes
// to exercise the scalar leftover loop.
static const std::vector<int> kTestSizes = {
    1, 8, 15, 16, 17, 32, 33, 64, 80, 96, 128, 129, 192, 256, 512, 1024
};

// Fill a vector with random f16 values in N(0, 0.1) — typical layer
// activation range after LayerNorm. Uses a fixed seed for determinism.
static void fill_random_f16(std::vector<ggml_fp16_t> &out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto &v : out) {
        v = ggml_fp32_to_fp16(dist(rng));
    }
}

// Tolerance for vec_dot_f16: both implementations convert to f32 and
// sum, but in different chunk sizes (AVX2: 8-wide; AVX-512: 16-wide).
// The reduction order differs, so we allow a relative error
// proportional to n (Wilkinson-style summed-product bound: O(n) terms
// each at unit-roundoff eps/2, with a small safety factor). max_abs
// here is the peak |x[i]*y[i]|, an upper bound for the running partial
// sum scale.
static float vec_dot_tol(int n, const std::vector<ggml_fp16_t> &x,
                          const std::vector<ggml_fp16_t> &y) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float xv = ggml_fp16_to_fp32(x[i]);
        float yv = ggml_fp16_to_fp32(y[i]);
        max_abs = std::max(max_abs, std::fabs(xv) * std::fabs(yv));
    }
    return std::max(1e-5f, 8.0f * (float)n * FLT_EPSILON * max_abs);
}

// ============================================================================
// fp16_to_fp32_row: per-element conversion. Both ggml's helper and ours
// use _mm*_cvtph_ps, which is IEEE-754 deterministic per element. We
// expect BIT-IDENTICAL output regardless of SIMD width.
// ============================================================================

TEST(fp16_to_fp32_row_normal_inputs) {
    for (int n : kTestSizes) {
        std::vector<ggml_fp16_t> input(n);
        std::vector<float> ref_out(n), our_out(n);
        fill_random_f16(input, /*seed=*/0xDEADBEEFCAFEull ^ (uint64_t)n);

        ggml_fp16_to_fp32_row(input.data(), ref_out.data(), n);
        bool handled = llamafile_fa_fp16_to_fp32_row(input.data(), our_out.data(), n);

        if (!handled) {
            skip_count++;
            continue;
        }

        char msg[128];
        for (int i = 0; i < n; ++i) {
            snprintf(msg, sizeof(msg),
                     "fp16_to_fp32_row n=%d i=%d (normal inputs)", n, i);
            ASSERT_BITEQ_F32(ref_out[i], our_out[i], msg);
        }
    }
}

TEST(fp16_to_fp32_row_edge_values) {
    // Include ±0, ±INF, denormals, NaN, and the f16 min/max.
    std::vector<ggml_fp16_t> input = {
        ggml_fp32_to_fp16( 0.0f),
        ggml_fp32_to_fp16(-0.0f),
        ggml_fp32_to_fp16( INFINITY),
        ggml_fp32_to_fp16(-INFINITY),
        ggml_fp32_to_fp16(NAN),
        ggml_fp32_to_fp16( 65504.0f),   // f16 max finite
        ggml_fp32_to_fp16(-65504.0f),   // f16 min finite
        ggml_fp32_to_fp16( 6.10e-5f),   // f16 smallest normal
        ggml_fp32_to_fp16( 5.96e-8f),   // f16 smallest denormal
        ggml_fp32_to_fp16( 1.0f),
        ggml_fp32_to_fp16(-1.0f),
        ggml_fp32_to_fp16( 0.5f),
        ggml_fp32_to_fp16( 0.125f),
        ggml_fp32_to_fp16( 0.0625f),
        ggml_fp32_to_fp16( 0.00390625f),
        ggml_fp32_to_fp16( 65000.0f),
    };
    int n = (int)input.size();
    std::vector<float> ref_out(n), our_out(n);

    ggml_fp16_to_fp32_row(input.data(), ref_out.data(), n);
    bool handled = llamafile_fa_fp16_to_fp32_row(input.data(), our_out.data(), n);

    if (!handled) {
        skip_count++;
        return;
    }

    char msg[128];
    for (int i = 0; i < n; ++i) {
        // For NaN, only assert both are NaN (any NaN representation is OK).
        if (std::isnan(ref_out[i]) || std::isnan(our_out[i])) {
            test_count++;
            if (!(std::isnan(ref_out[i]) && std::isnan(our_out[i]))) {
                fprintf(stderr,
                        "FAIL: fp16_to_fp32_row edge values i=%d: "
                        "NaN mismatch (ref nan=%d, our nan=%d)\n",
                        i, std::isnan(ref_out[i]), std::isnan(our_out[i]));
                fail_count++;
            }
            continue;
        }
        snprintf(msg, sizeof(msg),
                 "fp16_to_fp32_row edge values i=%d", i);
        ASSERT_BITEQ_F32(ref_out[i], our_out[i], msg);
    }
}

// ============================================================================
// vec_dot_f16: reduction, allow ULP tolerance because chunk widths
// differ between ref (AVX2 8-wide) and ours (AVX-512 16-wide).
// ============================================================================

TEST(vec_dot_f16_normal_inputs) {
    for (int n : kTestSizes) {
        std::vector<ggml_fp16_t> x(n), y(n);
        fill_random_f16(x, /*seed=*/0xA5A5A5A5A5A5A5A5ull ^ (uint64_t)n);
        fill_random_f16(y, /*seed=*/0x5A5A5A5A5A5A5A5Aull ^ (uint64_t)n);

        float ref = 0.0f, our = 0.0f;
        ggml_vec_dot_f16(n, &ref, /*bs=*/0, x.data(), /*bx=*/0,
                          y.data(), /*by=*/0, /*nrc=*/1);
        bool handled = llamafile_fa_vec_dot_f16(n, &our, x.data(), y.data());

        if (!handled) {
            skip_count++;
            continue;
        }

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "vec_dot_f16 n=%d (normal inputs)", n);
        ASSERT_NEAR_F32(ref, our, vec_dot_tol(n, x, y), msg);
    }
}

TEST(vec_dot_f16_adversarial_reduction_order) {
    // To actually stress reduction-order sensitivity, the per-element
    // products x[i]*y[i] must alternate sign — otherwise SIMD width
    // doesn't change the result. Use a scale of 10 so |product| = 100
    // (well within f16 range): partial sums oscillate by ±100 across
    // steps, and AVX2 (8-wide) vs AVX-512 (16-wide) reduction trees
    // pair the cancelling terms in different orders, amplifying any
    // floating-point drift.
    //   x[i] = (i & 1) ? -10 : +10                 → [+, -, +, -, ...]
    //   y[i] = ((i / 2) & 1) ? -10 : +10           → [+, +, -, -, ...]
    //   products = x[i]*y[i] alternate as          [+, -, -, +, +, -, -, +, ...]
    for (int n : kTestSizes) {
        if (n < 2) continue;
        std::vector<ggml_fp16_t> x(n), y(n);
        const float scale = 10.0f;
        for (int i = 0; i < n; ++i) {
            x[i] = ggml_fp32_to_fp16((i & 1)        ? -scale : scale);
            y[i] = ggml_fp32_to_fp16(((i / 2) & 1) ? -scale : scale);
        }

        float ref = 0.0f, our = 0.0f;
        ggml_vec_dot_f16(n, &ref, 0, x.data(), 0, y.data(), 0, 1);
        bool handled = llamafile_fa_vec_dot_f16(n, &our, x.data(), y.data());

        if (!handled) {
            skip_count++;
            continue;
        }

        // Worst-case drift bound: ~n * eps * max_partial_sum_scale.
        // max |x[i]*y[i]| = scale^2 = 100. Use 4 * n * eps * 100 with
        // a small absolute floor for tiny n.
        float tol = std::max(1e-4f, 4.0f * (float)n * FLT_EPSILON * scale * scale);
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "vec_dot_f16 n=%d (adversarial alternating-sign products)", n);
        ASSERT_NEAR_F32(ref, our, tol, msg);
    }
}

TEST(vec_dot_f16_zeros) {
    // All-zero inputs: result must be exactly 0.0f, bit-identical.
    for (int n : kTestSizes) {
        std::vector<ggml_fp16_t> x(n, ggml_fp32_to_fp16(0.0f));
        std::vector<ggml_fp16_t> y(n, ggml_fp32_to_fp16(0.0f));

        float ref = 999.0f, our = 999.0f;
        ggml_vec_dot_f16(n, &ref, 0, x.data(), 0, y.data(), 0, 1);
        bool handled = llamafile_fa_vec_dot_f16(n, &our, x.data(), y.data());

        if (!handled) {
            skip_count++;
            continue;
        }

        float zero = 0.0f;
        char msg[128];
        snprintf(msg, sizeof(msg), "vec_dot_f16 n=%d (zeros) ref", n);
        ASSERT_BITEQ_F32(zero, ref, msg);
        snprintf(msg, sizeof(msg), "vec_dot_f16 n=%d (zeros) ours", n);
        ASSERT_BITEQ_F32(zero, our, msg);
    }
}

TEST(dispatch_wiring) {
    // Sanity check: regardless of whether the AVX-512F variant is
    // available, the wrapper must return without crashing for trivial
    // input. If our dispatch table got mis-wired (e.g., function pointer
    // to wrong arch), this would catch it.
    std::vector<ggml_fp16_t> tiny = {
        ggml_fp32_to_fp16(0.5f), ggml_fp32_to_fp16(-0.25f),
        ggml_fp32_to_fp16(1.0f), ggml_fp32_to_fp16( 0.125f),
    };
    float dot = -1.0f;
    bool dot_handled = llamafile_fa_vec_dot_f16((int)tiny.size(), &dot,
                                                  tiny.data(), tiny.data());
    ASSERT_TRUE(dot_handled || dot == -1.0f,
                "vec_dot_f16: if not handled, output must be untouched");

    std::vector<float> out(tiny.size(), -999.0f);
    bool conv_handled = llamafile_fa_fp16_to_fp32_row(tiny.data(), out.data(),
                                                       (int64_t)tiny.size());
    ASSERT_TRUE(conv_handled || out[0] == -999.0f,
                "fp16_to_fp32_row: if not handled, output must be untouched");
}

// ============================================================================
// simd_gemm: C[M x N] += A[M x K] * B[K x N], all f32. We compare our
// AVX-512 wrapper against a scalar reference because the upstream
// ggml-cpu simd_gemm is `static inline` in a header and isn't directly
// linkable here. The scalar reference is the exact math the wrapper
// must reproduce; both implementations are well-defined up to FMA
// associativity (FMA rounding doesn't commute across reduction orders).
// Tolerance: 4 * K * eps * max(|A|*|B|) which matches the worst-case
// summed-product bound.
// ============================================================================

static void scalar_gemm_reference(float *C, const float *A, const float *B,
                                   int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = C[i * N + j];
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

static float gemm_tol(int K, float scale) {
    return std::max(1e-4f, 8.0f * (float)K * FLT_EPSILON * scale * scale);
}

static void fill_random_f32(std::vector<float> &v, uint64_t seed,
                             float stddev = 0.1f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (auto &x : v) x = dist(rng);
}

TEST(simd_gemm_typical_fa_tile_shapes) {
    // Shapes that match the FA tile dispatch in ops.cpp's tiled function
    // (Q_TILE_SZ x KV_TILE_SZ for the KQ matmul, Q_TILE_SZ x DV for the
    // VKQ accumulation). Common head dims: 64, 80, 96, 128. Common tile
    // sizes from ggml_fa_tile_config: 4-16 Q rows, 32-128 KV columns.
    struct Shape { int M, K, N; const char *label; };
    Shape shapes[] = {
        {  4, 128,  64, "KQ: Q4 x DK128 x KV64" },
        {  8, 128,  64, "KQ: Q8 x DK128 x KV64" },
        { 16, 128, 128, "KQ: Q16 x DK128 x KV128" },
        {  4,  64, 128, "VKQ: Q4 x KV64 x DV128" },
        {  8,  64, 128, "VKQ: Q8 x KV64 x DV128" },
        { 16, 128, 128, "VKQ: Q16 x KV128 x DV128" },
        {  4,  96,  80, "odd: Q4 x 96 x 80" },
        { 16,  64,  32, "narrow: Q16 x 64 x 32" },
    };
    for (auto &s : shapes) {
        std::vector<float> A(s.M * s.K), B(s.K * s.N);
        std::vector<float> C_ref(s.M * s.N), C_our(s.M * s.N);
        uint64_t seed = 0xC0FFEEull ^ (uint64_t)(s.M * 31 + s.K * 17 + s.N);
        fill_random_f32(A, seed);
        fill_random_f32(B, seed ^ 0xDEADBEEFull);
        // Pre-fill C with random values to test the += accumulation.
        fill_random_f32(C_ref, seed ^ 0xFACEFEEDull);
        C_our = C_ref;

        scalar_gemm_reference(C_ref.data(), A.data(), B.data(),
                              s.M, s.K, s.N);
        bool handled = llamafile_fa_simd_gemm(C_our.data(), A.data(),
                                               B.data(), s.M, s.K, s.N);

        if (!handled) {
            skip_count++;
            continue;
        }

        char msg[160];
        float tol = gemm_tol(s.K, 0.5f);  // stddev 0.1 → max |a|*|b| ~ 0.5
        for (int i = 0; i < s.M; ++i) {
            for (int j = 0; j < s.N; ++j) {
                snprintf(msg, sizeof(msg),
                         "simd_gemm %s i=%d j=%d", s.label, i, j);
                ASSERT_NEAR_F32(C_ref[i * s.N + j], C_our[i * s.N + j], tol, msg);
            }
        }
    }
}

TEST(simd_gemm_zero_accumulator) {
    // Start from C=0 so the result is purely A*B (no += round-off).
    // Verifies the kernel handles a zero starting accumulator correctly.
    int M = 8, K = 128, N = 128;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_ref(M * N, 0.0f), C_our(M * N, 0.0f);
    fill_random_f32(A, 0x1234);
    fill_random_f32(B, 0x5678);

    scalar_gemm_reference(C_ref.data(), A.data(), B.data(), M, K, N);
    bool handled = llamafile_fa_simd_gemm(C_our.data(), A.data(), B.data(),
                                           M, K, N);
    if (!handled) {
        skip_count++;
        return;
    }
    char msg[160];
    float tol = gemm_tol(K, 0.5f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            snprintf(msg, sizeof(msg),
                     "simd_gemm zero-acc i=%d j=%d", i, j);
            ASSERT_NEAR_F32(C_ref[i * N + j], C_our[i * N + j], tol, msg);
        }
    }
}

TEST(simd_gemm_dispatch_wiring) {
    int M = 4, K = 16, N = 32;
    std::vector<float> A(M * K, 1.0f), B(K * N, 1.0f);
    std::vector<float> C(M * N, -777.0f);
    bool handled = llamafile_fa_simd_gemm(C.data(), A.data(), B.data(),
                                           M, K, N);
    ASSERT_TRUE(handled || C[0] == -777.0f,
                "simd_gemm: if not handled, output must be untouched");
    if (handled) {
        // Each C[i,j] = -777 + sum_k(1 * 1) = -777 + K = -761
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                test_count++;
                float expected = -777.0f + (float)K;
                if (fabsf(C[i * N + j] - expected) > 1e-5f) {
                    fprintf(stderr, "FAIL: simd_gemm dispatch i=%d j=%d "
                            "expected %.3f got %.3f\n",
                            i, j, expected, C[i * N + j]);
                    fail_count++;
                }
            }
        }
    }
}

// ggml's f16↔f32 lookup table (used by ggml_vec_dot_f16's scalar tail
// in simd-mappings.h via GGML_CPU_FP16_TO_FP32). Populated by ggml's
// graph compute first-call init; we replicate the population here so
// the tests don't depend on running a graph.
extern "C" float ggml_table_f32_f16[1 << 16];

static void init_ggml_tables() {
    for (int i = 0; i < (1 << 16); ++i) {
        ggml_fp16_t f16 = (ggml_fp16_t)(uint16_t)i;
        ggml_table_f32_f16[i] = ggml_fp16_to_fp32(f16);
    }
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    // Must happen before any test runs ggml_vec_dot_f16.
    init_ggml_tables();

    fprintf(stderr, "Running fa_helpers tests (issue #975) — %d tests registered\n",
            g_test_count_registered);

    for (int i = 0; i < g_test_count_registered; ++i) {
        fprintf(stderr, "  [%d/%d] %s\n", i + 1, g_test_count_registered,
                g_tests[i].name);
        g_tests[i].fn();
    }

    if (fail_count > 0) {
        fprintf(stderr, "\n%d/%d assertions FAILED (%d skipped — CPU "
                "doesn't have the optimized variant)\n",
                fail_count, test_count, skip_count);
        return 1;
    }

    fprintf(stderr, "All %d assertions PASSED (%d skipped — CPU doesn't "
            "have the optimized variant)\n",
            test_count, skip_count);
    return 0;
}
