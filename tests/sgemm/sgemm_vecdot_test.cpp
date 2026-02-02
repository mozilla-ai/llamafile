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
// ============================================================================
// sgemm_vecdot_test: Vector Dot Product Benchmark (n=1 edge case)
// ============================================================================
//
// PURPOSE:
//   This test benchmarks the special case where n=1, which reduces matrix
//   multiplication to a series of independent vector dot products. This case
//   is common in autoregressive LLM inference where we process one token at
//   a time.
//
// OPERATION:
//   Computes C = A^T * B where B is a single column vector (n=1):
//     - A is a k x m matrix
//     - B is a k x 1 vector
//     - C is the resulting m x 1 vector
//   Each C[i] = dot(A[:,i], B) - a simple dot product of two k-element vectors.
//
// WHY THIS CASE MATTERS:
//   - In autoregressive generation, each new token requires multiplying the
//     weight matrices by a single input vector
//   - This is memory-bandwidth bound rather than compute bound
//   - Performance depends on how efficiently we can stream data from memory
//   - Both implementations tend to perform similarly here since memory is
//     the bottleneck, not arithmetic
//
// IMPLEMENTATIONS COMPARED:
//
//   1. ggmlBLAS::sgemm (baseline/fallback)
//      - Uses llama.cpp's ggml_vec_dot_f32() function internally
//      - This is the EXACT code path llama.cpp takes when llamafile_sgemm()
//        returns false
//      - For n=1, this is essentially just m independent dot products
//
//   2. llamafile_sgemm_openmp (optimized)
//      - llamafile's optimized tinyblas CPU kernels
//      - For n=1, may fall back to similar dot-product approach
//      - Can be disabled via LLAMAFILE_DISABLE_SGEMM=1 environment variable
//
// METRICS:
//
//   Performance (microseconds):
//     - Time to complete all m dot products
//     - Lower is better
//     - For n=1, expect similar performance between implementations
//       (both are memory-bandwidth limited)
//
//   Accuracy (ULP = Units in Last Place):
//     - Measures floating-point precision difference between implementations
//     - "ulp average": Mean ULP difference across all m output elements
//     - "ulp worst": Maximum ULP difference for any single element
//
// TEST PARAMETERS:
//   - m=1024, n=1, k=260000 (simulates single-token inference)
//   - Large k simulates LLM hidden dimensions
//

#include "sgemm_test_utils.h"
#include "sgemm.h"
#include "ggml.h"
#include "ggml-cpu-impl.h"
#include <cassert>
#include <cmath>
#include <thread>

#define ITERATIONS 30
#define ALLOC(n) (float *)aligned_alloc(4096, sizeof(float) * (n))

static int get_num_threads() {
    return std::thread::hardware_concurrency();
}

// Returns true if sgemm was able to compute the result
bool llamafile_sgemm_openmp(long m, long n, long k, const void *A, long lda, const void *B,
                            long ldb, void *C, long ldc, int Atype, int Btype, int Ctype) {
    int nth = get_num_threads();
    bool all_ok = true;
#pragma omp parallel for reduction(&& : all_ok)
    for (int ith = 0; ith < nth; ++ith) {
        ggml_compute_params params = {/*.ith=*/ith, /*.nth=*/nth, 0, nullptr, nullptr};
        bool res = llamafile_sgemm(&params, m, n, k, A, lda, B, ldb, C, ldc, Atype, Btype, Ctype);
        all_ok = all_ok && res;
    }
    return all_ok;
}

int test(void) {
    int m = 1024;
    int n = 1;  // Vector dot product case
    int k = 260000;
    int lda = ROUNDUP(k, 16);
    int ldb = ROUNDUP(k, 16);
    int ldc = ROUNDUP(m, 16);
    float *A = ALLOC(lda * m);
    float *B = ALLOC(ldb * n);
    float *C = ALLOC(ldc * n);
    float *G = ALLOC(ldc * n);
    broadcast(A, lda * m, NAN);
    broadcast(B, ldb * n, NAN);
    broadcast(C, ldc * n, NAN);
    broadcast(G, ldc * n, NAN);
    randomize(k, m, A, lda);
    randomize(k, n, B, ldb);

    int nth = get_num_threads();
    printf("Using %d threads\n", nth);
    printf("Matrix dimensions: m=%d n=%d k=%d (vector dot product)\n", m, n, k);

    printf("\n--- Benchmarks ---\n");

    // ggmlBLAS uses llama.cpp's actual ggml_vec_dot_f32 function - this is what
    // the production fallback uses when llamafile_sgemm returns false
    BENCH(ggmlBLAS::sgemm(m, n, k, A, lda, B, ldb, G, ldc, nth), ITERATIONS);

    // Check if our optimized kernel is available
    bool sgemm_available = llamafile_sgemm_openmp(m, n, k, A, lda, B, ldb, C, ldc,
                                                   GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
    if (sgemm_available) {
        BENCH(llamafile_sgemm_openmp(m, n, k, A, lda, B, ldb, C, ldc, GGML_TYPE_F32, GGML_TYPE_F32,
                                     GGML_TYPE_F32), ITERATIONS);
    } else {
        printf("%12s %s\n", "N/A", "llamafile_sgemm_openmp (disabled or unsupported)");
    }

    // Accuracy comparison: our optimized kernel vs reference
    if (sgemm_available) {
        printf("\n--- Accuracy (optimized vs fallback) ---\n");
        double err_sum = 0;
        long long err_worst = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float g = G[ldc * j + i];
                float c = C[ldc * j + i];
                if (flt::isnan(g)) {
                    fprintf(stderr, "%s:%d: found nan in reference matrix: i=%d j=%d\n", __FILE__,
                            __LINE__, i, j);
                    return 3;
                }
                if (flt::isnan(c)) {
                    fprintf(stderr, "%s:%d: found nan in output matrix: i=%d j=%d\n", __FILE__,
                            __LINE__, i, j);
                    return 4;
                }
                long long gi = flt::toint(g);
                long long ci = flt::toint(c);
                long long err = gi - ci;
                if (err < 0)
                    err = -err;
                err_sum += err;
                if (err > err_worst)
                    err_worst = err;
            }
        }

        double err_avg = err_sum / (m * n);
        fprintf(stderr, "%12g ulp average\n", err_avg);
        fprintf(stderr, "%12lld ulp worst\n", err_worst);
    } else {
        printf("\n--- Accuracy ---\n");
        printf("(skipped - optimized kernel not available, would use fallback)\n");
    }

    free(G);
    free(C);
    free(B);
    free(A);

    return 0;
}

void print_test_info() {
    printf("============================================================================\n");
    printf("sgemm_vecdot_test: Vector Dot Product Benchmark (n=1 edge case)\n");
    printf("============================================================================\n");
    printf("\n");
    printf("OPERATION:\n");
    printf("  C = A^T * B  where B is a single column (n=1)\n");
    printf("  Result: m independent dot products of k-element vectors.\n");
    printf("  Simulates single-token autoregressive LLM inference.\n");
    printf("\n");
    printf("IMPLEMENTATIONS:\n");
    printf("  ggmlBLAS::sgemm        - llama.cpp's fallback using ggml_vec_dot_f32()\n");
    printf("  llamafile_sgemm_openmp - Optimized tinyblas CPU kernels (if available)\n");
    printf("\n");
    printf("NOTE: For n=1, both implementations are memory-bandwidth bound.\n");
    printf("      Expect similar performance regardless of kernel optimizations.\n");
    printf("\n");
    printf("METRICS:\n");
    printf("  Time (us)   - Microseconds to complete. Lower is better.\n");
    printf("  ulp average - Mean precision difference (Units in Last Place).\n");
    printf("  ulp worst   - Maximum precision difference for any element.\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rc;

    print_test_info();

    const char *kernel = llamafile_sgemm_name();
    printf("Selected kernel: %s\n", kernel);
    if (strcmp(kernel, "unsupported") == 0) {
        printf("  -> Optimized kernel unavailable. ggmlBLAS shows production fallback performance.\n");
    } else {
        printf("  -> Comparing optimized kernel against ggmlBLAS fallback.\n");
    }
    printf("\n");

    printf("=== Run 1 ===\n");
    if ((rc = test()))
        return rc;

    printf("=== Run 2 ===\n");
    if ((rc = test()))
        return rc;

    printf("=== Run 3 ===\n");
    if ((rc = test()))
        return rc;

    return 0;
}
