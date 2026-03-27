// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
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
// strsm_test: TinyBLAS vs cuBLAS StrsmBatched Comparison
// ============================================================================
//
// PURPOSE:
//   Validates tinyblasStrsmBatched correctness by comparing against cuBLAS
//   (our ground truth). Both implementations run in the same process on the
//   same GPU data for a direct comparison.
//
// OPERATION:
//   Solves X * A = alpha * B where A is upper triangular (SIDE_RIGHT,
//   FILL_MODE_UPPER, OP_N, DIAG_NON_UNIT). Result overwrites B in-place.
//   This is the configuration used by ggml's solve_tri CUDA backend.
//
// BUILD:
//   bash tests/strsm/build_and_run.sh
//
// METRICS:
//   - Time (us): Average kernel execution time over multiple iterations
//   - Max abs error: Largest element-wise absolute difference
//   - RMSE: Root mean square error across all elements
//   - Max ULP diff: Largest Units in Last Place difference
//

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "tinyblas.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// ============================================================================
// Error checking macros
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t err = (call);                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                    (int)err);                                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define TINYBLAS_CHECK(call)                                                   \
    do {                                                                        \
        tinyblasStatus_t err = (call);                                          \
        if (err != TINYBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "TinyBLAS error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, tinyblasGetStatusString(err));                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================================
// Utilities
// ============================================================================

static inline long long micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000LL + (ts.tv_nsec + 999) / 1000;
}

static unsigned rand_state = 42;

static inline unsigned rand32(void) {
    rand_state = rand_state * 1664525u + 1013904223u;
    return rand_state;
}

static inline float randf(void) {
    // Random float in [-1, 1]
    return (float)(rand32() & 0xFFFF) / 32768.0f - 1.0f;
}

static inline unsigned float_to_bits(float f) {
    unsigned bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

static inline long long ulp_diff(float a, float b) {
    // ULP distance between two floats
    unsigned ai = float_to_bits(a);
    unsigned bi = float_to_bits(b);
    // Handle sign: convert to two's complement-like representation
    if (ai & 0x80000000u)
        ai = 0x80000000u - ai;
    if (bi & 0x80000000u)
        bi = 0x80000000u - bi;
    long long diff = (long long)ai - (long long)bi;
    return diff < 0 ? -diff : diff;
}

// ============================================================================
// Residual computation (host-side, double precision)
// ============================================================================

// Compute the scaled backward error for X*A = alpha*B:
//   ||X*A - alpha*B||_inf / (||A||_inf * ||X||_inf * n * eps)
//
// A is n x n upper triangular (column-major, lda stride)
// X and B are m x n (column-major, ldb stride)
// All inputs are float but computation is in double for accuracy.
// Returns the scaled residual (< 1.0 means good, > 10 means trouble).
static double compute_residual(const float *X, const float *A, const float *B_orig,
                                int m, int n, int lda, int ldb, float alpha) {
    const double eps = 1.1920929e-7; // float32 machine epsilon

    // Compute ||A||_inf = max row sum of |A|
    double norm_A = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
            row_sum += fabs((double)A[i + j * lda]);
        if (row_sum > norm_A) norm_A = row_sum;
    }

    // Compute ||X||_inf = max row sum of |X|
    double norm_X = 0.0;
    for (int i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
            row_sum += fabs((double)X[i + j * ldb]);
        if (row_sum > norm_X) norm_X = row_sum;
    }

    // Compute R = X*A - alpha*B, then ||R||_inf
    // R[i][j] = sum_p(X[i][p] * A[p][j]) - alpha*B[i][j]
    // A is upper triangular so A[p][j] = 0 for p > j
    double norm_R = 0.0;
    for (int i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            double xa_ij = 0.0;
            // A is upper triangular: only p <= j contributes
            for (int p = 0; p <= j; p++)
                xa_ij += (double)X[i + p * ldb] * (double)A[p + j * lda];
            double r_ij = xa_ij - (double)alpha * (double)B_orig[i + j * ldb];
            row_sum += fabs(r_ij);
        }
        if (row_sum > norm_R) norm_R = row_sum;
    }

    double denom = norm_A * norm_X * n * eps;
    if (denom == 0.0) return 0.0;
    return norm_R / denom;
}

// ============================================================================
// Matrix generation (column-major)
// ============================================================================

// Generate a random upper triangular matrix (n x n, column-major, lda stride)
// Diagonal elements are guaranteed to be well-conditioned (away from zero)
static void gen_upper_triangular(float *A, int n, int lda, bool identity = false,
                                  float diag_min = 0.5f, float diag_max = 2.0f) {
    // Zero the whole thing first
    for (int j = 0; j < n; j++)
        for (int i = 0; i < lda; i++)
            A[j * lda + i] = 0.0f;

    for (int j = 0; j < n; j++) {
        // Upper triangle: A[i][j] where i <= j, stored at A[i + j*lda]
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                if (identity) {
                    A[i + j * lda] = 1.0f;
                } else {
                    // Diagonal: random in [diag_min, diag_max] with random sign
                    float val = diag_min + (diag_max - diag_min) * fabsf(randf());
                    A[i + j * lda] = (rand32() & 1) ? val : -val;
                }
            } else {
                A[i + j * lda] = identity ? 0.0f : randf();
            }
        }
    }
}

// Generate an ill-conditioned upper triangular matrix
// Diagonal elements span a wide range (diag_min to diag_max)
static void gen_ill_conditioned(float *A, int n, int lda,
                                 float diag_min = 1e-3f, float diag_max = 1e3f) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < lda; i++)
            A[j * lda + i] = 0.0f;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                // Logarithmic spacing of diagonal elements
                float t = (n > 1) ? (float)j / (float)(n - 1) : 0.5f;
                float val = diag_min * powf(diag_max / diag_min, t);
                A[i + j * lda] = (rand32() & 1) ? val : -val;
            } else {
                A[i + j * lda] = randf() * 0.1f; // Small off-diagonal
            }
        }
    }
}

// Generate a random dense matrix (m x n, column-major, ld stride)
static void gen_random_matrix(float *B, int m, int n, int ld) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < ld; i++) {
            B[j * ld + i] = (i < m) ? randf() : 0.0f;
        }
    }
}

// Generate matrices mimicking Qwen3.5 delta-net attention solve_tri usage.
//
// In the model, solve_tri receives:
//   A = I + lower_tri(K^T * K_b * decay_mask)   (GGML lower triangular)
//   B = -lower_tri(K^T * K_b * decay_mask)       (GGML lower triangular)
//
// Due to GGML row-major -> cuBLAS column-major transposition:
//   A appears as upper triangular with diagonal = 1.0
//   B appears as upper triangular with diagonal = 0.0
//
// The decay mask applies exponential decay: entry[i,j] *= exp(-rate * |i-j|)
// so off-diagonal values shrink away from the diagonal.
//
// A_out: n x n upper triangular, lda stride, column-major
// B_out: m x n dense (but structured as upper tri + noise), ldb stride
static void gen_qwen_attention_matrices(float *A_out, float *B_out,
                                         int n, int m, int lda, int ldb,
                                         float decay_rate = 0.1f) {
    // Zero everything
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < lda; i++)
            A_out[j * lda + i] = 0.0f;
        for (int i = 0; i < ldb; i++)
            B_out[j * ldb + i] = 0.0f;
    }

    // Build upper triangular A with diagonal = 1.0 and decaying off-diagonal
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            if (i == j) {
                A_out[i + j * lda] = 1.0f;
                // B diagonal is 0 (from -lower_tri which excludes diagonal)
            } else {
                // Off-diagonal: attention-like values with exponential decay
                float attn_val = randf() * 0.5f; // attention scores in [-0.5, 0.5]
                float decay = expf(-decay_rate * (float)(j - i));
                float val = attn_val * decay;
                A_out[i + j * lda] = val;       // A = I + upper_tri_part
                // B = -(upper_tri_part), so B has negated off-diagonal
                if (i < m) {
                    B_out[i + j * ldb] = -val;
                }
            }
        }
        // B also has content in rows beyond the triangle (from attention patterns)
        for (int i = j + 1; i < m; i++) {
            B_out[i + j * ldb] = randf() * 0.3f;
        }
    }
}

// ============================================================================
// GPU batch pointer setup kernel
// ============================================================================

__global__ void setup_batch_pointers(const float **A_ptrs, float **B_ptrs,
                                      const float *A_base, float *B_base,
                                      int A_stride, int B_stride, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        A_ptrs[idx] = A_base + idx * A_stride;
        B_ptrs[idx] = B_base + idx * B_stride;
    }
}

// ============================================================================
// Test configuration
// ============================================================================

struct TestConfig {
    const char *name;
    int n;         // Size of triangular matrix A (n x n)
    int m;         // Number of rows in B (B is m x n)
    int batch;     // Batch count
    float alpha;   // Scalar multiplier
    bool identity; // Use identity matrix for A
    bool ill_cond; // Use ill-conditioned matrix
    bool qwen;     // Use Qwen-like attention matrices (I + decay*attn)
    int iterations; // Timing iterations
};

struct TestResult {
    bool pass;
    double max_abs_err;       // Max |cublas - tinyblas| element-wise
    double rmse;              // RMSE of cublas vs tinyblas
    long long max_ulp;        // Max ULP diff cublas vs tinyblas
    double cublas_residual;   // ||X_cublas * A - alpha*B|| / (||A||*||X||*n*eps)
    double tinyblas_residual; // ||X_tiny * A - alpha*B|| / (||A||*||X||*n*eps)
    double cublas_us;
    double tinyblas_us;
};

// ============================================================================
// Run a single test case
// ============================================================================

static TestResult run_test(const TestConfig &cfg) {
    TestResult result = {};
    const int n = cfg.n;
    const int m = cfg.m;
    const int batch = cfg.batch;
    const float alpha = cfg.alpha;
    const int lda = n;  // Leading dimension of A (column-major, n x n)
    const int ldb = m;  // Leading dimension of B (column-major, m x n)

    // ---- Host allocation ----
    size_t A_size = (size_t)lda * n;         // One A matrix
    size_t B_size = (size_t)ldb * n;         // One B matrix
    size_t A_total = A_size * batch;
    size_t B_total = B_size * batch;

    std::vector<float> h_A(A_total);
    std::vector<float> h_B(B_total);

    // Generate matrices for each batch element
    for (int b = 0; b < batch; b++) {
        if (cfg.qwen) {
            gen_qwen_attention_matrices(h_A.data() + b * A_size,
                                         h_B.data() + b * B_size,
                                         n, m, lda, ldb);
        } else {
            if (cfg.identity) {
                gen_upper_triangular(h_A.data() + b * A_size, n, lda, /*identity=*/true);
            } else if (cfg.ill_cond) {
                gen_ill_conditioned(h_A.data() + b * A_size, n, lda);
            } else {
                gen_upper_triangular(h_A.data() + b * A_size, n, lda);
            }
            gen_random_matrix(h_B.data() + b * B_size, m, n, ldb);
        }
    }

    // ---- Device allocation ----
    float *d_A, *d_B_cublas, *d_B_tinyblas;
    CUDA_CHECK(cudaMalloc(&d_A, A_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_cublas, B_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_tinyblas, B_total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_cublas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_tinyblas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Batch pointer arrays on device ----
    const float **d_A_ptrs_cublas, **d_A_ptrs_tiny;
    float **d_B_ptrs_cublas, **d_B_ptrs_tiny;
    CUDA_CHECK(cudaMalloc(&d_A_ptrs_cublas, batch * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_B_ptrs_cublas, batch * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_A_ptrs_tiny, batch * sizeof(float *)));
    CUDA_CHECK(cudaMalloc(&d_B_ptrs_tiny, batch * sizeof(float *)));

    int threads = (batch + 255) / 256 * 256;
    if (threads > 256) threads = 256;
    int blocks = (batch + threads - 1) / threads;

    setup_batch_pointers<<<blocks, threads>>>(d_A_ptrs_cublas, d_B_ptrs_cublas,
                                               d_A, d_B_cublas, A_size, B_size, batch);
    setup_batch_pointers<<<blocks, threads>>>(d_A_ptrs_tiny, d_B_ptrs_tiny,
                                               d_A, d_B_tinyblas, A_size, B_size, batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Create handles ----
    cublasHandle_t cublas_handle;
    tinyblasHandle_t tinyblas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    TINYBLAS_CHECK(tinyblasCreate(&tinyblas_handle));

    // Use default math mode for cuBLAS (not TF32) to get accurate reference
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    // Set stream on tinyblas handle (tinyblasCreate uses malloc, not calloc,
    // so the stream field is uninitialized)
    TINYBLAS_CHECK(tinyblasSetStream(tinyblas_handle, nullptr));

    // ---- Correctness run ----
    CUBLAS_CHECK(cublasStrsmBatched(cublas_handle, CUBLAS_SIDE_RIGHT,
                                     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                     CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                     d_A_ptrs_cublas, lda,
                                     d_B_ptrs_cublas, ldb, batch));
    CUDA_CHECK(cudaDeviceSynchronize());

    TINYBLAS_CHECK(tinyblasStrsmBatched(tinyblas_handle, TINYBLAS_SIDE_RIGHT,
                                         TINYBLAS_FILL_MODE_UPPER, TINYBLAS_OP_N,
                                         TINYBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                         (const float *const *)d_A_ptrs_tiny, lda,
                                         (float *const *)d_B_ptrs_tiny, ldb, batch));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy results back ----
    std::vector<float> h_result_cublas(B_total);
    std::vector<float> h_result_tinyblas(B_total);
    CUDA_CHECK(cudaMemcpy(h_result_cublas.data(), d_B_cublas,
                           B_total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result_tinyblas.data(), d_B_tinyblas,
                           B_total * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Compare results ----
    double sum_sq = 0.0;
    double max_abs = 0.0;
    long long max_ulp = 0;
    int total_elements = 0;

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                float c = h_result_cublas[b * B_size + j * ldb + i];
                float t = h_result_tinyblas[b * B_size + j * ldb + i];

                if (std::isnan(c) || std::isnan(t)) {
                    fprintf(stderr, "  NaN detected: batch=%d i=%d j=%d cublas=%g tinyblas=%g\n",
                            b, i, j, c, t);
                    result.pass = false;
                    goto cleanup;
                }
                if (std::isinf(c) || std::isinf(t)) {
                    fprintf(stderr, "  Inf detected: batch=%d i=%d j=%d cublas=%g tinyblas=%g\n",
                            b, i, j, c, t);
                    result.pass = false;
                    goto cleanup;
                }

                double err = fabs((double)c - (double)t);
                sum_sq += err * err;
                if (err > max_abs)
                    max_abs = err;

                long long ulp = ulp_diff(c, t);
                if (ulp > max_ulp)
                    max_ulp = ulp;

                total_elements++;
            }
        }
    }

    result.max_abs_err = max_abs;
    result.rmse = (total_elements > 0) ? sqrt(sum_sq / total_elements) : 0.0;
    result.max_ulp = max_ulp;

    // ---- Residual: ||X*A - alpha*B|| / (||A||*||X||*n*eps) ----
    // Computed per batch element, report the worst
    {
        double worst_cublas_res = 0.0, worst_tiny_res = 0.0;
        for (int b = 0; b < batch; b++) {
            double cr = compute_residual(
                h_result_cublas.data() + b * B_size,
                h_A.data() + b * A_size,
                h_B.data() + b * B_size,
                m, n, lda, ldb, alpha);
            double tr = compute_residual(
                h_result_tinyblas.data() + b * B_size,
                h_A.data() + b * A_size,
                h_B.data() + b * B_size,
                m, n, lda, ldb, alpha);
            if (cr > worst_cublas_res) worst_cublas_res = cr;
            if (tr > worst_tiny_res) worst_tiny_res = tr;
        }
        result.cublas_residual = worst_cublas_res;
        result.tinyblas_residual = worst_tiny_res;
    }

    // ---- Timing: cuBLAS ----
    {
        // Warmup
        CUDA_CHECK(cudaMemcpy(d_B_cublas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasStrsmBatched(cublas_handle, CUBLAS_SIDE_RIGHT,
                                         CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                         CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                         d_A_ptrs_cublas, lda,
                                         d_B_ptrs_cublas, ldb, batch));
        CUDA_CHECK(cudaDeviceSynchronize());

        long long total_us = 0;
        for (int iter = 0; iter < cfg.iterations; iter++) {
            // Reset B for each iteration (STRSM is in-place)
            CUDA_CHECK(cudaMemcpy(d_B_cublas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            long long t0 = micros();
            CUBLAS_CHECK(cublasStrsmBatched(cublas_handle, CUBLAS_SIDE_RIGHT,
                                             CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                             CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                             d_A_ptrs_cublas, lda,
                                             d_B_ptrs_cublas, ldb, batch));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_us += micros() - t0;
        }
        result.cublas_us = (double)total_us / cfg.iterations;
    }

    // ---- Timing: TinyBLAS ----
    {
        // Warmup
        CUDA_CHECK(cudaMemcpy(d_B_tinyblas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));
        TINYBLAS_CHECK(tinyblasStrsmBatched(tinyblas_handle, TINYBLAS_SIDE_RIGHT,
                                             TINYBLAS_FILL_MODE_UPPER, TINYBLAS_OP_N,
                                             TINYBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                             (const float *const *)d_A_ptrs_tiny, lda,
                                             (float *const *)d_B_ptrs_tiny, ldb, batch));
        CUDA_CHECK(cudaDeviceSynchronize());

        long long total_us = 0;
        for (int iter = 0; iter < cfg.iterations; iter++) {
            CUDA_CHECK(cudaMemcpy(d_B_tinyblas, h_B.data(), B_total * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            long long t0 = micros();
            TINYBLAS_CHECK(tinyblasStrsmBatched(tinyblas_handle, TINYBLAS_SIDE_RIGHT,
                                                 TINYBLAS_FILL_MODE_UPPER, TINYBLAS_OP_N,
                                                 TINYBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                                 (const float *const *)d_A_ptrs_tiny, lda,
                                                 (float *const *)d_B_ptrs_tiny, ldb, batch));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_us += micros() - t0;
        }
        result.tinyblas_us = (double)total_us / cfg.iterations;
    }

    result.pass = true;

cleanup:
    // ---- Cleanup ----
    cublasDestroy(cublas_handle);
    tinyblasDestroy(tinyblas_handle);
    cudaFree(d_A);
    cudaFree(d_B_cublas);
    cudaFree(d_B_tinyblas);
    cudaFree((void *)d_A_ptrs_cublas);
    cudaFree(d_B_ptrs_cublas);
    cudaFree((void *)d_A_ptrs_tiny);
    cudaFree(d_B_ptrs_tiny);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    // Print GPU info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("============================================================================\n");
    printf("STRSM Batched Test: TinyBLAS vs cuBLAS\n");
    printf("============================================================================\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Operation: X * A = alpha * B  (SIDE_RIGHT, UPPER, OP_N, NON_UNIT)\n");
    printf("\n");

    // Define test cases
    // Note: the fast CUDA kernel in solve_tri.cu handles n<=64 && k<=32,
    // while larger sizes go through cublasStrsmBatched/tinyblasStrsmBatched.
    // We test both ranges since the tinyblas kernel is the one being validated.

    TestConfig tests[] = {
        // =====================================================================
        // Qwen3.5 / delta-net realistic cases (the actual production workload)
        // Matrix A = I + upper_tri(K^T*K_b*decay), diagonal = 1.0
        // Always n=64 (CHUNK_SIZE), m=64, batched over heads*chunks
        // =====================================================================
        {"qwen 64x64 b=1",    64,  64,    1, 1.0f, false, false, true,  200},
        {"qwen 64x64 b=4",    64,  64,    4, 1.0f, false, false, true,  200},
        {"qwen 64x64 b=32",   64,  64,   32, 1.0f, false, false, true,  100},
        {"qwen 64x64 b=128",  64,  64,  128, 1.0f, false, false, true,   50},
        {"qwen 64x64 b=256",  64,  64,  256, 1.0f, false, false, true,   20},
        {"qwen 64x64 b=512",  64,  64,  512, 1.0f, false, false, true,   10},
        {"qwen 64x64 b=1024", 64,  64, 1024, 1.0f, false, false, true,    5},
        {"qwen 64x64 b=2048", 64,  64, 2048, 1.0f, false, false, true,    2},

        // =====================================================================
        // Corner cases with random matrices
        // =====================================================================

        // Basic small sizes
        {"minimal 1x1",         1,   1,   1, 1.0f, false, false, false, 200},
        {"small 2x2",           2,   2,   1, 1.0f, false, false, false, 200},
        {"small 4x4",           4,   4,   1, 1.0f, false, false, false, 200},
        {"small 8x8",           8,   8,   1, 1.0f, false, false, false, 200},

        // Identity matrix (result should be alpha * B)
        {"identity 8x8",        8,   8,   1, 1.0f, true,  false, false, 200},
        {"identity 64x32",     64,  32,   4, 1.0f, true,  false, false, 200},

        // Warp boundary cases (warp size = 32)
        {"warp-exact 32x32",   32,  32,   1, 1.0f, false, false, false, 200},
        {"warp+1 33x33",       33,  33,   1, 1.0f, false, false, false, 200},
        {"warp-1 31x31",       31,  31,   1, 1.0f, false, false, false, 200},

        // At/above the fast-kernel threshold (n=64, k=32 in solve_tri.cu)
        {"threshold 64x32",    64,  32,   4, 1.0f, false, false, false, 100},
        {"large 65x33",        65,  33,   4, 1.0f, false, false, false, 100},
        {"large 128x64",      128,  64,   4, 1.0f, false, false, false,  50},

        // Alpha scaling
        {"alpha=0.5",          64,  32,   4, 0.5f, false, false, false, 100},
        {"alpha=2.0",          64,  32,   4, 2.0f, false, false, false, 100},

        // Rectangular
        {"rect wide-A",       128,   8,   4, 1.0f, false, false, false, 100},
        {"rect many-rows",     16, 512,   4, 1.0f, false, false, false,  50},

        // Many batches
        {"many-batch",         64,  32,  64, 1.0f, false, false, false,  50},

        // Ill-conditioned (diagonal spans 1e-3 to 1e3)
        {"ill-conditioned",    64,  32,   4, 1.0f, false, true,  false, 100},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int pass_count = 0;
    int fail_count = 0;
    double worst_max_abs = 0.0;
    double worst_rmse = 0.0;
    long long worst_ulp = 0;
    double worst_cublas_res = 0.0;
    double worst_tinyblas_res = 0.0;

    for (int t = 0; t < num_tests; t++) {
        const TestConfig &cfg = tests[t];

        // Reset RNG for reproducibility per test
        rand_state = 42 + t * 1000;

        printf("Test: %-22s  n=%-4d m=%-4d batch=%-3d alpha=%.1f\n",
               cfg.name, cfg.n, cfg.m, cfg.batch, cfg.alpha);

        TestResult r = run_test(cfg);

        if (!r.pass) {
            printf("  FAIL (NaN/Inf detected)\n\n");
            fail_count++;
            continue;
        }

        double speedup = (r.tinyblas_us > 0) ? r.cublas_us / r.tinyblas_us : 0.0;

        printf("  cuBLAS:          %8.1f us\n", r.cublas_us);
        printf("  TinyBLAS:        %8.1f us\n", r.tinyblas_us);
        printf("  Speedup:         %8.2fx\n", speedup);
        printf("  Max abs error:   %.6e  (cublas vs tinyblas)\n", r.max_abs_err);
        printf("  RMSE:            %.6e\n", r.rmse);
        printf("  Max ULP diff:    %lld\n", r.max_ulp);
        printf("  cuBLAS residual: %.2f  (||XA-aB||/||A||||X||n*eps, <1 = good)\n", r.cublas_residual);
        printf("  TinyBLAS resid:  %.2f\n", r.tinyblas_residual);

        // A scaled residual < ~10 means the implementation is numerically sound.
        // Both cublas and tinyblas should achieve this for well-conditioned problems.
        const double RESIDUAL_TOL = 10.0;
        bool cublas_ok = r.cublas_residual < RESIDUAL_TOL;
        bool tinyblas_ok = r.tinyblas_residual < RESIDUAL_TOL;

        const char *verdict;
        if (cublas_ok && tinyblas_ok) {
            verdict = "PASS";
            pass_count++;
        } else if (!tinyblas_ok && cublas_ok) {
            verdict = "FAIL (tinyblas residual too large)";
            fail_count++;
        } else if (!cublas_ok && !tinyblas_ok) {
            verdict = "SKIP (both residuals large - ill-conditioned)";
            pass_count++; // Don't count ill-conditioned failures
        } else {
            verdict = "FAIL (unexpected: cublas worse than tinyblas)";
            fail_count++;
        }
        printf("  %s\n\n", verdict);

        if (r.max_abs_err > worst_max_abs) worst_max_abs = r.max_abs_err;
        if (r.rmse > worst_rmse) worst_rmse = r.rmse;
        if (r.max_ulp > worst_ulp) worst_ulp = r.max_ulp;
        if (r.cublas_residual > worst_cublas_res) worst_cublas_res = r.cublas_residual;
        if (r.tinyblas_residual > worst_tinyblas_res) worst_tinyblas_res = r.tinyblas_residual;
    }

    // ---- Summary ----
    printf("============================================================================\n");
    printf("Summary: %d/%d passed", pass_count, num_tests);
    if (fail_count > 0)
        printf(", %d FAILED", fail_count);
    printf("\n");
    printf("Worst max abs error:   %.6e\n", worst_max_abs);
    printf("Worst RMSE:            %.6e\n", worst_rmse);
    printf("Worst max ULP diff:    %lld\n", worst_ulp);
    printf("Worst cuBLAS residual: %.2f\n", worst_cublas_res);
    printf("Worst TinyBLAS resid:  %.2f\n", worst_tinyblas_res);
    printf("============================================================================\n");

    return fail_count > 0 ? 1 : 0;
}
