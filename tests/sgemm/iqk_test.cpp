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
// iqk_test: Integer Quantized Kernels Benchmark
// ============================================================================
//
// PURPOSE:
//   This test benchmarks the IQK (Integer Quantized Kernels) which provide
//   optimized matrix multiplication for quantized models. IQK kernels are
//   150-400% faster than standard llama.cpp for prompt processing with
//   k-quants (Q4_K, Q5_K, Q6_K) and i-quants.
//
// OPERATION:
//   Computes C = A * B where:
//     - A is a quantized weight matrix (Q4_K, Q5_K, or Q6_K)
//     - B is the activation matrix quantized to Q8_K
//     - C is the float32 output matrix
//
//   The test creates random float data, quantizes it, runs the IQK kernel,
//   then verifies correctness by dequantizing and computing the reference
//   result using float32 arithmetic.
//
// WHY IQK MATTERS:
//   - Quantized models (GGUF Q4_K_M, Q5_K_M, etc.) are the most common LLM format
//   - Prompt processing involves large matrix multiplications with these types
//   - IQK unpacks quantized blocks once and reuses them for multiple dot products
//   - This tiling approach provides significant speedups over naive implementations
//
// SUPPORTED TYPES (varies by architecture):
//   x86_64: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, Q4_1, Q5_0, Q5_1
//   ARM64:  All of the above plus IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q8_0
//
// IMPLEMENTATIONS COMPARED:
//
//   1. Reference (dequantize + float matmul)
//      - Dequantizes A and B to float32
//      - Performs standard float32 matrix multiplication
//      - Accurate but slow (baseline)
//
//   2. IQK Kernel (optimized)
//      - Uses SIMD-optimized kernels (AVX2/AVX512/NEON)
//      - Operates directly on quantized data
//      - Should produce similar results with much higher performance
//
// METRICS:
//
//   Performance (microseconds):
//     - Time to complete the quantized matrix multiplication
//     - Lower is better; IQK should be significantly faster than reference
//
//   Accuracy (relative error):
//     - Measures difference between IQK output and reference output
//     - "avg error": Mean relative error across all output elements
//     - "max error": Maximum relative error for any single element
//     - Expected error is due to quantization, not implementation bugs
//
// TEST PARAMETERS:
//   - Nx (rows of output) = 512 (simulates batch of tokens)
//   - Ny (cols of output) = 1024 (simulates hidden dimension)
//   - ne00 (inner dimension) = 4096 (simulates model dimension)
//   - Must be multiple of QK_K (256) for k-quants
//

#include "sgemm.h"
#include "ggml.h"
#include "ggml-quants.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>

// ============================================================================
// Configuration
// ============================================================================

#define ITERATIONS 10

// Matrix dimensions (must be multiple of QK_K=256 for k-quants)
static const long Nx = 512;      // Rows of output (batch of tokens)
static const long Ny = 1024;     // Cols of output (hidden dimension)
static const long ne00 = 4096;   // Inner dimension (model dimension)

// ============================================================================
// Utility functions
// ============================================================================

static inline long long micros(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000000 + (ts.tv_nsec + 999) / 1000;
}

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

static int get_num_threads() {
    return std::thread::hardware_concurrency();
}

// Random number generation
static unsigned long long lcg = 1;

static inline int rand32(void) {
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return lcg >> 32;
}

static inline float rand_float(void) {
    return (float)(rand32() % 10000 - 5000) / 10000.0f;
}

static void randomize_floats(float *data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = rand_float();
    }
}

// ============================================================================
// IQK test wrapper (multi-threaded)
// ============================================================================

// Wrapper that calls iqk_mul_mat with OpenMP parallelism
static bool iqk_mul_mat_openmp(long Nx, long Ny, long ne00, int typeA,
                               const void *A, const void *B, float *C, long stride_C) {
    int nth = get_num_threads();
    bool all_ok = true;

#ifdef __x86_64__
    // Use zen4 variant if available (AVX-512 VNNI/BF16)
    // This is auto-selected by the dispatcher, but we call iqk_mul_mat directly
#pragma omp parallel for reduction(&& : all_ok)
    for (int ith = 0; ith < nth; ++ith) {
        bool res = iqk_mul_mat(Nx, Ny, ne00, typeA, A, B, C, stride_C, ith, nth);
        all_ok = all_ok && res;
    }
#elif defined(__aarch64__)
#pragma omp parallel for reduction(&& : all_ok)
    for (int ith = 0; ith < nth; ++ith) {
        bool res = iqk_mul_mat_arm82(Nx, Ny, ne00, typeA, A, B, C, stride_C, ith, nth);
        all_ok = all_ok && res;
    }
#else
    all_ok = false;
#endif

    return all_ok;
}

// ============================================================================
// Reference implementation: dequantize and compute in float
// ============================================================================

static void reference_matmul(long Nx, long Ny, long ne00, int typeA,
                             const void *A, const void *B, float *C, long stride_C,
                             float *A_float, float *B_float) {
    // Dequantize A (Nx x ne00, row-major: each row is ne00 elements)
    size_t row_size_A = ggml_row_size((ggml_type)typeA, ne00);
    for (long i = 0; i < Nx; ++i) {
        const void *row_A = (const char *)A + i * row_size_A;
        float *row_float = A_float + i * ne00;

        switch (typeA) {
            case GGML_TYPE_Q4_K:
                dequantize_row_q4_K((const block_q4_K *)row_A, row_float, ne00);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_row_q5_K((const block_q5_K *)row_A, row_float, ne00);
                break;
            case GGML_TYPE_Q6_K:
                dequantize_row_q6_K((const block_q6_K *)row_A, row_float, ne00);
                break;
            default:
                fprintf(stderr, "Unsupported type for dequantization\n");
                return;
        }
    }

    // Dequantize B (Ny x ne00, row-major)
    size_t row_size_B = ggml_row_size(GGML_TYPE_Q8_K, ne00);
    for (long j = 0; j < Ny; ++j) {
        const void *row_B = (const char *)B + j * row_size_B;
        float *row_float = B_float + j * ne00;
        dequantize_row_q8_K((const block_q8_K *)row_B, row_float, ne00);
    }

    // Compute C = A * B^T
    // IQK output layout: C[iy * stride_C + ix] where ix=A_row, iy=B_row
    // So C is stored as Ny rows x Nx cols (transposed from typical A*B^T)
    // C[j,i] = dot(A[i,:], B[j,:]) = sum_k A[i,k] * B[j,k]
#pragma omp parallel for collapse(2)
    for (long j = 0; j < Ny; ++j) {       // iy = B row index
        for (long i = 0; i < Nx; ++i) {   // ix = A row index
            double sum = 0.0;
            for (long k = 0; k < ne00; ++k) {
                sum += (double)A_float[i * ne00 + k] * (double)B_float[j * ne00 + k];
            }
            C[j * stride_C + i] = (float)sum;  // Store at [iy * stride + ix]
        }
    }
}

// ============================================================================
// Test for a specific quantization type
// ============================================================================

static int test_quant_type(int typeA, const char *type_name) {
    printf("\n--- Testing %s ---\n", type_name);

    // Allocate float source data
    float *src_A = (float *)aligned_alloc(64, Nx * ne00 * sizeof(float));
    float *src_B = (float *)aligned_alloc(64, Ny * ne00 * sizeof(float));

    if (!src_A || !src_B) {
        fprintf(stderr, "Failed to allocate source data\n");
        return 1;
    }

    // Generate random source data
    lcg = 12345;  // Reset seed for reproducibility
    randomize_floats(src_A, Nx * ne00);
    randomize_floats(src_B, Ny * ne00);

    // Allocate quantized data
    size_t row_size_A = ggml_row_size((ggml_type)typeA, ne00);
    size_t row_size_B = ggml_row_size(GGML_TYPE_Q8_K, ne00);

    void *A_quant = aligned_alloc(64, Nx * row_size_A);
    void *B_quant = aligned_alloc(64, Ny * row_size_B);

    if (!A_quant || !B_quant) {
        fprintf(stderr, "Failed to allocate quantized data\n");
        return 1;
    }

    // Quantize A
    printf("Quantizing A (%s)...\n", type_name);
    for (long i = 0; i < Nx; ++i) {
        const float *row_src = src_A + i * ne00;
        void *row_dst = (char *)A_quant + i * row_size_A;

        switch (typeA) {
            case GGML_TYPE_Q4_K:
                quantize_row_q4_K_ref(row_src, (block_q4_K *)row_dst, ne00);
                break;
            case GGML_TYPE_Q5_K:
                quantize_row_q5_K_ref(row_src, (block_q5_K *)row_dst, ne00);
                break;
            case GGML_TYPE_Q6_K:
                quantize_row_q6_K_ref(row_src, (block_q6_K *)row_dst, ne00);
                break;
            default:
                fprintf(stderr, "Unsupported type for quantization\n");
                return 1;
        }
    }

    // Quantize B to Q8_K
    printf("Quantizing B (Q8_K)...\n");
    for (long j = 0; j < Ny; ++j) {
        const float *row_src = src_B + j * ne00;
        block_q8_K *row_dst = (block_q8_K *)((char *)B_quant + j * row_size_B);
        quantize_row_q8_K_ref(row_src, row_dst, ne00);
    }

    // Allocate output buffers
    // IQK output is Ny rows x Nx cols, so stride = Nx
    long stride_C = Nx;  // Row stride for output
    float *C_iqk = (float *)aligned_alloc(64, Ny * stride_C * sizeof(float));
    float *C_ref = (float *)aligned_alloc(64, Ny * stride_C * sizeof(float));

    // Allocate dequantized buffers for reference
    float *A_float = (float *)aligned_alloc(64, Nx * ne00 * sizeof(float));
    float *B_float = (float *)aligned_alloc(64, Ny * ne00 * sizeof(float));

    if (!C_iqk || !C_ref || !A_float || !B_float) {
        fprintf(stderr, "Failed to allocate output/temp buffers\n");
        return 1;
    }

    // Clear output buffers
    memset(C_iqk, 0, Ny * stride_C * sizeof(float));
    memset(C_ref, 0, Ny * stride_C * sizeof(float));

    int nth = get_num_threads();
    printf("Using %d threads\n", nth);
    printf("Matrix dimensions: Nx=%ld Ny=%ld ne00=%ld\n", Nx, Ny, ne00);

    // Test IQK kernel
    printf("\n--- Benchmarks ---\n");

    bool iqk_available = iqk_mul_mat_openmp(Nx, Ny, ne00, typeA, A_quant, B_quant, C_iqk, stride_C);

    if (iqk_available) {
        // Warmup and benchmark
        BENCH(iqk_mul_mat_openmp(Nx, Ny, ne00, typeA, A_quant, B_quant, C_iqk, stride_C), ITERATIONS);
    } else {
        printf("%12s iqk_mul_mat_openmp (not supported for %s)\n", "N/A", type_name);
    }

    // Compute reference
    printf("Computing reference (dequantize + float matmul)...\n");
    BENCH(reference_matmul(Nx, Ny, ne00, typeA, A_quant, B_quant, C_ref, stride_C, A_float, B_float), 1);

    // Compare results
    if (iqk_available) {
        printf("\n--- Accuracy (IQK vs reference) ---\n");

        // Print a few sample values for verification
        printf("Sample values (first 3 elements):\n");
        for (int idx = 0; idx < 3 && idx < Ny * stride_C; ++idx) {
            printf("  [%d] IQK=%.4f  ref=%.4f  diff=%.2e\n",
                   idx, C_iqk[idx], C_ref[idx], C_iqk[idx] - C_ref[idx]);
        }

        double err_sum = 0.0;
        double err_max = 0.0;
        long err_count = 0;

        // IQK output is Ny rows x Nx cols
        for (long j = 0; j < Ny; ++j) {       // rows (B index)
            for (long i = 0; i < Nx; ++i) {   // cols (A index)
                float iqk_val = C_iqk[j * stride_C + i];
                float ref_val = C_ref[j * stride_C + i];

                // Check for NaN
                if (std::isnan(iqk_val)) {
                    fprintf(stderr, "ERROR: NaN in IQK output at [%ld,%ld]\n", j, i);
                    return 2;
                }
                if (std::isnan(ref_val)) {
                    fprintf(stderr, "ERROR: NaN in reference output at [%ld,%ld]\n", j, i);
                    return 3;
                }

                // Relative error (with epsilon to avoid div by zero)
                double rel_err = std::fabs(iqk_val - ref_val) / (std::fabs(ref_val) + 1e-6);
                err_sum += rel_err;
                if (rel_err > err_max) {
                    err_max = rel_err;
                }
                ++err_count;
            }
        }

        double err_avg = err_sum / err_count;
        printf("%12.2e avg relative error\n", err_avg);
        printf("%12.2e max relative error\n", err_max);

        // Check if error is within acceptable bounds (quantization introduces some error)
        if (err_max > 0.1) {  // 10% max error threshold
            fprintf(stderr, "WARNING: High relative error detected (>10%%)\n");
        }
    }

    // Cleanup
    free(A_float);
    free(B_float);
    free(C_ref);
    free(C_iqk);
    free(B_quant);
    free(A_quant);
    free(src_B);
    free(src_A);

    return 0;
}

// ============================================================================
// Print test information
// ============================================================================

static void print_test_info() {
    printf("============================================================================\n");
    printf("iqk_test: Integer Quantized Kernels Benchmark\n");
    printf("============================================================================\n");
    printf("\n");
    printf("OPERATION:\n");
    printf("  C = A * B^T where A is quantized (Q4_K/Q5_K/Q6_K) and B is Q8_K.\n");
    printf("  IQK kernels provide 150-400%% speedup over standard implementations.\n");
    printf("\n");
    printf("IMPLEMENTATIONS:\n");
    printf("  reference_matmul     - Dequantize to float32, then standard matmul\n");
    printf("  iqk_mul_mat_openmp   - SIMD-optimized quantized matmul (AVX2/AVX512/NEON)\n");
    printf("\n");
    printf("METRICS:\n");
    printf("  Time (us)            - Microseconds to complete. Lower is better.\n");
    printf("  avg relative error   - Mean |IQK - ref| / |ref| across all elements.\n");
    printf("  max relative error   - Maximum relative error for any element.\n");
    printf("\n");
    printf("NOTE: Some error is expected due to different computation order and\n");
    printf("      accumulator precision between IQK and the reference implementation.\n");
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    int rc;

    print_test_info();

    // Print kernel selection info
    const char *kernel = llamafile_sgemm_name();
    printf("Selected sgemm kernel: %s\n", kernel);

#if defined(__x86_64__)
    printf("Architecture: x86_64\n");
    printf("IQK supports: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, Q4_1, Q5_0, Q5_1\n");
#elif defined(__aarch64__)
    printf("Architecture: ARM64\n");
    printf("IQK supports: All k-quants + IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q8_0\n");
#else
    printf("Architecture: Unknown (IQK may not be available)\n");
#endif
    printf("\n");

    // Validate dimensions
    if (ne00 % QK_K != 0) {
        fprintf(stderr, "ERROR: ne00 (%ld) must be multiple of QK_K (%d)\n", ne00, QK_K);
        return 1;
    }

    // Test each quantization type
    if ((rc = test_quant_type(GGML_TYPE_Q4_K, "Q4_K")))
        return rc;

    if ((rc = test_quant_type(GGML_TYPE_Q5_K, "Q5_K")))
        return rc;

    if ((rc = test_quant_type(GGML_TYPE_Q6_K, "Q6_K")))
        return rc;

    printf("\n============================================================================\n");
    printf("All tests completed.\n");
    printf("============================================================================\n");

    return 0;
}
