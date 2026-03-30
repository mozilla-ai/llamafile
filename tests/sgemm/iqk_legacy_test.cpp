// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Test for legacy quant types (Q4_0, Q4_1, Q5_0, Q5_1) through IQK path.
// Reproduces https://github.com/mozilla-ai/llamafile/issues/922
//
// Tests two classes of bugs:
// 1. Q81::load_scales layout mismatch (affects Q4_1, Q5_1 at all dimensions)
// 2. DequantizerQ51::prepare1(int) missing 5th bit (affects Q5_1 when nb%4 != 0)

#include "sgemm.h"
#include "ggml.h"
#include "ggml-quants.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static const long Nx = 32;
static const long Ny = 64;

static unsigned long long lcg = 12345;

static inline float rand_float(void) {
    lcg *= 6364136223846793005;
    lcg += 1442695040888963407;
    return (float)((int)(lcg >> 32) % 10000 - 5000) / 10000.0f;
}

// Dequantize Q8_1 (not provided by ggml-quants.h)
static void dequantize_row_q8_1(const block_q8_1 *x, float *y, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_1; ++j) {
            y[i * QK8_1 + j] = x[i].qs[j] * d;
        }
    }
}

static bool iqk_mul_mat_wrapper(long Nx, long Ny, long ne00, int typeA,
                                const void *A, const void *B, float *C, long stride_C) {
    int nth = std::thread::hardware_concurrency();
    bool all_ok = true;
    for (int ith = 0; ith < nth; ++ith) {
#if defined(__x86_64__)
        bool res = iqk_mul_mat(Nx, Ny, ne00, typeA, A, B, C, stride_C, ith, nth);
#elif defined(__aarch64__)
        bool res = iqk_mul_mat_arm82(Nx, Ny, ne00, typeA, A, B, C, stride_C, ith, nth);
#else
        bool res = false;
#endif
        all_ok = all_ok && res;
    }
    return all_ok;
}

struct QuantType {
    int type;
    const char *name;
    void (*quantize)(const float *, void *, int64_t);
    void (*dequantize)(const void *, float *, int64_t);
    int q8_type;
};

static void quantize_q4_0(const float *x, void *y, int64_t k) {
    quantize_row_q4_0_ref(x, (block_q4_0 *)y, k);
}
static void dequantize_q4_0(const void *x, float *y, int64_t k) {
    dequantize_row_q4_0((const block_q4_0 *)x, y, k);
}
static void quantize_q4_1(const float *x, void *y, int64_t k) {
    quantize_row_q4_1_ref(x, (block_q4_1 *)y, k);
}
static void dequantize_q4_1(const void *x, float *y, int64_t k) {
    dequantize_row_q4_1((const block_q4_1 *)x, y, k);
}
static void quantize_q5_0(const float *x, void *y, int64_t k) {
    quantize_row_q5_0_ref(x, (block_q5_0 *)y, k);
}
static void dequantize_q5_0(const void *x, float *y, int64_t k) {
    dequantize_row_q5_0((const block_q5_0 *)x, y, k);
}
static void quantize_q5_1(const float *x, void *y, int64_t k) {
    quantize_row_q5_1_ref(x, (block_q5_1 *)y, k);
}
static void dequantize_q5_1(const void *x, float *y, int64_t k) {
    dequantize_row_q5_1((const block_q5_1 *)x, y, k);
}

static QuantType quant_types[] = {
    {GGML_TYPE_Q4_0, "Q4_0", quantize_q4_0, dequantize_q4_0, GGML_TYPE_Q8_0},
    {GGML_TYPE_Q4_1, "Q4_1", quantize_q4_1, dequantize_q4_1, GGML_TYPE_Q8_1},
    {GGML_TYPE_Q5_0, "Q5_0", quantize_q5_0, dequantize_q5_0, GGML_TYPE_Q8_0},
    {GGML_TYPE_Q5_1, "Q5_1", quantize_q5_1, dequantize_q5_1, GGML_TYPE_Q8_1},
};

// Test a specific quant type with a given inner dimension.
// ne00 must be a multiple of 32.
static int test_type(const QuantType &qt, long ne00) {
    int nb = ne00 / 32;
    printf("\n--- %s ne00=%ld (nb=%d, nb%%4=%d) ---\n", qt.name, ne00, nb, nb % 4);

    float *src_A = (float *)aligned_alloc(64, Nx * ne00 * sizeof(float));
    float *src_B = (float *)aligned_alloc(64, Ny * ne00 * sizeof(float));

    lcg = 12345;
    for (long i = 0; i < Nx * ne00; ++i) src_A[i] = rand_float();
    for (long i = 0; i < Ny * ne00; ++i) src_B[i] = rand_float();

    size_t row_size_A = ggml_row_size((ggml_type)qt.type, ne00);
    void *A_quant = aligned_alloc(64, Nx * row_size_A);
    for (long i = 0; i < Nx; ++i) {
        qt.quantize(src_A + i * ne00, (char *)A_quant + i * row_size_A, ne00);
    }

    size_t row_size_B = ggml_row_size((ggml_type)qt.q8_type, ne00);
    void *B_quant = aligned_alloc(64, Ny * row_size_B);
    for (long j = 0; j < Ny; ++j) {
        if (qt.q8_type == GGML_TYPE_Q8_0) {
            quantize_row_q8_0_ref(src_B + j * ne00, (block_q8_0 *)((char *)B_quant + j * row_size_B), ne00);
        } else {
            quantize_row_q8_1_ref(src_B + j * ne00, (block_q8_1 *)((char *)B_quant + j * row_size_B), ne00);
        }
    }

    long stride_C = Nx;
    float *C_iqk = (float *)aligned_alloc(64, Ny * stride_C * sizeof(float));
    memset(C_iqk, 0, Ny * stride_C * sizeof(float));

    bool ok = iqk_mul_mat_wrapper(Nx, Ny, ne00, qt.type, A_quant, B_quant, C_iqk, stride_C);
    if (!ok) {
        printf("  IQK not supported, skipping\n");
        free(C_iqk); free(B_quant); free(A_quant); free(src_B); free(src_A);
        return 0;
    }

    // Reference: dequantize both sides and do float matmul
    float *A_float = (float *)aligned_alloc(64, Nx * ne00 * sizeof(float));
    float *B_float = (float *)aligned_alloc(64, Ny * ne00 * sizeof(float));

    for (long i = 0; i < Nx; ++i) {
        qt.dequantize((char *)A_quant + i * row_size_A, A_float + i * ne00, ne00);
    }
    for (long j = 0; j < Ny; ++j) {
        if (qt.q8_type == GGML_TYPE_Q8_0) {
            dequantize_row_q8_0((const block_q8_0 *)((char *)B_quant + j * row_size_B),
                                B_float + j * ne00, ne00);
        } else {
            dequantize_row_q8_1((const block_q8_1 *)((char *)B_quant + j * row_size_B),
                                B_float + j * ne00, ne00);
        }
    }

    float *C_ref = (float *)aligned_alloc(64, Ny * stride_C * sizeof(float));
    for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            double sum = 0.0;
            for (long k = 0; k < ne00; ++k) {
                sum += (double)A_float[i * ne00 + k] * (double)B_float[j * ne00 + k];
            }
            C_ref[j * stride_C + i] = (float)sum;
        }
    }

    int nan_count = 0;
    int inf_count = 0;
    double err_sum = 0.0;
    double err_max = 0.0;
    long err_count = 0;

    for (long j = 0; j < Ny; ++j) {
        for (long i = 0; i < Nx; ++i) {
            float iqk_val = C_iqk[j * stride_C + i];
            float ref_val = C_ref[j * stride_C + i];

            if (std::isnan(iqk_val)) { nan_count++; continue; }
            if (std::isinf(iqk_val)) { inf_count++; continue; }

            double rel_err = std::fabs(iqk_val - ref_val) / (std::fabs(ref_val) + 1e-6);
            err_sum += rel_err;
            if (rel_err > err_max) err_max = rel_err;
            ++err_count;
        }
    }

    int rc = 0;
    if (nan_count > 0 || inf_count > 0) {
        printf("  FAIL: %d NaN, %d Inf\n", nan_count, inf_count);
        rc = 1;
    } else if (err_max > 0.05) {
        printf("  FAIL: max error %.2e exceeds threshold\n", err_max);
        rc = 1;
    } else {
        printf("  PASS (avg=%.2e max=%.2e)\n", err_sum / err_count, err_max);
    }

    free(C_ref); free(B_float); free(A_float);
    free(C_iqk); free(B_quant); free(A_quant);
    free(src_B); free(src_A);
    return rc;
}

int main() {
    printf("iqk_legacy_test: Testing legacy quant types through IQK path\n");
    printf("Threads: %u\n", std::thread::hardware_concurrency());

    // ne00 must be a multiple of 256 because iqk_mul_mat unconditionally
    // computes ggml_row_size(Q8_K, ne00) which asserts ne00 % 256 == 0.
    // Since 256/32 = 8 blocks and 8%4 == 0, the process_1_block remainder
    // path (nb%4 != 0) is unreachable for legacy quants through this API.
    // The prepare1 fix for DequantizerQ51 is still correct but cannot be
    // exercised here. We test multiple dimensions for general coverage.
    static const long test_dims[] = { 256, 512, 1024 };

    int failures = 0;
    for (const auto &qt : quant_types) {
        for (long ne00 : test_dims) {
            if (test_type(qt, ne00) != 0)
                failures++;
        }
    }

    printf("\n============================================\n");
    if (failures > 0) {
        printf("FAILED: %d test(s) failed\n", failures);
        return 1;
    }
    printf("All tests passed.\n");
    return 0;
}
