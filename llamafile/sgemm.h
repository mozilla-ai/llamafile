#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

bool iqk_mul_mat(long, long, long, int, const void *, const void *, float *, long, int, int);
bool iqk_mul_mat_zen4(long, long, long, int, const void *, const void *, float *, long, int, int);
bool iqk_mul_mat_arm82(long, long, long, int, const void *, const void *, float *, long, int, int);

bool iqk_mul_mat_moe(long, long, long, int, int, const void *, const void *, float *, long, long,
                     const void *, int, int);
bool iqk_mul_mat_moe_zen4(long, long, long, int, int, const void *, const void *, float *, long,
                          long, const void *, int, int);
bool iqk_mul_mat_moe_arm82(long, long, long, int, int, const void *, const void *, float *, long,
                           long, const void *, int, int);
bool iqk_mul_mat_moe_unsupported(long, long, long, int, int, const void *, const void *, float *,
                                 long, long, const void *, int, int);

// Public API - matches upstream llama.cpp signature
// Guarded to prevent macro expansion in internal arch-specific implementations
#ifndef llamafile_sgemm
bool llamafile_sgemm(const struct ggml_compute_params *, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);
#endif
bool llamafile_mixmul(const struct ggml_compute_params *, const struct ggml_tensor *,
                      const struct ggml_tensor *, const struct ggml_tensor *, struct ggml_tensor *);
size_t llamafile_mixmul_needs(const struct ggml_tensor *, const struct ggml_tensor *,
                              const struct ggml_tensor *);

// Returns the name of the selected sgemm kernel for diagnostics
const char *llamafile_sgemm_name(void);

// Internal arch-specific implementations (called by dispatcher)
bool llamafile_sgemm_unsupported(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int);
bool llamafile_sgemm_amd_avx(long, long, long, const void *, long, const void *, long, void *, long,
                             int, int, int, int, int);
bool llamafile_sgemm_amd_fma(long, long, long, const void *, long, const void *, long, void *, long,
                             int, int, int, int, int);
bool llamafile_sgemm_amd_avx2(long, long, long, const void *, long, const void *, long, void *,
                              long, int, int, int, int, int);
bool llamafile_sgemm_amd_avxvnni(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int);
bool llamafile_sgemm_amd_avx512f(long, long, long, const void *, long, const void *, long, void *,
                                 long, int, int, int, int, int);
bool llamafile_sgemm_amd_zen4(long, long, long, const void *, long, const void *, long, void *,
                              long, int, int, int, int, int);
bool llamafile_sgemm_arm80(long, long, long, const void *, long, const void *, long, void *, long,
                           int, int, int, int, int);
bool llamafile_sgemm_arm82(long, long, long, const void *, long, const void *, long, void *, long,
                           int, int, int, int, int);

bool llamafile_mixmul_unsupported(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx(const struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_fma(const struct ggml_compute_params *, const struct ggml_tensor *,
                              const struct ggml_tensor *, const struct ggml_tensor *,
                              struct ggml_tensor *);
bool llamafile_mixmul_amd_avx2(const struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_amd_avxvnni(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_avx512f(const struct ggml_compute_params *, const struct ggml_tensor *,
                                  const struct ggml_tensor *, const struct ggml_tensor *,
                                  struct ggml_tensor *);
bool llamafile_mixmul_amd_zen4(const struct ggml_compute_params *, const struct ggml_tensor *,
                               const struct ggml_tensor *, const struct ggml_tensor *,
                               struct ggml_tensor *);
bool llamafile_mixmul_arm80(const struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);
bool llamafile_mixmul_arm82(const struct ggml_compute_params *, const struct ggml_tensor *,
                            const struct ggml_tensor *, const struct ggml_tensor *,
                            struct ggml_tensor *);
bool llamafile_mixmul_iqk(long, long, long, int, int, const void *, const void *, float *, long,
                          long, const void *, int, int);

// Flash-attention helpers (issue #975). Optimized replacements for the
// hot ggml-cpu inner-loop helpers that cosmocc compiles at baseline
// AVX2. Returns true when handled by the optimized kernel; caller
// falls back to upstream's ggml helper on false. Public API takes
// void* so callers don't need ggml.h.
bool llamafile_fa_vec_dot_f16(int n, float *s, const void *x, const void *y);
bool llamafile_fa_fp16_to_fp32_row(const void *x, float *y, int64_t n);
// C[M x N] += A[M x K] * B[K x N], all f32.
bool llamafile_fa_simd_gemm(float *C, const float *A, const float *B,
                             int M, int K, int N);

// Internal arch-specific implementations of the FA helpers.
bool llamafile_fa_vec_dot_f16_amd_avx512f(int, float *, const void *, const void *);
bool llamafile_fa_vec_dot_f16_unsupported(int, float *, const void *, const void *);
bool llamafile_fa_fp16_to_fp32_row_amd_avx512f(const void *, float *, int64_t);
bool llamafile_fa_fp16_to_fp32_row_unsupported(const void *, float *, int64_t);
bool llamafile_fa_simd_gemm_amd_avx512f(float *, const float *, const float *,
                                         int, int, int);
bool llamafile_fa_simd_gemm_unsupported(float *, const float *, const float *,
                                         int, int, int);

#ifdef __cplusplus
}
#endif
