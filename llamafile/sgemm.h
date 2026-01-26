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

#ifdef __cplusplus
}
#endif
