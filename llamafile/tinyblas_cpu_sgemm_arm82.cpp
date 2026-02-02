#ifdef __aarch64__
#define llamafile_sgemm llamafile_sgemm_arm82
#define iqk_mul_mat iqk_mul_mat_arm82
#include "tinyblas_cpu_sgemm.inc"
#endif // __aarch64__
