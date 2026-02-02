#ifdef __x86_64__
#define llamafile_sgemm llamafile_sgemm_amd_zen4
#define iqk_mul_mat iqk_mul_mat_zen4
#include "tinyblas_cpu_sgemm.inc"
#endif // __x86_64__
