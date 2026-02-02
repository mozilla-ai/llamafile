#ifdef __x86_64__
#define llamafile_sgemm llamafile_sgemm_amd_avxvnni
#include "tinyblas_cpu_sgemm.inc"
#endif // __x86_64__
