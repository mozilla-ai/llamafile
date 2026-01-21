#ifdef __x86_64__
#define llamafile_mixmul llamafile_mixmul_amd_avx2
#include "tinyblas_cpu_mixmul.inc"
#endif // __x86_64__
