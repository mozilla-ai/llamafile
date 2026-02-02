#ifdef __x86_64__
#define iqk_mul_mat iqk_mul_mat_zen4
#define iqk_mul_mat_moe iqk_mul_mat_moe_zen4
#include "iqk_mul_mat.inc"
#endif // __x86_64__
