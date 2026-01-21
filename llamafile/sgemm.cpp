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

#include "sgemm.h"
#include <cassert>
#include <cosmo.h>
#include <cpuid.h>
#include <libc/sysv/consts/hwcap.h>
#include <sys/auxv.h>

static const struct GemmFuncs {
    typeof(llamafile_sgemm) *sgemm;
    typeof(llamafile_mixmul) *mixmul;
    GemmFuncs() {
#ifdef __x86_64__
        if (X86_HAVE(AVX)) {
            if (X86_HAVE(FMA)) {
                if (X86_HAVE(AVX2)) {
                    if (X86_HAVE(AVX512F)) {
                        if (X86_HAVE(AVX512VL) && //
                            X86_HAVE(AVX512BW) && //
                            X86_HAVE(AVX512DQ) && //
                            X86_HAVE(AVX512_VNNI) && //
                            X86_HAVE(AVX512_BF16)) {
                            // AMD Zen4+ (2023-)
                            sgemm = llamafile_sgemm_amd_zen4;
                            mixmul = llamafile_mixmul_amd_zen4;
                        } else {
                            // Intel Xeon Skylake+ (2015-)
                            sgemm = llamafile_sgemm_amd_avx512f;
                            mixmul = llamafile_mixmul_amd_avx512f;
                        }
                    } else if (X86_HAVE(AVXVNNI)) {
                        // Intel Alderlake (2021-)
                        sgemm = llamafile_sgemm_amd_avxvnni;
                        mixmul = llamafile_mixmul_amd_avxvnni;
                    } else {
                        // Intel Haswell/Broadwell/Skylake (2013-2020)
                        // AMD Excavator (2015-2022)
                        sgemm = llamafile_sgemm_amd_avx2;
                        mixmul = llamafile_mixmul_amd_avx2;
                    }
                } else {
                    // AMD Piledriver (2011-2014)
                    sgemm = llamafile_sgemm_amd_fma;
                    mixmul = llamafile_mixmul_amd_fma;
                }
            } else {
                // Intel Sandybridge/Ivybridge (2010-2012)
                // AMD Bulldozer (2011)
                sgemm = llamafile_sgemm_amd_avx;
                mixmul = llamafile_mixmul_amd_avx;
            }
        } else {
            // AMD K8/Barcelona (2003-2010)
            // Intel Core/Nehalem (2006-2009)
            sgemm = llamafile_sgemm_unsupported;
            mixmul = llamafile_mixmul_unsupported;
        }
#elif defined(__aarch64__)
        long hwcap = getauxval(AT_HWCAP);
        if ((hwcap & HWCAP_FPHP) && // fp16 scalar isa (ID_AA64PFR0_EL1.FP == 1)
            (hwcap & HWCAP_ASIMDHP) && // fp16 vector isa (ID_AA64PFR0_EL1.AdvSIMD == 1)
            (hwcap & HWCAP_ASIMDDP)) { // dotprod isa (ID_AA64ISAR0_EL1.DP == 1)
            // e.g. Apple M1, Raspberry Pi 5
            sgemm = llamafile_sgemm_arm82;
            mixmul = llamafile_mixmul_arm82;
        } else {
            // ARM64 baseline ISA
            sgemm = llamafile_sgemm_arm80;
            mixmul = llamafile_mixmul_arm80;
        }
#else
        sgemm = llamafile_sgemm_unsupported;
        mixmul = llamafile_mixmul_unsupported;
#endif
    }
} funcs;

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(long m, long n, long k, const void *A, long lda, const void *B, long ldb,
                     void *C, long ldc, int ith, int nth, int Atype, int Btype, int Ctype) {
    return funcs.sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, Atype, Btype, Ctype);
}

/**
 * Performs "mixture of experts" tensor multiplication on CPU.
 */
bool llamafile_mixmul(const ggml_compute_params *params, const ggml_tensor *weights,
                      const ggml_tensor *thought, const ggml_tensor *plan, ggml_tensor *result) {
    return funcs.mixmul(params, weights, thought, plan, result);
}

// llamafile_mixmul_needs is defined in tinyblas_cpu_mixmul_*.cpp files
