// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/sgemm.cpp
// Copyrigth 2024 Mozilla Foundation.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

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
// #include <cosmo.h>
// #include <cpuid.h>
// #include <libc/sysv/consts/hwcap.h>
#include <stdio.h>
#include <sys/auxv.h>
#include <cassert>
// #include "llamafile.h"

static const struct GemmFuncs {
    bool (*sgemm)(long, long, long, const void*, long, const void*, long, void*, long, int, int, int, int, int, int, int);
    bool (*mixmul)(const struct ggml_compute_params*, const struct ggml_tensor*, const struct ggml_tensor*, const struct ggml_tensor*, struct ggml_tensor*);
    bool (*iqk_mixmul)(long, long, long, int, int, const void*, const void*, float*, long, long, const void*, int, int);
    // typeof(llamafile_sgemm)* sgemm;
    // typeof(llamafile_mixmul)* mixmul;
    // typeof(llamafile_mixmul_iqk)* iqk_mixmul = iqk_mul_mat_moe_unsupported;
    GemmFuncs() {
//#if defined(__x86_64__) || defined(_M_X64)
        // if (X86_HAVE(AVX)) {
        //     if (X86_HAVE(FMA)) {
        //         if (X86_HAVE(AVX2)) {
        //             if (X86_HAVE(AVX512F)) {
        //                 if (X86_HAVE(AVX512VL) &&     //
        //                     X86_HAVE(AVX512BW) &&     //
        //                     X86_HAVE(AVX512DQ) &&     //
        //                     X86_HAVE(AVX512_VNNI) &&  //
        //                     X86_HAVE(AVX512_BF16)) {
        //                     // AMD Zen4+ (2023-)
        //                     sgemm = llamafile_sgemm_amd_zen4;
        //                     mixmul = llamafile_mixmul_amd_zen4;
        //                     iqk_mixmul = iqk_mul_mat_moe_zen4;
        //                 } else {
        //                     // Intel Xeon Skylake+ (2015-)
        //                     sgemm = llamafile_sgemm_amd_avx512f;
        //                     mixmul = llamafile_mixmul_amd_avx512f;
        //                     iqk_mixmul = iqk_mul_mat_moe;
        //                 }
        //             } else if (X86_HAVE(AVXVNNI)) {
        //                 // Intel Alderlake (2021-)
        //                 sgemm = llamafile_sgemm_amd_avxvnni;
        //                 mixmul = llamafile_mixmul_amd_avxvnni;
        //                 iqk_mixmul = iqk_mul_mat_moe;
        //             } else {
        //                 // Intel Haswell/Broadwell/Skylake (2013-2020)
        //                 // AMD Excavator (2015-2022)
        //                 sgemm = llamafile_sgemm_amd_avx2;
        //                 mixmul = llamafile_mixmul_amd_avx2;
        //                 if (X86_HAVE(F16C))
        //                     iqk_mixmul = iqk_mul_mat_moe;
        //             }
        //         } else {
        //             // AMD Piledriver (2011-2014)
        //             sgemm = llamafile_sgemm_amd_fma;
        //             mixmul = llamafile_mixmul_amd_fma;
        //             if (X86_HAVE(F16C))
        //                 iqk_mixmul = iqk_mul_mat_moe;
        //         }
        //     } else {
        //         // Intel Sandybridge/Ivybridge (2010-2012)
        //         // AMD Bulldozer (2011)
        //         sgemm = llamafile_sgemm_amd_avx;
        //         mixmul = llamafile_mixmul_amd_avx;
        //     }
        // } else {
        //     // AMD K8/Barcelona (2003-2010)
        //     // Intel Core/Nehalem (2006-2009)
        //     sgemm = llamafile_sgemm_unsupported;
        //     mixmul = llamafile_mixmul_unsupported;
        // }


//#elif defined(__aarch64__)
        //long hwcap = getauxval(AT_HWCAP);
        //if ((hwcap & HWCAP_FPHP) &&     // fp16 scalar isa (ID_AA64PFR0_EL1.FP == 1)
        //   (hwcap & HWCAP_ASIMDHP) &&  // fp16 vector isa (ID_AA64PFR0_EL1.AdvSIMD == 1)
        //    (hwcap & HWCAP_ASIMDDP)) {  // dotprod isa (ID_AA64ISAR0_EL1.DP == 1)
            // e.g. Apple M1, Raspberry Pi 5
            sgemm = llamafile_sgemm_arm82;
            mixmul = llamafile_mixmul_arm82;
            iqk_mixmul = iqk_mul_mat_moe_arm82;
        
//#endif
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
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @param precision may be used to control the internal compute type
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(long m, long n, long k, const void* A, long lda, const void* B, long ldb, void* C, long ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype, int precision) {
    return funcs.sgemm(m, n, k, A, lda, B, ldb, C, ldc, ith, nth, task, Atype, Btype, Ctype,
                       precision);
}

/**
 * Performs "mixture of experts" tensor multiplication on CPU.
 */
bool llamafile_mixmul(const ggml_compute_params* params, const ggml_tensor* weights, const ggml_tensor* thought, const ggml_tensor* plan, ggml_tensor* result) {
    return funcs.mixmul(params, weights, thought, plan, result);
}

bool llamafile_mixmul_iqk(long Nx, long Ny, long ne00, int ne11, int typeA, const void* A, const void* B, float* C, long nb1, long nb2, const void* vrow_mapping, int ith, int nth) {
    return funcs.iqk_mixmul(Nx, Ny, ne00, ne11, typeA, A, B, C, nb1, nb2, vrow_mapping, ith, nth);
}
