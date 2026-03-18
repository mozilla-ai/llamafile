// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
// Copyright 2026 Mozilla.ai
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

//
// TinyBLAS compatibility layer for cuBLAS
//
// This header provides macro definitions to map cuBLAS API calls to TinyBLAS
// equivalents, allowing the ggml CUDA backend to use TinyBLAS instead of cuBLAS.
//
// TinyBLAS is a lightweight BLAS implementation created by Mozilla Foundation
// that provides API-compatible replacements for cuBLAS GEMM functions with:
// - Smaller binary size (no cuBLAS dependency)
// - Better portability across CUDA versions
// - Works on systems without cuBLAS installed
//

#pragma once

#ifdef GGML_USE_TINYBLAS

#include "tinyblas.h"

// ============================================================================
// Type mappings
// ============================================================================

#define cublasHandle_t            tinyblasHandle_t
#define cublasStatus_t            tinyblasStatus_t
#define cublasComputeType_t       tinyblasComputeType_t
#define cublasOperation_t         tinyblasOperation_t
#define cublasGemmAlgo_t          tinyblasGemmAlgo_t

// Note: cudaDataType_t is a CUDA runtime type, not cuBLAS
// We map it to tinyblasDataType_t when used with BLAS functions
#define cudaDataType_t            tinyblasDataType_t

// ============================================================================
// Function mappings
// ============================================================================

#define cublasCreate              tinyblasCreate
#define cublasDestroy             tinyblasDestroy
#define cublasSetStream           tinyblasSetStream
#define cublasGetStream           tinyblasGetStream
#define cublasSgemm               tinyblasSgemm
#define cublasGemmEx              tinyblasGemmEx
#define cublasGemmBatchedEx       tinyblasGemmBatchedEx
#define cublasGemmStridedBatchedEx tinyblasGemmStridedBatchedEx
#define cublasGetStatusString     tinyblasGetStatusString

// ============================================================================
// Operation constant mappings
// ============================================================================

#define CUBLAS_OP_N               TINYBLAS_OP_N
#define CUBLAS_OP_T               TINYBLAS_OP_T

// ============================================================================
// Status constant mappings
// ============================================================================

#define CUBLAS_STATUS_SUCCESS     TINYBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED  TINYBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ALLOC_FAILED     TINYBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE    TINYBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH    TINYBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_MAPPING_ERROR    TINYBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_EXECUTION_FAILED TINYBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR   TINYBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_NOT_SUPPORTED    TINYBLAS_STATUS_NOT_SUPPORTED

// ============================================================================
// Compute type constant mappings
// ============================================================================

#define CUBLAS_COMPUTE_16F        TINYBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F        TINYBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F TINYBLAS_COMPUTE_32F

// ============================================================================
// Data type constant mappings
// ============================================================================

#define CUDA_R_16F                TINYBLAS_R_16F
#define CUDA_R_32F                TINYBLAS_R_32F

// WARNING: BF16 (bfloat16) is NOT supported by TinyBLAS.
// This mapping to FP16 is INCORRECT and will produce garbage/NaN values because
// BF16 and FP16 have incompatible bit layouts:
// - BF16: 1 sign + 8 exponent + 7 mantissa (same exponent range as FP32)
// - FP16: 1 sign + 5 exponent + 10 mantissa (smaller range, more precision)
// Interpreting BF16 bits as FP16 causes exponent bit misalignment resulting in
// completely wrong values (often infinity or NaN).
//
// The CUDA backend (ggml-cuda.cu and common.cuh) should check GGML_USE_TINYBLAS
// and disable BF16 code paths, falling back to FP16 or FP32 conversion instead.
// If this mapping is ever reached, it indicates a bug in the fallback logic.
#define CUDA_R_16BF               TINYBLAS_R_16F

// ============================================================================
// GEMM algorithm constant mappings
// ============================================================================

#define CUBLAS_GEMM_DEFAULT       TINYBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP TINYBLAS_GEMM_DEFAULT

// ============================================================================
// Math mode - TinyBLAS manages precision internally, so these are no-ops
// ============================================================================

#define cublasSetMathMode(handle, mode) TINYBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUBLAS_DEFAULT_MATH 0

// ============================================================================
// Triangular solve (TRSM) - Not supported in TinyBLAS
//
// cublasStrsmBatched is used only by solve_tri.cu for Qwen3-Next model's
// gated delta net attention. When TinyBLAS is used, the CUDA backend will
// fall back to CPU for this operation.
// ============================================================================

#define GGML_TINYBLAS_NO_TRSM 1

// Define TRSM-related constants so code compiles, but operations will fail
#define CUBLAS_SIDE_LEFT          0
#define CUBLAS_SIDE_RIGHT         1
#define CUBLAS_FILL_MODE_LOWER    0
#define CUBLAS_FILL_MODE_UPPER    1
#define CUBLAS_DIAG_NON_UNIT      0
#define CUBLAS_DIAG_UNIT          1

// Stub for cublasStrsmBatched - always returns not supported
// The solve_tri.cu code should check GGML_TINYBLAS_NO_TRSM and fall back to CPU
#define cublasStrsmBatched(...) TINYBLAS_STATUS_NOT_SUPPORTED

#endif // GGML_USE_TINYBLAS
