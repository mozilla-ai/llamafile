// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

typedef enum tinyblasOperation {
    TINYBLAS_OP_N,
    TINYBLAS_OP_T,
} tinyblasOperation_t;

typedef enum tinyblasDataType {
    TINYBLAS_R_32F,
    TINYBLAS_R_16F,
} tinyblasDataType_t;

typedef enum tinyblasComputeType {
    TINYBLAS_COMPUTE_32F,
    TINYBLAS_COMPUTE_16F,
} tinyblasComputeType_t;

typedef enum tinyblasGemmAlgo {
    TINYBLAS_GEMM_DEFAULT,
} tinyblasGemmAlgo_t;

typedef enum tinyblasStatus {
    TINYBLAS_STATUS_SUCCESS,
    TINYBLAS_STATUS_ALLOC_FAILED,
    TINYBLAS_STATUS_INVALID_VALUE,
    TINYBLAS_STATUS_NOT_SUPPORTED,
    TINYBLAS_STATUS_EXECUTION_FAILED,
    TINYBLAS_STATUS_DIMENSION_OVERLAP,
    TINYBLAS_STATUS_DIMENSION_OVERFLOW,
} tinyblasStatus_t;

struct tinyblasContext;
typedef struct tinyblasContext *tinyblasHandle_t;

const char *tinyblasGetStatusString(tinyblasStatus_t);

tinyblasStatus_t tinyblasCreate(tinyblasHandle_t *);
tinyblasStatus_t tinyblasDestroy(tinyblasHandle_t);
tinyblasStatus_t tinyblasSetStream(tinyblasHandle_t, void *);
tinyblasStatus_t tinyblasGetStream(tinyblasHandle_t, void **);

tinyblasStatus_t tinyblasSgemm(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int, int,
                               int, const float *, const float *, int, const float *, int,
                               const float *, float *, int);

tinyblasStatus_t tinyblasGemmEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t, int,
                                int, int, const void *, const void *, tinyblasDataType_t, int,
                                const void *, tinyblasDataType_t, int, const void *, void *,
                                tinyblasDataType_t, int, tinyblasComputeType_t, tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmBatchedEx(tinyblasHandle_t, tinyblasOperation_t, tinyblasOperation_t,
                                       int, int, int, const void *, const void *const[],
                                       tinyblasDataType_t, int, const void *const[],
                                       tinyblasDataType_t, int, const void *, void *const[],
                                       tinyblasDataType_t, int, int, tinyblasComputeType_t,
                                       tinyblasGemmAlgo_t);

tinyblasStatus_t tinyblasGemmStridedBatchedEx(tinyblasHandle_t, tinyblasOperation_t,
                                              tinyblasOperation_t, int, int, int, const void *,
                                              const void *, tinyblasDataType_t, int, long long,
                                              const void *, tinyblasDataType_t, int, long long,
                                              const void *, void *, tinyblasDataType_t, int,
                                              long long, int, tinyblasComputeType_t,
                                              tinyblasGemmAlgo_t);
