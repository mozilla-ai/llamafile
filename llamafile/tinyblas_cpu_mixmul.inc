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

#include "tinyblas_cpu.h"

//
//
//                                ██████╗ ██╗   █████╗ ██████╗
//         ██████╗██╗██╗ ██╗██═██╗██╔══██╗██║  ██╔══██╗██╔═══╝
//         ╚═██╔═╝██║███▄██║██ ██║██████╔╝██║  ███████║██████╗
//           ██║  ██║██▀███║╚███╔╝██╔══██╗██║  ██╔══██║╔═══██║
//           ██║  ██║██║ ██║ ███║ ██████╔╝████╗██║  ██║██████║
//           ╚═╝  ╚═╝╚═╝ ╚═╝ ╚══╝ ╚═════╝ ╚═══╝╚═╝  ╚═╝╚═════╝
//
//               MIXTURE OF EXPERTS TENSOR MULTIPLICATION
//
//
// SHAPES
//
//   - weights [cols, rows, experts]
//   - thought [cols, tasks, tokens] w/ tasks ≤ thinkers
//   - result  [rows, thinkers, tokens] w/ thinkers ≤ experts
//   - plan    [thinkers, tokens] w/ i32 < experts
//
// DEFINITION
//
//   for thinker in range(thinkers):
//     for token in range(tokens):
//       for row in range(rows):
//         c = 0
//         for col in range(cols):
//           expert = plan[token][thinker]
//           a = weights[expert][row][col]
//           b = thought[token][thinker % tasks][col]
//           c += a * b
//         result[token][thinker][row] = c
//
// REGULARITIES
//
//   - tokens can be odd
//   - thinkers is usually 2
//   - tasks is usually 1 or 2
//   - cols should be a multiple of 64
//   - rows should be a multiple of 64
//   - experts is usually 8 but could be 60
//   - tokens is always 1 for token generation
//   - tokens can be huge for prompt processing
//
// EXAMPLE
//
//   mixtral 8x7b w/ 217 token prompt
//
//           |  ne*0 ne*1 ne*2 ne*3 | nb*0    nb*1      nb*2       nb*3 | type
//   =========================================================================
//   weights | 16384 6144    8    1 |   18  0x2400 0x3600000 0x1b000000 | q4_0
//   thought | 16384    2  217    1 |    4 0x10000   0x20000  0x1b20000 | f32
//   result  |  6144    2  217    1 |    4  0x6000    0xc000   0xa2c000 | f32
//   plan    |     2  217    1    1 |    4    0x20    0x1b20     0x1b20 | i32
//

namespace {

class MixMul {
  public:
    MixMul(const ggml_compute_params *params, const ggml_tensor *weights,
           const ggml_tensor *thought, const ggml_tensor *plan, ggml_tensor *result)
        : params(params),
          weights(weights),
          thought(thought),
          plan(plan),
          result(result),
          rows(weights->ne[1]),
          cols(weights->ne[0]),
          experts(weights->ne[2]),
          thinkers(plan->ne[0]),
          tasks(thought->ne[1]),
          tokens(thought->ne[2]),
          ldq((cols * 2 + ROW_ALIGN - 1) & -ROW_ALIGN),
          wdata_((char *)(((uintptr_t)params->wdata + MAX_ALIGN - 1) & -MAX_ALIGN)),
          allocated_(0) {
    }

    bool allocate_shared_memory() {
        if (!(quantized_thought_ = allocate<char>(MATRIX_ALIGN, tokens * tasks * ldq)))
            return false;
        if (!(rowptr_result_ = allocate<uintptr_t>(ROW_ALIGN, experts * tokens * thinkers)))
            return false;
        if (!(rowptr_thought_ = allocate<uintptr_t>(ROW_ALIGN, experts * tokens * thinkers)))
            return false;
        if (!(rowptr_count_ = allocate<long>(sizeof(long), experts)))
            return false;
        return true;
    }

    size_t get_allocated_bytes() {
        return (wdata_ - (char *)params->wdata) + allocated_;
    }

    bool mixmul() {

        // invariants
        assert(tasks <= thinkers);
        assert(thinkers <= experts);
        assert(tokens == plan->ne[1]);
        assert(rows == result->ne[0]);
        assert(cols == thought->ne[0]);
        assert(tokens == result->ne[2]);
        assert(thinkers == result->ne[1]);

        // dimensionality
        assert(plan->ne[2] == 1);
        assert(plan->ne[3] == 1);
        assert(result->ne[3] == 1);
        assert(weights->ne[3] == 1);
        assert(thought->ne[3] == 1);

        // miscellaneous
        assert(params->nth > 0);
        assert(params->ith < params->nth);
        assert(plan->type == GGML_TYPE_I32);

        // check nb01 is convertible to lda
        if (weights->nb[1] % ggml_type_size(weights->type))
            return false;

        // no support for column strides
        if (result->nb[0] != ggml_type_size(result->type))
            return false;
        if (thought->nb[0] != ggml_type_size(thought->type))
            return false;
        if (weights->nb[0] != ggml_type_size(weights->type))
            return false;

        // supported output types
        switch (result->type) {
        case GGML_TYPE_F32:
            return mixmuler<float>();
        default:
            return false;
        }
    }

  private:
    template <typename TC>
    bool mixmuler() {
        switch (weights->type) {

        case GGML_TYPE_F32:
            if (thought->type != GGML_TYPE_F32)
                return false;
#if defined(__AVX512F__)
            return mixmat<16, 1, tinyBLAS<NCB | NCC, 16, __m512, __m512, float, float, TC>, float,
                          float, TC>();
#elif defined(__AVX__) || defined(__AVX2__)
            return mixmat<8, 1, tinyBLAS<NCB | NCC, 8, __m256, __m256, float, float, TC>, float,
                          float, TC>();
#elif defined(__SSE__)
            return mixmat<4, 1, tinyBLAS<NCB | NCC, 4, __m128, __m128, float, float, TC>, float,
                          float, TC>();
#elif defined(__ARM_NEON)
            return mixmat<4, 1, tinyBLAS<NCB | NCC, 4, float32x4_t, float32x4_t, float, float, TC>,
                          float, float, TC>();
#else
            return false;
#endif

        case GGML_TYPE_BF16:
            if (thought->type != GGML_TYPE_F32 && thought->type != GGML_TYPE_BF16)
                return false;
#if defined(__AVX512BF16__)
            if (!FLAG_precise) {
                return mixmat<
                    32, 1, tinyBLAS<NCB | NCC, 32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, TC>,
                    ggml_bf16_t, ggml_bf16_t, TC>();
            } else {
                return mixmat<16, 1,
                              tinyBLAS<NCB | NCC, 16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, TC>,
                              ggml_bf16_t, ggml_bf16_t, TC>();
            }
#elif defined(__AVX512F__)
            return mixmat<16, 1,
                          tinyBLAS<NCB | NCC, 16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, TC>,
                          ggml_bf16_t, ggml_bf16_t, TC>();
#elif defined(__AVX2__)
            return mixmat<8, 1,
                          tinyBLAS<NCB | NCC, 8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, TC>,
                          ggml_bf16_t, ggml_bf16_t, TC>();
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
            return mixmat<
                4, 1,
                tinyBLAS<NCB | NCC, 4, float32x4_t, float32x4_t, ggml_bf16_t, ggml_bf16_t, TC>,
                ggml_bf16_t, ggml_bf16_t, TC>();
#else
            return false;
#endif

        case GGML_TYPE_F16:
            if (thought->type != GGML_TYPE_F32 && thought->type != GGML_TYPE_F16)
                return false;
#if defined(__AVX512F__)
            return mixmat<16, 1,
                          tinyBLAS<NCB | NCC, 16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, TC>,
                          ggml_fp16_t, ggml_fp16_t, TC>();
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
            if (X86_CHECK(F16C)) {
                return mixmat<8, 1,
                              tinyBLAS<NCB | NCC, 8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, TC>,
                              ggml_fp16_t, ggml_fp16_t, TC>();
            } else {
                return false;
            }
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
            return mixmat<
                8, 1,
                tinyBLAS<NCB | NCC, 8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, TC>,
                ggml_fp16_t, ggml_fp16_t, TC>();
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
            return mixmat<
                4, 1,
                tinyBLAS<NCB | NCC, 4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, TC>,
                ggml_fp16_t, ggml_fp16_t, TC>();
#else
            return false;
#endif

        case GGML_TYPE_Q4_0:
            if (thought->type != GGML_TYPE_F32 && thought->type != GGML_TYPE_Q8_0)
                return false;
#if defined(__AVX2__) || defined(__AVX512F__)
            return mixmat<32, 32, tinyBLAS_Q0_AVX2<NCB | NCC, block_q4_0, block_q8_0, TC>,
                          block_q4_0, block_q8_0, TC>();
#elif defined(__ARM_FEATURE_DOTPROD)
            return mixmat<32, 32, tinyBLAS_Q0_ARM<NCB | NCC, block_q4_0, block_q8_0, TC>,
                          block_q4_0, block_q8_0, TC>();
#else
            return false;
#endif

        case GGML_TYPE_Q8_0:
            if (thought->type != GGML_TYPE_F32 && thought->type != GGML_TYPE_Q8_0)
                return false;
#if defined(__AVX2__) || defined(__AVX512F__)
            return mixmat<32, 32, tinyBLAS_Q0_AVX2<NCB | NCC, block_q8_0, block_q8_0, TC>,
                          block_q8_0, block_q8_0, TC>();
#elif defined(__ARM_FEATURE_DOTPROD)
            return mixmat<32, 32, tinyBLAS_Q0_ARM<NCB | NCC, block_q8_0, block_q8_0, TC>,
                          block_q8_0, block_q8_0, TC>();
#else
            return false;
#endif

        default:
            return false;
        }
    }

    template <int KN, int BS, typename BLAS, typename TA, typename TB, typename TC>
    bool mixmat() {
        if (cols % KN)
            return false;
        if (thought->type != ggml_type_trait<TB>::id)
            quantize_thought(ggml_type_trait<TB>::id);
        build_row_pointers(ggml_type_trait<TB>::id);
        ggml_barrier(params->threadpool);
        assert(!(cols % BS));
        assert(!(weights->nb[1] % sizeof(TA)));
        // TODO(jart): parallelize this loop
        for (int expert = 0; expert < experts; ++expert) {
            BLAS tb{cols / BS,
                    (const TA *)((const char *)weights->data + expert * weights->nb[2]),
                    (long)(weights->nb[1] / sizeof(TA)),
                    (const TB *)(rowptr_thought_ + expert * tokens * thinkers),
                    0,
                    (TC *)(rowptr_result_ + expert * tokens * thinkers),
                    0,
                    params->ith,
                    params->nth};
            tb.matmul(rows, rowptr_count_[expert]);
        }
        return true;
    }

    void build_row_pointers(ggml_type vec_dot_type) {
        for (int expert = params->ith; expert < experts; expert += params->nth) {
            long count = 0;
            for (long token = 0; token < tokens; ++token)
                for (int thinker = 0; thinker < thinkers; ++thinker)
                    if (expert == *(const int32_t *)((const char *)plan->data +
                                                     token * plan->nb[1] + thinker * plan->nb[0])) {
                        long row = count++;
                        long idx = expert * thinkers * tokens + row;
                        rowptr_result_[idx] =
                            (uintptr_t)((char *)result->data + token * result->nb[2] +
                                        thinker * result->nb[1]);
                        if (thought->type == vec_dot_type)
                            rowptr_thought_[idx] =
                                (uintptr_t)((char *)thought->data + token * thought->nb[2] +
                                            thinker % tasks * thought->nb[1]);
                        else
                            rowptr_thought_[idx] =
                                (uintptr_t)((char *)quantized_thought_ + token * tasks * ldq +
                                            thinker % tasks * ldq);
                    }
            rowptr_count_[expert] = count;
        }
    }

    void quantize_thought(ggml_type vec_dot_type) {
        long chore = 0;
        for (long token = 0; token < tokens; ++token)
            for (int task = 0; task < tasks; ++task)
                if (chore++ % params->nth == params->ith)
                    quantize_row(quantized_thought_ + token * tasks * ldq + task * ldq,
                                 (const float *)((const char *)thought->data +
                                                 token * thought->nb[2] + task * thought->nb[1]),
                                 vec_dot_type);
    }

    void quantize_row(void *dst, const float *src, ggml_type type) {
        assert((long)ggml_row_size(type, cols) <= ldq);
        switch (type) {
        case GGML_TYPE_F16:
            ggml_fp32_to_fp16_row(src, (ggml_fp16_t *)dst, cols);
            break;
        case GGML_TYPE_BF16:
            ggml_fp32_to_bf16_row(src, (ggml_bf16_t *)dst, cols);
            break;
        case GGML_TYPE_Q8_0:
            quantize_row_q8_0_ref((const float *)src, (block_q8_0 *)dst, cols);
            break;
        default:
            GGML_UNREACHABLE();
        }
    }

    template <typename T>
    T *allocate(size_t align, size_t elems) {
        T *res = nullptr;
        size_t need = sizeof(T) * elems;
        size_t base = allocated_;
        base += align - 1;
        base &= -align;
        size_t toto = base + need;
        if (toto >= allocated_ && toto <= params->wsize) {
            res = (T *)(wdata_ + base);
            allocated_ = toto;
        }
        return res;
    }

    const ggml_compute_params *const params;
    const ggml_tensor *const weights;
    const ggml_tensor *const thought;
    const ggml_tensor *const plan;
    ggml_tensor *const result;
    const long rows;
    const long cols;
    const int experts;
    const int thinkers;
    const int tasks;
    const long tokens;
    const long ldq;

    // variables
    char *const wdata_;
    size_t allocated_;

    // shared memory
    long *rowptr_count_ /*[experts]*/;
    char *quantized_thought_ /*[tokens][tasks][cols][2]*/;
    uintptr_t *rowptr_result_ /*[experts][tokens*thinkers]*/;
    uintptr_t *rowptr_thought_ /*[experts][tokens*thinkers]*/;
};

} // namespace

/**
 * Performs "mixture of experts" tensor multiplication on CPU.
 */
bool llamafile_mixmul(const ggml_compute_params *params, const ggml_tensor *weights,
                      const ggml_tensor *thought, const ggml_tensor *plan, ggml_tensor *result) {
    MixMul mm{params, weights, thought, plan, result};
    return mm.allocate_shared_memory() && mm.mixmul();
}
