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

#include "chatbot.h"

#include <cosmo.h>
#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

#include "color.h"
#include "compute.h"
#include "string.h"
#include "llamafile.h"

// Version string - should be defined by build system
#ifndef LLAMAFILE_VERSION_STRING
#define LLAMAFILE_VERSION_STRING "0.10.0-dev"
#endif

namespace lf {
namespace chatbot {

// Global state
common_params *g_params = nullptr;      // pointer to params
common_sampler *g_sampler = nullptr;    // sampler context
mtmd_context *g_mtmd = nullptr;         // multimodal context
llama_model *g_model = nullptr;
llama_context *g_ctx = nullptr;

// Static storage for params
static common_params s_params;

std::string describe_compute(void) {
    // Check if using GPU based on params
    if (g_params && g_params->n_gpu_layers > 0 && llamafile_has_gpu()) {
        if (llamafile_has_metal()) {
            return "Apple Metal GPU";
        } else {
            // Try to get CUDA device info if available
            return llamafile_describe_cpu() + " (with GPU acceleration)";
        }
    } else {
        return llamafile_describe_cpu();
    }
}

std::string token_to_piece(const struct llama_context *ctx, llama_token token, bool special) {
    if (token == IMAGE_PLACEHOLDER_TOKEN)
        return "⁑";
    return llamafile_token_to_piece(ctx, token, special);
}

const char *tip() {
    if (g_params->verbosity)
        return "";
    return " (use the --verbose flag for further details)";
}

bool is_base_model() {
    // check if user explicitly passed --chat-template flag
    if (!g_params->chat_template.empty())
        return false;

    // check if gguf metadata has chat template. this should always be
    // present for "instruct" models, and never specified on base ones
    return llama_model_meta_val_str(g_model, "tokenizer.chat_template", 0, 0) == -1;
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

    // print logo
    logo(argv);

    // Check if verbose mode is requested (must be set before Metal init)
    bool verbose = llamafile_has(argv, "--verbose");
    FLAG_verbose = verbose ? 1 : 0;

    // Initialize params with defaults
    g_params = &s_params;
    g_params->sampling.n_prev = 64;
    g_params->n_batch = 256;  // for better progress indication
    g_params->sampling.temp = 0;  // don't use randomness by default
    g_params->prompt = DEFAULT_SYSTEM_PROMPT;

    // Initialize GPU support (must happen BEFORE llama_backend_init())
    // This triggers dynamic compilation and loading of GPU backends
    print_ephemeral("initializing gpu...");
    if (!verbose) {
        // disable ggml verbose logging
        if (llamafile_has_metal()) {
            llamafile_metal_log_set(llamafile_log_callback_null, NULL);
        } else if (llamafile_has_cuda() || llamafile_has_amd_gpu()) {
            llamafile_cuda_log_set(llamafile_log_callback_null, NULL);
        }
    } else {
        clear_ephemeral();
    }

    // parse flags
    print_ephemeral("loading backend...");
    llama_backend_init();
    common_init();

    // NOTE that we are currently using llama.cpp flags parser here, so
    // either we create a new kind of example for a custom set of flags
    // or we need to deal with them separately and remove them prior to
    // this step (see removeArgs in main.cpp)
    if (!common_params_parse(argc, argv, *g_params, LLAMA_EXAMPLE_CLI)) {
        fprintf(stderr, "error: failed to parse flags\n");
        exit(1);
    }

    clear_ephemeral();

    // Suppress logging for model loading unless --verbose was specified
    // We must set this AFTER common_init() since it overwrites the log callback
    // and BEFORE model loading to suppress those logs
    if (!verbose) {
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
        // Also suppress LOG_INF() and LOG_WRN() messages from common_log (used by LLM loader)
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
        // Suppress mtmd/CLIP and mtmd-helper logging
        mtmd_helper_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
    }

    print_ephemeral("loading model...");
    llama_model_params model_params = common_model_params_to_llama(*g_params);
    llama_context_params ctx_params = common_context_params_to_llama(*g_params);

    // Save original values to detect changes
    const int32_t orig_n_gpu_layers = model_params.n_gpu_layers;
    const uint32_t orig_n_ctx = ctx_params.n_ctx;

    // Use llama_params_fit to adjust parameters to available GPU memory
    // This prevents Metal memory corruption when models exceed available VRAM
    if (g_params->fit_params && llamafile_has_gpu()) {
        llama_params_fit_status status = llama_params_fit(
            g_params->model.path.c_str(),
            &model_params,
            &ctx_params,
            g_params->tensor_split,
            g_params->tensor_buft_overrides.data(),
            g_params->fit_params_target,
            g_params->fit_params_min_ctx,
            verbose ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_ERROR
        );

        if (status == LLAMA_PARAMS_FIT_STATUS_FAILURE) {
            fprintf(stderr, "warning: could not fit model to GPU memory, using default parameters\n");
        } else if (status == LLAMA_PARAMS_FIT_STATUS_ERROR) {
            fprintf(stderr, "warning: error during memory fitting, using default parameters\n");
        } else if (status == LLAMA_PARAMS_FIT_STATUS_SUCCESS) {
            // Notify user of any adjustments made to fit GPU memory
            bool adjusted = false;
            if (model_params.n_gpu_layers != orig_n_gpu_layers) {
                fprintf(stderr, "note: GPU layers adjusted to %d to fit in GPU memory",
                        model_params.n_gpu_layers);
                adjusted = true;
            }
            if (ctx_params.n_ctx != orig_n_ctx && orig_n_ctx != 0) {
                if (adjusted) {
                    fprintf(stderr, ", context size reduced to %u", ctx_params.n_ctx);
                } else {
                    fprintf(stderr, "note: context size reduced to %u to fit in GPU memory",
                            ctx_params.n_ctx);
                    adjusted = true;
                }
            }
            if (adjusted) {
                fprintf(stderr, " (use --fit off to disable)\n");
            }
        }
    } else if (llamafile_has_metal() && g_params->n_gpu_layers < 0) {
        // If fit_params is disabled but Metal is available and no ngl was specified,
        // default to INT_MAX (original behavior)
        model_params.n_gpu_layers = INT_MAX;
    }

    g_model = llama_model_load_from_file(g_params->model.path.c_str(), model_params);
    clear_ephemeral();
    if (g_model == NULL) {
        fprintf(stderr, "%s: failed to load model%s\n", g_params->model.path.c_str(), tip());
        exit(2);
    }

    // Adjust context size
    if (g_params->n_ctx <= 0 || g_params->n_ctx > (int)llama_model_n_ctx_train(g_model))
        g_params->n_ctx = llama_model_n_ctx_train(g_model);
    if (g_params->n_ctx < g_params->n_batch)
        g_params->n_batch = g_params->n_ctx;

    // Print info
    if (!FLAG_nologo) {
        printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n"
               BOLD "model" UNBOLD ":    %s\n",
               basename(g_params->model.path).c_str());
        if (is_base_model())
            printf(BOLD "mode" UNBOLD ":     RAW TEXT COMPLETION (base model)\n");
        printf(BOLD "compute" UNBOLD ":  %s\n", describe_compute().c_str());
        printf("\n");
    }

    print_ephemeral("initializing context...");
    // Update ctx_params with adjusted n_ctx if not using fit_params
    // (fit_params already handles context size adjustments)
    if (!g_params->fit_params || !llamafile_has_gpu()) {
        ctx_params = common_context_params_to_llama(*g_params);
    }
    g_ctx = llama_init_from_model(g_model, ctx_params);
    clear_ephemeral();
    if (!g_ctx) {
        fprintf(stderr, "error: failed to initialize context%s\n", tip());
        exit(3);
    }

    if (llama_model_has_encoder(g_model))
        fprintf(stderr, "warning: this model has an encoder\n");

    // Initialize sampler
    g_sampler = common_sampler_init(g_model, g_params->sampling);
    if (!g_sampler) {
        fprintf(stderr, "error: failed to initialize sampler\n");
        exit(4);
    }

    // Initialize multimodal if mmproj is specified
    if (!g_params->mmproj.path.empty()) {
        print_ephemeral("initializing vision model...");
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = g_params->mmproj_use_gpu;
        mparams.n_threads = g_params->cpuparams.n_threads;
        mparams.print_timings = g_params->verbosity > 0;
        g_mtmd = mtmd_init_from_file(g_params->mmproj.path.c_str(), g_model, mparams);
        clear_ephemeral();
        if (!g_mtmd) {
            fprintf(stderr, "%s: failed to initialize multimodal model%s\n",
                    g_params->mmproj.path.c_str(), tip());
            exit(5);
        }
    }

    // Run the REPL
    repl();

    // Synchronize before cleanup to ensure all GPU operations complete
    if (g_ctx) {
        llama_synchronize(g_ctx);
    }

    // Cleanup
    if (g_mtmd) {
        print_ephemeral("freeing vision model...");
        mtmd_free(g_mtmd);
        clear_ephemeral();
    }

    if (g_sampler) {
        common_sampler_free(g_sampler);
    }

    // If interrupted, directly exit to avoid Metal backend crash on exit
    // (NOTE: the issue occurs when llama_free(g_ctx) is run)
    if (g_interrupted_exit) {
        _exit(0);
    }

    print_ephemeral("freeing context...");
    llama_free(g_ctx);
    clear_ephemeral();

    print_ephemeral("freeing model...");
    llama_model_free(g_model);
    clear_ephemeral();

    print_ephemeral("freeing backend...");
    llama_backend_free();
    clear_ephemeral();

    return 0;
}

} // namespace chatbot
} // namespace lf
