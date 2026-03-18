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

#include "chatbot.h"

#include <cosmo.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits.h>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include "color.h"
#include "compute.h"
#include "string.h"
#include <cpp-httplib/httplib.h>
#include "llamafile.h"

#include "version.h"

namespace lf {
namespace chatbot {

// Global state
common_params *g_params = nullptr;      // pointer to params
common_sampler *g_sampler = nullptr;    // sampler context
mtmd_context *g_mtmd = nullptr;         // multimodal context
llama_model *g_model = nullptr;
llama_context *g_ctx = nullptr;
common_chat_templates_ptr g_chat_templates;  // chat template handler
common_chat_parser_params g_chat_syntax;            // chat syntax for parsing
std::string g_pending_file_content;                 // accumulated /upload content awaiting user message

// Static storage for params
static common_params s_params;

// Track whether we own the model (for cleanup)
static bool g_owns_model = true;

std::string describe_compute(void) {
    // Check if using GPU based on params
    // n_gpu_layers > 0 means explicitly enabled, < 0 means "auto" (use GPU if available)
    if (g_params && g_params->n_gpu_layers != 0 && llamafile_has_gpu()) {
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
    // API mode: no local model, assume chat model
    if (!g_model)
        return false;

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

    // FLAG_verbose is set by parse_llamafile_args() in args.cpp
    bool verbose = FLAG_verbose;

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
    // Pause common_log BEFORE common_init() to suppress llama.cpp build info line
    if (!verbose)
        common_log_pause(common_log_main());
    common_init();
    if (!verbose)
        common_log_resume(common_log_main());

    // NOTE that we are currently using llama.cpp flags parser here, so
    // either we create a new kind of example for a custom set of flags
    // or we need to deal with them separately and remove them prior to
    // this step (see removeArgs in main.cpp)
    if (!common_params_parse(argc, argv, *g_params, LLAMA_EXAMPLE_CLI)) {
        fprintf(stderr, "error: failed to parse flags\n");
        exit(1);
    }

    if (llamafile_has_metal() && g_params->n_gpu_layers < 0) {
        // if Metal and no ngl was specified, default to INT_MAX
        g_params->n_gpu_layers = INT_MAX;
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

    // Print info (format line is added later after template detection)
    if (!FLAG_nologo) {
        printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n"
               BOLD "model" UNBOLD ":    %s\n",
               basename(g_params->model.path).c_str());
        if (is_base_model())
            printf(BOLD "mode" UNBOLD ":     RAW TEXT COMPLETION (base model)\n");
        printf(BOLD "compute" UNBOLD ":  %s\n", describe_compute().c_str());
    }

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = common_context_params_to_llama(*g_params);
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
        mparams.flash_attn_type = g_params->flash_attn_type;
        mparams.warmup = g_params->warmup;
        mparams.image_min_tokens = g_params->image_min_tokens;
        mparams.image_max_tokens = g_params->image_max_tokens;
        g_mtmd = mtmd_init_from_file(g_params->mmproj.path.c_str(), g_model, mparams);
        clear_ephemeral();
        if (!g_mtmd) {
            fprintf(stderr, "%s: failed to initialize multimodal model%s\n",
                    g_params->mmproj.path.c_str(), tip());
            exit(5);
        }
    }

    // Initialize chat templates for output parsing (e.g., gpt-oss think mode)
    // Use the same approach as common_chat_verify_template() - provide a dummy message
    if (!is_base_model()) {
        g_chat_templates = common_chat_templates_init(g_model, g_params->chat_template);
        if (g_chat_templates) {
            // Provide a minimal dummy message (same approach as common_chat_verify_template)
            common_chat_msg dummy_msg;
            dummy_msg.role = "user";
            dummy_msg.content = "test";

            // Check if the template supports enable_thinking (like llama.cpp server does).
            // This is needed for models like Qwen3.5 that check enable_thinking in their
            // template - without this, the template outputs a closed thinking block.
            bool supports_thinking = common_chat_templates_support_enable_thinking(g_chat_templates.get());

            common_chat_templates_inputs inputs;
            inputs.messages = {dummy_msg};
            inputs.use_jinja = true;
            inputs.enable_thinking = supports_thinking;
            // CRITICAL: Set reasoning_format BEFORE applying templates. The PEG parser
            // is built during common_chat_templates_apply() and checks this value to
            // decide whether to include reasoning extraction in the grammar.
            inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

            try {
                auto chat_params = common_chat_templates_apply(g_chat_templates.get(), inputs);
                g_chat_syntax.format = chat_params.format;
                g_chat_syntax.thinking_forced_open = chat_params.thinking_forced_open;

                // Load the PEG parser if one was provided
                if (!chat_params.parser.empty()) {
                    g_chat_syntax.parser.load(chat_params.parser);
                }

                // Copy reasoning format to chat syntax for use by the parser at runtime
                g_chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                g_chat_syntax.reasoning_in_content = false;

                // Print detected format
                if (!FLAG_nologo && g_chat_syntax.format != COMMON_CHAT_FORMAT_CONTENT_ONLY) {
                    printf(BOLD "format" UNBOLD ":   %s\n", common_chat_format_name(g_chat_syntax.format));
                }
            } catch (const std::exception &e) {
                // Template application failed, fall back to content-only parsing
                LOG_DBG("chat template application failed: %s\n", e.what());
            }
        }
    }

    // Ensure there's a blank line after info block
    if (!FLAG_nologo) {
        printf("\n");
    }

    // Create direct backend and run the REPL
    auto backend = create_direct_backend();
    g_backend = backend.get();

    // Direct-backend-specific init: evaluate BOS token and system prompt
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    if (llama_vocab_get_add_bos(vocab)) {
        print_ephemeral("loading bos token...");
        eval_token(llama_vocab_bos(vocab));
    }
    record_undo();

    // Make base models have no system prompt by default
    if (is_base_model() && g_params->prompt == DEFAULT_SYSTEM_PROMPT)
        g_params->prompt = "";

    // For base models, evaluate system prompt directly (no template)
    if (!g_params->prompt.empty() && is_base_model()) {
        print_ephemeral("loading system prompt...");
        std::string msg = g_params->prompt;
        if (!eval_string(msg, DONT_ADD_SPECIAL, PARSE_SPECIAL))
            exit(6);
        llama_synchronize(g_ctx);
        clear_ephemeral();
    }

    repl(*backend);

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

    // Only free the model if we own it
    if (g_owns_model) {
        print_ephemeral("freeing model...");
        llama_model_free(g_model);
        clear_ephemeral();

        print_ephemeral("freeing backend...");
        llama_backend_free();
        clear_ephemeral();
    }

    return 0;
}

// API client entry point for combined mode.
// Runs TUI chatbot that communicates with the server via HTTP.
int api_main(const std::string &server_url, const std::string &system_prompt,
             const std::string &model_path, std::function<void()> shutdown_fn) {
    signal(SIGPIPE, SIG_IGN);

    // Initialize minimal params
    g_params = &s_params;
    g_params->prompt = system_prompt.empty() ? DEFAULT_SYSTEM_PROMPT : system_prompt;

    // Print logo and info
    char *fake_argv[] = {const_cast<char*>("llamafile"), nullptr};
    if (!FLAG_nologo) {
        logo(fake_argv);
        printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n"
               BOLD "model" UNBOLD ":    %s\n"
               BOLD "compute" UNBOLD ":  %s\n"
               BOLD "server" UNBOLD ":   %s\n",
               basename(model_path).c_str(),
               describe_compute().c_str(),
               server_url.c_str());
        printf("\n");
    }

    // Create API backend
    auto backend = create_api_backend(server_url);
    g_backend = backend.get();

    // Run REPL
    repl(*backend);

    // Signal the server to shut down when the TUI exits
    if (shutdown_fn) {
        shutdown_fn();
    }

    return 0;
}

} // namespace chatbot
} // namespace lf
