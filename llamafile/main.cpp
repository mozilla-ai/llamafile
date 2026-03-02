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

//
// llamafile - Main entry point
//
// This is the main entry point for llamafile. It provides multiple execution
// modes for interacting with LLMs:
//
// Usage:
//   llamafile -m model.gguf              # Combined: TUI chat + HTTP server
//   llamafile -m model.gguf --chat       # TUI chat only
//   llamafile -m model.gguf --server     # HTTP server only
//   llamafile -m model.gguf --cli -p "prompt"  # Single prompt -> response
//

// Server context for combined mode
// NOTE: These must be included BEFORE cosmo.h because cosmo.h defines a
// 'defer' macro that conflicts with llama.cpp's defer() function
#include "common.h"
#include "arg.h"
#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-http.h"
#include "server-common.h"

#include "args.h"
#include "chatbot.h"
#include "llamafile.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <thread>
#include <exception>

#ifdef COSMOCC
#include <cosmo.h>
#endif

// Forward declarations
extern int server_main(int argc, char **argv);

namespace lf {

// Exception wrapper for HTTP handlers (same as in server.cpp)
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
        } catch (...) {
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// Combined mode: run server and chatbot together, sharing the model
static int combined_main(int argc, char **argv) {
    common_params params;

    // Parse parameters (using SERVER example type for full server options)
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        fprintf(stderr, "error: failed to parse arguments\n");
        return 1;
    }

    // Handle n_parallel auto-selection (server sets -1 for "auto")
    // Also limit to a reasonable value for combined mode since:
    // 1. Server defaults are too high for combined chatbot + server use
    // 2. Some models (ISWA) have a hard limit of n_seq_max <= 256
    if (params.n_parallel < 0 || params.n_parallel > 256) {
        params.n_parallel = 16;  // Reasonable default for combined mode
        params.kv_unified = true;
    }

    // Suppress server/model loading logs unless --verbose was specified
    // This gives combined mode a clean UX similar to --chat mode
    if (!FLAG_verbose) {
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
        mtmd_helper_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
    }

    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Create server context
    server_context ctx_server;

    // Initialize HTTP server
    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        fprintf(stderr, "error: failed to initialize HTTP server\n");
        llama_backend_free();
        return 1;
    }

    // Register API routes
    server_routes routes(params, ctx_server);

    ctx_http.get ("/health",              ex_wrapper(routes.get_health));
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health));
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));
    ctx_http.post("/props",               ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",            ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",              ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models));
    ctx_http.get ("/api/tags",            ex_wrapper(routes.get_models));
    ctx_http.post("/completion",          ex_wrapper(routes.post_completions));
    ctx_http.post("/completions",         ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",            ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/responses",        ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/messages",         ex_wrapper(routes.post_anthropic_messages));
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens));
    ctx_http.post("/infill",              ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",           ex_wrapper(routes.post_embeddings));
    ctx_http.post("/embeddings",          ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",              ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",        ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",            ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",          ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",      ex_wrapper(routes.post_apply_template));
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));

    // Start HTTP server before loading model (to serve /health during loading)
    if (!ctx_http.start()) {
        fprintf(stderr, "error: failed to start HTTP server\n");
        llama_backend_free();
        return 1;
    }

    // Load model
    if (!ctx_server.load_model(params)) {
        fprintf(stderr, "error: failed to load model\n");
        ctx_http.stop();
        llama_backend_free();
        return 1;
    }

    // Update routes metadata now that model is loaded
    routes.update_meta(ctx_server);

    // Mark server as ready
    ctx_http.is_ready.store(true);

    if (FLAG_verbose) {
        LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
    }

    // Get the shared model pointer from server's context
    llama_context *server_ctx = ctx_server.get_llama_context();
    if (!server_ctx) {
        fprintf(stderr, "error: server context not initialized\n");
        ctx_http.stop();
        llama_backend_free();
        return 1;
    }
    llama_model *shared_model = const_cast<llama_model *>(llama_get_model(server_ctx));

    // Start server inference loop in background thread
    std::thread server_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    // Run chatbot in main thread with shared model
    int result = chatbot::main_with_model(shared_model, params);

    // Cleanup: terminate server and wait for thread
    ctx_http.stop();
    ctx_server.terminate();
    server_thread.join();

    llama_backend_free();
    return result;
}

} // namespace lf

int main(int argc, char **argv) {
    // Load arguments from zip file if present (for bundled llamafiles)
#ifdef COSMOCC
    argc = cosmo_args("/zip/.args", &argv);
#endif

    // Parse llamafile arguments and determine execution mode
    // This also handles GPU initialization via llamafile_early_gpu_init()
    lf::LlamafileArgs args = lf::parse_llamafile_args(argc, argv);

    // Suppress GPU and backend logging unless --verbose was specified
    // This must happen BEFORE llamafile_has_gpu() which triggers Metal/CUDA init
    if (!FLAG_verbose) {
        llamafile_metal_log_set(llamafile_log_callback_null, NULL);
        llamafile_cuda_log_set(llamafile_log_callback_null, NULL);
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
    }

    // For CLI mode, suppress logo (but respect --verbose if user specified it)
    if (args.mode == lf::ProgramMode::CLI) {
        FLAG_nologo = 1;
    }

    // Initialize GPU support (triggers dynamic loading of GPU backends)
    llamafile_has_gpu();

    // Route to appropriate mode
    switch (args.mode) {
        case lf::ProgramMode::SERVER:
            // Server only mode
            return server_main(args.llama_argc, args.llama_argv);

        case lf::ProgramMode::CHAT:
            // Chat only mode (no server)
            return lf::chatbot::main(args.llama_argc, args.llama_argv);

        case lf::ProgramMode::CLI:
            // Single prompt -> response mode
            return lf::chatbot::cli_main(args.llama_argc, args.llama_argv);

        case lf::ProgramMode::AUTO:
            // Combined mode: chat + server sharing model
            return lf::combined_main(args.llama_argc, args.llama_argv);
    }

    return 1;
}
