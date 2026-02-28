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

#include "args.h"
#include "chatbot.h"
#include "llamafile.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <thread>

#ifdef COSMOCC
#include <cosmo.h>
#endif

// Forward declarations
extern int server_main(int argc, char **argv);

namespace lf {

// Combined mode: run server and chatbot together, sharing the model
static int combined_main(int argc, char **argv, bool verbose) {
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
    if (!verbose) {
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);
        mtmd_helper_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
    }

    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Create server context and load model
    server_context ctx_server;

    if (!ctx_server.load_model(params)) {
        fprintf(stderr, "error: failed to load model\n");
        llama_backend_free();
        return 1;
    }

    // Get the shared model pointer from server's context
    llama_context *server_ctx = ctx_server.get_llama_context();
    if (!server_ctx) {
        fprintf(stderr, "error: server context not initialized\n");
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

    // Suppress GPU logging unless --verbose was specified
    // This must happen BEFORE llamafile_has_gpu() which triggers Metal/CUDA init
    if (!args.verbose) {
        llamafile_metal_log_set(llamafile_log_callback_null, NULL);
        llamafile_cuda_log_set(llamafile_log_callback_null, NULL);
    }

    // For CLI mode, also suppress logo
    if (args.mode == lf::ProgramMode::CLI) {
        FLAG_verbose = 0;
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
            return lf::combined_main(args.llama_argc, args.llama_argv, args.verbose);
    }

    return 1;
}
