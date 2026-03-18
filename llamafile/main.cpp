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

#include "args.h"
#include "chatbot.h"
#include "llamafile.h"
#include "version.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <vector>
#include <condition_variable>

#ifdef COSMOCC
#include <cosmo.h>
#endif

// Forward declarations
extern int server_main(int argc, char **argv,
                       std::function<void(const std::string &)> on_ready,
                       std::function<void(std::function<void()>)> on_shutdown_available);

namespace lf {

// Context passed to the TUI thread via pthread
struct TuiThreadCtx {
    std::function<void()> *shutdown_fn;
    std::mutex *mu;
    std::condition_variable *cv;
    bool *shutdown_ready;
    std::string listen_addr;
    std::string system_prompt;
    std::string model_path;
};

static void *tui_thread_fn(void *arg) {
    auto *ctx = static_cast<TuiThreadCtx *>(arg);

    // Wait for shutdown function to be available
    {
        std::unique_lock<std::mutex> lock(*ctx->mu);
        ctx->cv->wait(lock, [&] { return *ctx->shutdown_ready; });
    }

    chatbot::api_main(ctx->listen_addr, ctx->system_prompt, ctx->model_path, *ctx->shutdown_fn);
    delete ctx;
    return nullptr;
}

// Combined mode: server on main thread (owns GPU/CUDA), TUI on background thread (HTTP client)
static int combined_main(const LlamafileArgs &args) {
    std::function<void()> shutdown_fn;
    pthread_t tui_tid = 0;
    std::mutex mu;
    std::condition_variable cv;
    bool shutdown_ready = false;

    // Called when server is fully loaded and ready to accept requests
    auto on_ready = [&](const std::string &listen_addr) {
        // Start TUI chatbot on background thread as HTTP client
        // Use pthread with explicit 8 MiB stack to avoid stack overflow
        // in nlohmann/json's recursive parser (default Cosmopolitan thread
        // stack is too small for the httplib + SSE + JSON parsing call chain)
        auto *ctx = new TuiThreadCtx{
            &shutdown_fn, &mu, &cv, &shutdown_ready,
            listen_addr, args.system_prompt, args.model_path
        };

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
        pthread_create(&tui_tid, &attr, tui_thread_fn, ctx);
        pthread_attr_destroy(&attr);
    };

    // Called when server's shutdown mechanism is available
    auto on_shutdown = [&](std::function<void()> fn) {
        std::lock_guard<std::mutex> lock(mu);
        shutdown_fn = std::move(fn);
        shutdown_ready = true;
        cv.notify_one();
    };

    // Run server on main thread (blocks until terminated)
    // This ensures all CUDA/GPU operations happen on the OS main thread
    int rc = server_main(args.llama_argc, args.llama_argv, on_ready, on_shutdown);

    if (tui_tid)
        pthread_join(tui_tid, nullptr);

    return rc;
}

} // namespace lf

int main(int argc, char **argv) {
    // Load arguments from zip file if present (for bundled llamafiles)
#ifdef COSMOCC
    argc = cosmo_args("/zip/.args", &argv);
#endif

    // Handle --version before anything else (ignores all other arguments)
    if (llamafile_has(argv, "--version")) {
        puts("llamafile v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    // Parse llamafile arguments and determine execution mode
    // This also handles GPU initialization via llamafile_early_gpu_init()
    lf::LlamafileArgs args = lf::parse_llamafile_args(argc, argv);

    // All modes require a model file (but let --help/-h pass through).
    if (args.model_path.empty() &&
        !llamafile_has(argv, "--help") && !llamafile_has(argv, "-h")) {
        fprintf(stderr, "usage: llamafile -m MODEL.gguf [options]\n"
                        "\n"
                        "modes:\n"
                        "  (default)   combined TUI chat + HTTP server\n"
                        "  --server    HTTP server only\n"
                        "  --chat      TUI chat only\n"
                        "  --cli       single prompt/response (requires -p)\n"
                        "\n"
                        "options:\n"
                        "  -m FILE     path to GGUF model file (required)\n"
                        "  -p TEXT     system prompt\n"
                        "  --gpu MODE  GPU mode (auto, nvidia, amd, apple, disable)\n"
                        "  -ngl N      number of layers to offload to GPU\n"
                        "  --verbose   enable verbose logging\n"
                        "  --version   show version information\n"
                        "\n"
                        "example:\n"
                        "  llamafile -m model.gguf\n"
                        "  llamafile -m model.gguf --server --port 8080\n"
                        "  llamafile -m model.gguf --cli -p \"explain quantum computing\"\n"
                        "\n"
                        "run llamafile --help for full documentation\n");
        return 1;
    }

    // Suppress GPU and backend logging unless --verbose was specified.
    // Order matters: llama_log_set must come FIRST because llamafile_cuda_log_set
    // triggers DSO loading which calls ggml_backend_register() in the main exe.
    if (!FLAG_verbose) {
        llama_log_set((ggml_log_callback)llamafile_log_callback_null, NULL);
        llamafile_metal_log_set(llamafile_log_callback_null, NULL);
        llamafile_cuda_log_set(llamafile_log_callback_null, NULL);
    }

    // For CLI mode, suppress logo (but respect --verbose if user specified it)
    if (args.mode == lf::ProgramMode::CLI) {
        FLAG_nologo = 1;
    }

    // For non-server modes, suppress llama.cpp's own logging (model loading,
    // backend registration, CUDA init, etc.) unless --verbose was specified.
    // We inject --log-verbosity 1 (errors only) into the argv so that
    // common_init() picks it up when it re-sets the log callback.
    static char log_flag[] = "--log-verbosity";
    static char log_val[] = "1";
    std::vector<char *> quiet_argv;
    if (!FLAG_verbose && args.mode != lf::ProgramMode::SERVER) {
        for (int i = 0; i < args.llama_argc; i++)
            quiet_argv.push_back(args.llama_argv[i]);
        quiet_argv.push_back(log_flag);
        quiet_argv.push_back(log_val);
        quiet_argv.push_back(nullptr);
        args.llama_argc = static_cast<int>(quiet_argv.size()) - 1;
        args.llama_argv = quiet_argv.data();
    }

    // Initialize GPU support (triggers dynamic loading of GPU backends)
    llamafile_has_gpu();

    // Route to appropriate mode
    switch (args.mode) {
        case lf::ProgramMode::SERVER:
            // Server only mode
            return server_main(args.llama_argc, args.llama_argv, nullptr, nullptr);

        case lf::ProgramMode::CHAT:
            // Chat only mode (no server)
            return lf::chatbot::main(args.llama_argc, args.llama_argv);

        case lf::ProgramMode::CLI:
            // Single prompt -> response mode
            return lf::chatbot::cli_main(args.llama_argc, args.llama_argv);

        case lf::ProgramMode::AUTO:
            // Combined mode: server on main thread, TUI as HTTP client on background thread
            return lf::combined_main(args);
    }

    return 1;
}
