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

#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/pool.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/signals.h"
#include "llamafile/server/slots.h"
#include "llamafile/server/time.h"
#include "llamafile/server/tokenbucket.h"
#include "llamafile/server/utils.h"
#include "llamafile/version.h"
#include <cassert>
#include <cosmo.h>

// Global LoRA adapter storage for multiple adapters
#define MAX_LORA_ADAPTERS 8
#include <string>
struct lora_adapter_container {
    struct llama_lora_adapter* adapter;
    float scale;
    std::string name;  // Model/adapter name for identification
    bool applied;      // Whether this adapter is currently applied to slots
};

// Make these externally accessible for HTTP endpoint
struct lora_adapter_container g_lora_adapters[MAX_LORA_ADAPTERS] = {};
int g_lora_adapters_count = 0;

// Function to get the first global LoRA adapter for backward compatibility
extern "C" struct llama_lora_adapter* llamafiler_get_lora_adapter() {
    return g_lora_adapters_count > 0 ? g_lora_adapters[0].adapter : nullptr;
}

// Function to get all LoRA adapters and their count
extern "C" int llamafiler_get_lora_adapters(struct llama_lora_adapter** adapters, float* scales, int max_adapters) {
    int count = g_lora_adapters_count < max_adapters ? g_lora_adapters_count : max_adapters;
    for (int i = 0; i < count; i++) {
        adapters[i] = g_lora_adapters[i].adapter;
        scales[i] = g_lora_adapters[i].scale;
    }
    return count;
}

namespace lf {
namespace server {

Server* g_server;

int
main(int argc, char* argv[])
{
    llamafile_check_cpu();
    signal(SIGPIPE, SIG_IGN);
    mallopt(M_GRANULARITY, 2 * 1024 * 1024);
    mallopt(M_MMAP_THRESHOLD, 16 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 128 * 1024 * 1024);
    ShowCrashReports();

    if (llamafile_has(argv, "--version")) {
        puts("llamafiler v" LLAMAFILE_VERSION_STRING);
        exit(0);
    }

    if (llamafile_has(argv, "-h") || llamafile_has(argv, "-help") ||
        llamafile_has(argv, "--help")) {
        llamafile_help("/zip/llamafile/server/main.1.asc");
        __builtin_unreachable();
    }

    // get config
    argc = cosmo_args("/zip/.args", &argv);
    llamafile_get_flags(argc, argv);

    // initialize subsystems
    time_init();
    tokenbucket_init();

    // we must disable the llama.cpp logger
    // otherwise pthread_cancel() will cause deadlocks
    if (!llamafile_has(argv, "--verbose"))
        FLAG_log_disable = true;

    // load model
    // --lora implies --no-mmap (as per llama.cpp server)
    bool use_mmap = FLAG_mmap && (FLAG_lora_adapters_count == 0);
    llama_model_params mparams = {
        .n_gpu_layers = FLAG_n_gpu_layers,
        .split_mode = (enum llama_split_mode)FLAG_split_mode,
        .main_gpu = FLAG_main_gpu,
        .tensor_split = nullptr,
        .rpc_servers = nullptr,
        .progress_callback = nullptr,
        .progress_callback_user_data = nullptr,
        .kv_overrides = nullptr,
        .vocab_only = false,
        .use_mmap = use_mmap,
        .use_mlock = FLAG_mlock,
        .check_tensors = false,
    };
    llama_model* model = llama_load_model_from_file(FLAG_model, mparams);
    if (!model) {
        fprintf(stderr, "%s: failed to load model\n", FLAG_model);
        exit(1);
    }

    // load LoRA adapters if specified
    if (FLAG_lora_adapters_count > 0) {
        const char* apply_mode = FLAG_lora_init_without_apply ? "without applying" : "and applying";
        SLOG("loading %d LoRA adapter(s) %s", FLAG_lora_adapters_count, apply_mode);
        
        for (int i = 0; i < FLAG_lora_adapters_count; i++) {
            char scale_buf[32];
            snprintf(scale_buf, sizeof(scale_buf), "%.2f", FLAG_lora_adapters[i].scale);
            
            // Generate model name from filename
            const char* path = FLAG_lora_adapters[i].path;
            const char* filename = strrchr(path, '/');
            filename = filename ? filename + 1 : path;
            
            // Remove file extension for cleaner name
            std::string model_name(filename);
            size_t dot_pos = model_name.find_last_of('.');
            if (dot_pos != std::string::npos) {
                model_name = model_name.substr(0, dot_pos);
            }
            
            SLOG("loading LoRA adapter %d ('%s') from %s with scale %s", i + 1, 
                 model_name.c_str(), path, scale_buf);
                 
            g_lora_adapters[i].adapter = llama_lora_adapter_init(model, path);
            g_lora_adapters[i].scale = FLAG_lora_adapters[i].scale;
            g_lora_adapters[i].name = model_name;
            g_lora_adapters[i].applied = !FLAG_lora_init_without_apply;  // Apply unless flag is set
            
            if (!g_lora_adapters[i].adapter) {
                fprintf(stderr, "%s: failed to load LoRA adapter from %s\n", FLAG_model, path);
                // Cleanup previously loaded adapters
                for (int j = 0; j < i; j++) {
                    if (g_lora_adapters[j].adapter) {
                        llama_lora_adapter_free(g_lora_adapters[j].adapter);
                    }
                }
                llama_free_model(model);
                exit(1);
            }
            g_lora_adapters_count++;
        }
        
        if (FLAG_lora_init_without_apply) {
            SLOG("all LoRA adapters loaded successfully but not applied (use /lora-adapters API to apply)");
        } else {
            SLOG("all LoRA adapters loaded and applied successfully");
        }
    }

    // create slots
    Slots* slots = new Slots(model);
    if (!slots->start(FLAG_slots)) {
        SLOG("no slots could be created");
        exit(1);
    }

    // create server
    if (FLAG_workers <= 0)
        FLAG_workers = __get_cpu_count() + 4;
    if (FLAG_workers <= 0)
        FLAG_workers = 16;
    set_thread_name("server");
    g_server =
      new Server(create_listening_socket(FLAG_listen, 0, 0), slots, model);
    for (int i = 0; i < FLAG_workers; ++i)
        npassert(!g_server->spawn());

    // run server
    signals_init();
    llama_backend_init();
    g_server->run();
    llama_backend_free();
    signals_destroy();

    // shutdown server
    SLOG("shutdown");
    g_server->shutdown();
    g_server->close();
    delete g_server;
    delete slots;
    
    // Cleanup LoRA adapters
    for (int i = 0; i < g_lora_adapters_count; i++) {
        if (g_lora_adapters[i].adapter) {
            llama_lora_adapter_free(g_lora_adapters[i].adapter);
        }
    }
    
    llama_free_model(model);
    tokenbucket_destroy();
    time_destroy();
    SLOG("exit");

    // quality assurance
    llamafile_task_shutdown();
    while (!pthread_orphan_np())
        pthread_decimate_np();
    CheckForMemoryLeaks();
    return 0;
}

} // namespace server
} // namespace lf
