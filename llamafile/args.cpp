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

#include "args.h"
#include "llamafile.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace lf {

// Static storage for filtered argv (persists after function returns)
static std::vector<char*> g_filtered_argv;

// Helper: returns true if arg is a llamafile-specific flag (not recognized by llama.cpp)
static bool is_llamafile_flag(const char* arg) {
    return strcmp(arg, "--server") == 0 ||
           strcmp(arg, "--chat") == 0 ||
           strcmp(arg, "--cli") == 0 ||
           strcmp(arg, "--gpu") == 0 ||
           strcmp(arg, "--ascii") == 0 ||
           strcmp(arg, "--nologo") == 0 ||
           strcmp(arg, "--nothink") == 0 ||
           strcmp(arg, "--unsecure") == 0 ||
           strcmp(arg, "--confine-reads") == 0 ||
           strcmp(arg, "--version") == 0;
}

// llama.cpp b11100 replaced --mmap/--no-mmap/--mlock/-dio/-ndio with
// --load-mode and now rejects them, which would break existing llamafiles
// whose .args still use them. They are dropped here and one equivalent
// "--load-mode <mode>" is inserted where the last of them was, so an
// explicit --load-mode keeps working and the last flag wins, as upstream.
//
// The mode follows the original semantics, where the flags were independent
// settings: mmap on by default, --mlock adds locking to it (mmap+mlock, not
// plain mlock), and direct I/O takes precedence over mmap.
static const char* legacy_load_flag(const char* arg, bool* mmap, bool* mlock, bool* dio) {
    if (strcmp(arg, "--mmap") == 0) {
        *mmap = true;
    } else if (strcmp(arg, "--no-mmap") == 0) {
        *mmap = false;
    } else if (strcmp(arg, "--mlock") == 0) {
        *mlock = true;
    } else if (strcmp(arg, "-dio") == 0 || strcmp(arg, "--direct-io") == 0) {
        *dio = true;
    } else if (strcmp(arg, "-ndio") == 0 || strcmp(arg, "--no-direct-io") == 0) {
        *dio = false;
    } else {
        return nullptr;
    }
    return arg;
}

static char* legacy_load_mode(bool mmap, bool mlock, bool dio) {
    static char kNone[] = "none", kMmap[] = "mmap", kMlock[] = "mlock",
                kMmapMlock[] = "mmap+mlock", kDio[] = "dio";
    if (dio)
        return kDio;
    if (mmap)
        return mlock ? kMmapMlock : kMmap;
    return mlock ? kMlock : kNone;
}

LlamafileArgs parse_llamafile_args(int argc, char** argv) {
    LlamafileArgs args;

    // Early GPU init must happen before we filter args
    // This reads --gpu and -ngl flags to set FLAG_gpu
    llamafile_early_gpu_init(argv);

    // Capture -p/--prompt value before filtering (needed for combined mode
    // where SERVER parsing excludes -p)
    // Note: Loop does not break early; if multiple -p flags are given,
    // the last occurrence wins (intentional for override flexibility)
    for (int i = 0; i < argc; ++i) {
        if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            args.system_prompt = argv[i + 1];
        }
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            args.model_path = argv[i + 1];
        }
        if ((strcmp(argv[i], "-hf") == 0 || strcmp(argv[i], "-hfr") == 0 ||
             strcmp(argv[i], "--hf-repo") == 0 || strcmp(argv[i], "-mu") == 0 ||
             strcmp(argv[i], "--model-url") == 0) && i + 1 < argc) {
            args.remote_model = argv[i + 1];
        }
    }

    // Determine execution mode from flags
    // Priority: explicit flags override defaults
    if (llamafile_has(argv, "--server")) {
        args.mode = ProgramMode::SERVER;
    } else if (llamafile_has(argv, "--chat")) {
        args.mode = ProgramMode::CHAT;
    } else if (llamafile_has(argv, "--cli")) {
        args.mode = ProgramMode::CLI;
    } else {
        // AUTO mode: will run combined chat + server
        args.mode = ProgramMode::AUTO;
    }

    // Check verbose flag
    FLAG_verbose = llamafile_has(argv, "--verbose") ? 1 : 0;

    // Check --nothink flag (filters thinking/reasoning content in CLI mode)
    FLAG_nothink = llamafile_has(argv, "--nothink");

    // Check logo flags
    FLAG_nologo = llamafile_has(argv, "--nologo");
    FLAG_ascii = llamafile_has(argv, "--ascii");

    // Check --unsecure flag (disables pledge() sandboxing, see sandbox.c)
    FLAG_unsecure = llamafile_has(argv, "--unsecure");

    // Check --confine-reads flag (opt-in unveil() path confinement, server mode)
    FLAG_confine_reads = llamafile_has(argv, "--confine-reads");

    // Filter out llamafile-specific arguments
    // These are not recognized by llama.cpp and would cause errors
    g_filtered_argv.clear();

    bool legacy_mmap = true, legacy_mlock = false, legacy_dio = false;
    const char* last_legacy = nullptr;
    size_t legacy_pos = 0;

    for (int i = 0; i < argc; ++i) {
        const char* arg = argv[i];

        // Translate removed load flags (see legacy_load_flag)
        if (const char* flag = legacy_load_flag(arg, &legacy_mmap, &legacy_mlock, &legacy_dio)) {
            last_legacy = flag;
            legacy_pos = g_filtered_argv.size();
            continue;
        }

        // Skip llamafile-specific flags
        if (is_llamafile_flag(arg)) {
            // --gpu takes a value argument, skip it too
            if (strcmp(arg, "--gpu") == 0 && i + 1 < argc) {
                ++i;
            }
            continue;
        }

        // Keep this argument
        g_filtered_argv.push_back(argv[i]);
    }

    if (last_legacy) {
        static char kLoadMode[] = "--load-mode";
        char* mode = legacy_load_mode(legacy_mmap, legacy_mlock, legacy_dio);
        fprintf(stderr, "warning: %s is no longer supported by llama.cpp; using --load-mode %s "
                        "(see docs/cli_arguments.md)\n", last_legacy, mode);
        g_filtered_argv.insert(g_filtered_argv.begin() + legacy_pos, {kLoadMode, mode});
    }

    // Null-terminate argv array (required by convention)
    g_filtered_argv.push_back(nullptr);

    args.llama_argc = static_cast<int>(g_filtered_argv.size()) - 1;
    args.llama_argv = g_filtered_argv.data();

    return args;
}

} // namespace lf
