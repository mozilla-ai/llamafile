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

#pragma once

#include <string>

namespace lf {

// Program execution modes
enum class ProgramMode {
    AUTO,      // Default: combined chat + server
    CHAT,      // --chat: TUI chat only
    SERVER,    // --server: HTTP server only
    CLI,       // --cli: Single prompt -> response, then exit
};

// Parsed llamafile arguments
struct LlamafileArgs {
    ProgramMode mode = ProgramMode::AUTO;

    // Filtered argc/argv for llama.cpp (excludes llamafile-specific args)
    int llama_argc = 0;
    char** llama_argv = nullptr;

    // System prompt captured from -p (needed for combined mode where SERVER
    // parsing excludes -p)
    std::string system_prompt;

    // Model path captured from -m (for display in combined mode TUI)
    std::string model_path;

    // Note: Llamafile-specific flags are stored in FLAG_* globals (llamafile.h):
    //   --verbose  -> FLAG_verbose
    //   --nothink  -> FLAG_nothink
    //   --gpu      -> FLAG_gpu (set by llamafile_early_gpu_init)
};

// Parse command line arguments, determine execution mode, and filter out
// llamafile-specific arguments before passing to llama.cpp.
//
// This function:
// 1. Calls llamafile_early_gpu_init() to handle GPU flags
// 2. Determines the program mode from --chat, --server, --cli flags
// 3. Removes llamafile-specific flags from argv
// 4. Returns filtered argc/argv suitable for llama.cpp
LlamafileArgs parse_llamafile_args(int argc, char** argv);

} // namespace lf
