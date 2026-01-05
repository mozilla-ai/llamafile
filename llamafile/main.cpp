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
// This is the main entry point for llamafile. It provides a TUI (Text User
// Interface) for interactive chatting with LLMs using the llama.cpp backend,
// or can run as an HTTP server for API access.
//
// Usage:
//   llamafile -m model.gguf              # Start TUI with model (default)
//   llamafile -m model.gguf --mmproj ... # TUI with vision model
//   llamafile -m model.gguf --chat       # TUI mode (explicit)
//   llamafile -m model.gguf --server     # HTTP server mode
//

#include "chatbot.h"
#include "llamafile.h"
#include <iostream>
#include <set>
#include <string>

#ifdef COSMOCC
#include <cosmo.h>
#endif

// Forward declaration for server main (defined in llama.cpp/tools/server/server.cpp)
extern int server_main(int argc, char **argv, char **envp);

enum Program {
    PROG_UNKNOWN,
    PROG_CHAT,
    PROG_SERVER,
};

static enum Program determine_program(char *argv[]) {
    enum Program prog = PROG_UNKNOWN;
    for (int i = 0; argv[i]; ++i) {
        if (!strcmp(argv[i], "--chat")) {
            prog = PROG_CHAT;
        } else if (!strcmp(argv[i], "--server")) {
            prog = PROG_SERVER;
        }
    }
    return prog;
}

int removeArgs(int argc, char* argv[], const std::set<std::string>& args_to_remove) {

    int write_idx = 0;
    for (int read_idx = 0; read_idx < argc; ++read_idx) {
        std::string current_arg = argv[read_idx];

        // Check if the current argument should be removed.
        // .count() is a clean way to check for existence in a set.
        if (args_to_remove.count(current_arg)) {
            continue; // Skip the current argument (and its value if applicable).
        }

        // Keep this argument.
        // If the write position is different from the read position, copy the pointer.
        if (write_idx != read_idx) {
            argv[write_idx] = argv[read_idx];
        }
        write_idx++;
    }

    // `write_idx` is now the new number of arguments
    // NULL out argv[write_idx] to guarantee that argv[argc] == NULL
    argv[write_idx] = nullptr;

    return write_idx;
}


int main(int argc, char **argv, char **envp) {
#ifdef COSMOCC
    // Load arguments from zip file if present (for bundled llamafiles)
    argc = cosmo_args("/zip/.args", &argv);
#endif

    enum Program prog = determine_program(argv);

    // Server mode: run HTTP server
    if (prog == PROG_SERVER) {
        // remove arguments which llama.cpp does not support
        std::set<std::string> args_to_remove = {"--server"};
        argc = removeArgs(argc, argv, args_to_remove);
        return server_main(argc, argv, envp);
    }

    // Chat mode (explicit --chat or default when no -p/-f/--random-prompt)
    if (prog == PROG_CHAT ||
        (prog == PROG_UNKNOWN &&
         !llamafile_has(argv, "-p") &&
         !llamafile_has(argv, "-f"))) {
        return lf::chatbot::main(argc, argv);
    }

    // If we have -p, -f, or --random-prompt without explicit mode,
    // default to chatbot for now (could add CLI mode later)
    return lf::chatbot::main(argc, argv);
}
