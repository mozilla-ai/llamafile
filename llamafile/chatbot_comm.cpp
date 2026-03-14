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
#include "chatbot_backend.h"

#include <cassert>
#include <sstream>
#include <vector>

#include "llama.h"
#include "llamafile.h"
#include "sampling.h"  // llama.cpp common/sampling.h

namespace lf {
namespace chatbot {

// handle irc style commands like: `/arg0 arg1 arg2`
bool handle_command(const char *command) {
    if (!strcmp(command, "/?")) {
        const std::vector<std::string> args = {"?"};
        on_help(args);
        return true;
    }
    if (!(command[0] == '/' && std::isalpha(command[1])))
        return false;
    std::vector<std::string> args;
    std::istringstream iss(command + 1);
    std::string arg;
    while (iss >> arg)
        args.push_back(arg);
    if (args[0] == "exit" || args[0] == "bye") {
        exit(0);
    } else if (args[0] == "help") {
        on_help(args);
    } else if (args[0] == "stats") {
        g_backend->print_stats();
    } else if (args[0] == "context") {
        int used = g_backend->context_used();
        int max = g_backend->context_max();
        printf("%d out of %d context tokens used (%d tokens remaining)\n",
               used, max, max - used);
    } else if (args[0] == "manual") {
        if (!g_backend->supports_manual_mode()) {
            err("manual mode not available in this mode — use --chat for direct model access");
        } else {
            on_manual(args);
        }
    } else if (args[0] == "clear") {
        g_backend->on_clear();
    } else if (args[0] == "dump") {
        if (!g_backend->supports_dump()) {
            err("dump not available in this mode — use --chat for direct model access");
        } else {
            on_dump(args);
        }
    } else if (args[0] == "push") {
        g_backend->on_push();
    } else if (args[0] == "pop") {
        g_backend->on_pop();
    } else if (args[0] == "undo") {
        g_backend->on_undo();
    } else if (args[0] == "forget") {
        g_backend->on_forget(1);
    } else if (args[0] == "stack") {
        on_stack(args);
    } else if (args[0] == "upload") {
        on_upload(args);
    } else {
        err("%s: unrecognized command", args[0].c_str());
    }
    return true;
}

} // namespace chatbot
} // namespace lf
