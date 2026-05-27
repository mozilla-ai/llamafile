// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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
// agentfile: cosmocc-compiled APE binary that runs an agent.cpp agent loop
// from a single CLI invocation.
//
// v0: ships 8 tools (read_file, file_glob_search, grep_search, write_file,
// edit_file, apply_diff, exec_shell_command, get_datetime) vendored from
// llama.cpp/tools/server/server-tools.cpp. Destructive tools prompt the user
// on stderr unless --yes is given.
//

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef COSMOCC
#include <cosmo.h>
#endif

#include "llama.h"
#include "chat.h"

#include "agent.h"
#include "callbacks.h"
#include "error.h"
#include "model.h"
#include "tool.h"

#include "llamafile.h"

#include "confirmation_callback.h"
#include "stdout_callback.h"
#include "tools/tools.h"

#include <sstream>

namespace {

void null_log_callback(ggml_log_level, const char *, void *) {}

void print_usage(const char *prog) {
    fprintf(stderr,
            "agentfile — agentic CLI on top of agent.cpp + llama.cpp\n"
            "\n"
            "Usage:\n"
            "  %s -m MODEL.gguf -p \"prompt\" [options]\n"
            "\n"
            "Options:\n"
            "  -m PATH              Path to a GGUF model file (required)\n"
            "  -p TEXT              User prompt (required)\n"
            "  -s TEXT              System instructions (default: helpful assistant)\n"
            "  --tools LIST         Comma-separated tool list, or \"all\" (default),\n"
            "                       or \"read_only\". Available tools:\n"
            "                       read_file, file_glob_search, grep_search,\n"
            "                       get_datetime, write_file, edit_file,\n"
            "                       apply_diff, exec_shell_command, http_fetch\n"
            "  --yes                Skip confirmation prompts for destructive tools\n"
            "  --max-iterations N   Cap agent loop at N LLM calls (default: unlimited)\n"
            "  --quiet              Don't print tool-execution progress to stderr\n"
            "  -h                   Show this help\n",
            prog);
}

// Filter a tool vector by name list. Tools whose name is NOT in `keep` are
// dropped. If `keep` is empty, all tools are kept.
void filter_tools(std::vector<std::unique_ptr<agent_cpp::Tool>> &tools,
                  const std::set<std::string> &keep) {
    if (keep.empty()) return;
    tools.erase(std::remove_if(tools.begin(), tools.end(),
                               [&](const std::unique_ptr<agent_cpp::Tool> &t) {
                                   return !keep.count(t->get_name());
                               }),
                tools.end());
}

std::set<std::string> parse_tools_flag(const std::string &spec) {
    if (spec == "all" || spec.empty()) return {};
    if (spec == "read_only") {
        return {"read_file", "file_glob_search", "grep_search", "get_datetime"};
    }
    std::set<std::string> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t a = item.find_first_not_of(" \t");
        size_t b = item.find_last_not_of(" \t");
        if (a != std::string::npos && b != std::string::npos)
            out.insert(item.substr(a, b - a + 1));
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);

#ifdef COSMOCC
    argc = cosmo_args("/zip/.args", &argv);
#endif

    std::string model_path;
    std::string prompt;
    std::string instructions =
        "You are a helpful assistant. Answer concisely.";
    bool always_yes = false;
    bool quiet = false;
    int max_iterations = 0;  // 0 = no cap
    std::string tools_spec = "all";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            instructions = argv[++i];
        } else if (std::strcmp(argv[i], "--tools") == 0 && i + 1 < argc) {
            tools_spec = argv[++i];
        } else if (std::strcmp(argv[i], "--yes") == 0) {
            always_yes = true;
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            quiet = true;
        } else if (std::strcmp(argv[i], "--max-iterations") == 0 && i + 1 < argc) {
            max_iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Silence llama.cpp / ggml logging before backend init.
    llama_log_set(null_log_callback, nullptr);

    // Initialize llamafile GPU backends (Metal, CUDA, Vulkan, ROCm).
    // This is also what triggers backend registration.
    llamafile_has_gpu();
    llama_backend_init();

    try {
        auto weights = agent_cpp::ModelWeights::create(model_path);

        agent_cpp::ModelConfig cfg;  // defaults: temp=0, top_p=1, top_k=0
        cfg.n_batch = 256;  // agent.cpp's default of -1 doesn't play with llama_context
        auto model = agent_cpp::Model::create_with_weights(weights, cfg);

        auto tools = agentfile::tools::build_default_tools();
        filter_tools(tools, parse_tools_flag(tools_spec));

        std::vector<std::unique_ptr<agent_cpp::Callback>> callbacks;
        callbacks.emplace_back(
            std::make_unique<agentfile::DestructiveOpsConfirmationCallback>(
                always_yes));
        if (!quiet) {
            callbacks.emplace_back(
                std::make_unique<agentfile::ProgressCallback>());
        }
        if (max_iterations > 0) {
            callbacks.emplace_back(
                std::make_unique<agentfile::MaxIterationsCallback>(
                    max_iterations));
        }
        agent_cpp::Agent agent(std::move(model), std::move(tools),
                               std::move(callbacks), instructions);

        std::vector<common_chat_msg> messages;
        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = prompt;
        messages.push_back(std::move(user_msg));

        std::string reply = agent.run_loop(messages);
        fputs(reply.c_str(), stdout);
        fputc('\n', stdout);
    } catch (const agent_cpp::Error &e) {
        fprintf(stderr, "agentfile error: %s\n", e.what());
        return 2;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 3;
    }

    llama_backend_free();
    return 0;
}
