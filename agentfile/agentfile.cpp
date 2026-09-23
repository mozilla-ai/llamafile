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
// agentfile: a self-contained agentic CLI. One APE binary runs an
// agent.cpp loop with llama.cpp's server tools (plus agentfile's own
// http_fetch/web_search), entirely in-process: prompt in, answer out.
//
// Layout: this file is the only translation unit — flags, wiring, and the
// interactive loop. Tools come through server_tools_adapter.h; the loop's
// observers (confirmation, progress, recorders, iteration cap) live in
// callbacks/. Answers go to stdout; everything else goes to stderr.
//

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
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

#include "callbacks/confirmation.h"
#include "callbacks/max_iterations.h"
#include "callbacks/progress.h"
#include "callbacks/session_recorder.h"
#include "callbacks/trace.h"
#include "server_tools_adapter.h"

#include <sstream>

namespace {

// Default context window. Sized so one full http_fetch result (64 KB body,
// roughly 16-24k tokens) plus history fits comfortably.
constexpr int kDefaultCtx = 32 * 1024;

void null_log_callback(ggml_log_level, const char *, void *) {}

// The tools section of the help is generated from the toolbox so it can't
// drift from the registered tools. SEARXNG_URL is honored so a configured
// environment (or packaged agent) shows its true availability.
void print_tools_help() {
    try {
        const char *env = std::getenv("SEARXNG_URL");
        agentfile::ServerToolbox toolbox(env ? env : "", "");
        auto guarded = toolbox.write_tool_names();
        std::string ro, wr;
        for (const auto &name : toolbox.tool_names()) {
            std::string &dst = guarded.count(name) ? wr : ro;
            dst += (dst.empty() ? "" : ", ") + name;
        }
        fprintf(stderr,
                "Tools (choose with --tools LIST, \"all\" or \"read_only\"; "
                "default: all):\n"
                "  read-only:  %s\n"
                "  guarded:    %s\n"
                "              (ask confirmation before each call; --yes skips,\n"
                "              --confirm restores)\n",
                ro.c_str(), wr.c_str());
        for (const auto &[name, why] : toolbox.missing()) {
            fprintf(stderr, "  unavailable: %s — %s\n", name.c_str(),
                    why.c_str());
        }
        fprintf(stderr,
                "  --searxng-url URL    SearXNG instance used by web_search;\n"
                "                       also read from SEARXNG_URL\n"
                "  --tools-runtime SPEC Run every tool inside an existing\n"
                "                       container, e.g. \"docker-container:NAME\"\n"
                "                       (default: tools run on this host)\n"
                "  Tools run with this process's permissions; agentfile does not\n"
                "  sandbox itself. The confirmation prompt is the only gate (--yes\n"
                "  removes it); use --tools-runtime for isolation.\n");
    } catch (const std::exception &e) {
        fprintf(stderr, "Tools: unavailable in this build (%s)\n", e.what());
    }
}

void print_usage(const char *prog) {
    fprintf(stderr,
            "agentfile — agentic CLI on top of agent.cpp + llama.cpp\n"
            "\n"
            "Usage:\n"
            "  %s -m MODEL.gguf -p \"prompt\" [options]\n"
            "  echo \"prompt\" | %s -m MODEL.gguf [options]\n"
            "\n"
            "Options:\n"
            "  -m PATH              Path to a GGUF model file (required)\n"
            "  -p TEXT              User prompt (read from stdin if omitted\n"
            "                       and stdin is not a terminal)\n"
            "  -c, --ctx-size N     Context window in tokens (default: %d).\n"
            "                       0 = the model's full native context — mind the\n"
            "                       KV-cache memory on long-context models\n"
            "  -s TEXT              System instructions (default: helpful assistant)\n"
            "  --system-file PATH   Read system instructions from a file\n"
            "                       (works with /zip/ paths in packaged agents)\n"
            "  --tools LIST         Tools the model may use (see Tools below)\n"
            "  --session FILE       Record the conversation as a pi session\n"
            "                       (https://pi.dev, session-format v3 JSONL;\n"
            "                       overwrites FILE)\n"
            "  --trace FILE         Record spans as OTLP/JSON lines (OpenTelemetry\n"
            "                       collector `otlpjsonfile` receiver format;\n"
            "                       overwrites FILE)\n"
            "  --think              Enable model reasoning (<think> blocks; shown\n"
            "                       dimmed on stderr, never in the answer).\n"
            "                       --no-think undoes it (default: off)\n"
            "  -i, --interactive    After the answer, prompt for a follow-up and\n"
            "                       continue the conversation (empty line or EOF\n"
            "                       ends the session). --no-interactive undoes it.\n"
            "  --yes                Skip confirmation prompts for destructive tools\n"
            "  --confirm            Re-enable confirmation prompts (inverse of\n"
            "                       --yes; later flag wins)\n"
            "  --max-iterations N   Cap agent loop at N LLM calls (default: unlimited)\n"
            "  --quiet              Don't print tool-execution progress to stderr\n"
            "                       (destructive tools run under --yes are still\n"
            "                       logged, one line each)\n"
            "  -v, --verbose        Also print truncated tool results to stderr\n"
            "  -h                   Show this help\n"
            "\n",
            prog, prog, kDefaultCtx);
    print_tools_help();
    fprintf(stderr,
            "\n"
            "Exit codes: 0 ok, 1 usage error, 2 agent error, 3 other error,\n"
            "            4 --max-iterations cap reached\n");
}

// Read a whole FILE* into a string.
bool slurp(FILE *f, std::string &out) {
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    return !ferror(f);
}

// Read one interactive follow-up, prompting on stderr. When the initial
// prompt was piped in, stdin is consumed — read from the controlling
// terminal instead. Returns an empty string on EOF, no terminal, or an
// empty line; the caller ends the session.
std::string read_followup() {
    fprintf(stderr, "%s\n> %s", agentfile::dim(), agentfile::dim_off());
    fflush(stderr);

    FILE *tty = nullptr;
    FILE *in = stdin;
    if (!isatty(STDIN_FILENO)) {
        tty = fopen("/dev/tty", "r");
        if (!tty) return "";
        in = tty;
    }
    std::string line;
    char buf[4096];
    while (fgets(buf, sizeof(buf), in)) {
        line += buf;
        if (line.back() == '\n') break;
    }
    if (tty) fclose(tty);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    return line;
}

std::set<std::string> parse_tools_flag(const std::string &spec) {
    if (spec == "all" || spec.empty()) return {};
    if (spec == "read_only") {
        return {"read_file", "file_glob_search", "grep_search",
                "get_datetime", "get_info"};
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
    bool interactive = false;
    bool think = false;
    int n_ctx = kDefaultCtx; // tokens; 0 = model native
    int verbosity = 1;       // 0 = --quiet, 1 = default, 2 = --verbose
    int max_iterations = 0;  // 0 = no cap
    std::string tools_spec = "all";
    std::string session_path;
    std::string trace_path;
    std::string searxng_url;
    std::string tools_runtime;
    if (const char *env = std::getenv("SEARXNG_URL")) searxng_url = env;

    // Parsing is strictly last-wins and every mode flag has an inverse, so
    // defaults baked into a packaged agent's /zip/.args (which cosmo_args
    // prepends before the real command line) can always be overridden.
    auto need_value = [&](int &i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "agentfile: %s requires a value\n", argv[i]);
            exit(1);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = need_value(i);
        } else if (std::strcmp(argv[i], "-p") == 0) {
            prompt = need_value(i);
        } else if (std::strcmp(argv[i], "-c") == 0 ||
                   std::strcmp(argv[i], "--ctx-size") == 0) {
            n_ctx = std::atoi(need_value(i));
        } else if (std::strcmp(argv[i], "-s") == 0) {
            instructions = need_value(i);
        } else if (std::strcmp(argv[i], "--system-file") == 0) {
            const char *path = need_value(i);
            FILE *f = fopen(path, "r");
            if (!f) {
                fprintf(stderr, "agentfile: --system-file: cannot open %s\n",
                        path);
                return 1;
            }
            instructions.clear();
            bool ok = slurp(f, instructions);
            fclose(f);
            if (!ok) {
                fprintf(stderr, "agentfile: --system-file: error reading %s\n",
                        path);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--tools") == 0) {
            tools_spec = need_value(i);
        } else if (std::strcmp(argv[i], "--searxng-url") == 0) {
            searxng_url = need_value(i);
        } else if (std::strcmp(argv[i], "--tools-runtime") == 0) {
            tools_runtime = need_value(i);
        } else if (std::strcmp(argv[i], "--session") == 0) {
            session_path = need_value(i);
        } else if (std::strcmp(argv[i], "--trace") == 0) {
            trace_path = need_value(i);
        } else if (std::strcmp(argv[i], "-i") == 0 ||
                   std::strcmp(argv[i], "--interactive") == 0) {
            interactive = true;
        } else if (std::strcmp(argv[i], "--no-interactive") == 0) {
            interactive = false;
        } else if (std::strcmp(argv[i], "--think") == 0) {
            think = true;
        } else if (std::strcmp(argv[i], "--no-think") == 0) {
            think = false;
        } else if (std::strcmp(argv[i], "--yes") == 0) {
            always_yes = true;
        } else if (std::strcmp(argv[i], "--confirm") == 0) {
            always_yes = false;
        } else if (std::strcmp(argv[i], "--quiet") == 0) {
            verbosity = 0;
        } else if (std::strcmp(argv[i], "-v") == 0 ||
                   std::strcmp(argv[i], "--verbose") == 0) {
            verbosity = 2;
        } else if (std::strcmp(argv[i], "--max-iterations") == 0) {
            max_iterations = std::atoi(need_value(i));
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "agentfile: unknown argument: %s\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // No -p and stdin is piped: the pipe is the prompt.
    if (prompt.empty() && !isatty(STDIN_FILENO)) {
        if (!slurp(stdin, prompt)) {
            fprintf(stderr, "agentfile: error reading prompt from stdin\n");
            return 1;
        }
        while (!prompt.empty() &&
               (prompt.back() == '\n' || prompt.back() == '\r'))
            prompt.pop_back();
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Silence llama.cpp / ggml logging before backend init. GPU backends are
    // runtime-loaded DSOs with their own copy of ggml's logger, so the
    // llamafile_*_log_set hooks must silence each one separately (they queue
    // the callback if the backend isn't loaded yet).
    llama_log_set(null_log_callback, nullptr);
    llamafile_metal_log_set(llamafile_log_callback_null, nullptr);
    llamafile_cuda_log_set(llamafile_log_callback_null, nullptr);
    llamafile_vulkan_log_set(llamafile_log_callback_null, nullptr);

    // Initialize llamafile GPU backends (Metal, CUDA, Vulkan, ROCm).
    // This is also what triggers backend registration.
    llamafile_has_gpu();
    llama_backend_init();

    try {
        // Build and validate the toolset before loading the model, so a bad
        // --tools value fails fast. A requested tool that isn't registered
        // is an error, not a silently smaller toolset — the model would
        // otherwise improvise with whatever tools remain.
        //
        // Tools come from llama.cpp's server-tools registry via
        // ServerToolbox (plus agentfile's own http_fetch/web_search); the
        // toolbox owns them and must outlive the Agent below.
        agentfile::ServerToolbox toolbox(searxng_url, tools_runtime);
        auto keep = parse_tools_flag(tools_spec);
        {
            auto known = toolbox.tool_names();
            for (const auto &name : keep) {
                if (known.count(name)) continue;
                auto it = toolbox.missing().find(name);
                if (it != toolbox.missing().end()) {
                    fprintf(stderr, "agentfile: %s %s\n", name.c_str(),
                            it->second.c_str());
                } else {
                    fprintf(stderr, "agentfile: unknown tool: %s\n",
                            name.c_str());
                }
                return 1;
            }
        }
        auto tools = toolbox.make_adapters(keep);

        auto weights = agent_cpp::ModelWeights::create(model_path);

        agent_cpp::ModelConfig cfg;  // defaults: temp=0, top_p=1, top_k=0
        // agent.cpp's default (-1) wraps to ~4 billion in llama_context's
        // unsigned n_batch and kills context creation, so always set one.
        // 2048 matches llama.cpp's default; llamafile's TUI uses 256 for
        // finer prefill progress display, which agentfile doesn't have.
        cfg.n_batch = 2048;
        cfg.n_ctx = n_ctx;  // 0 = model's native context
        cfg.enable_thinking = think;
        auto model = agent_cpp::Model::create_with_weights(weights, cfg);

        // Model name for session/trace records: the GGUF basename.
        std::string model_name = model_path;
        if (auto slash = model_name.find_last_of('/');
            slash != std::string::npos)
            model_name.erase(0, slash + 1);

        std::vector<std::unique_ptr<agent_cpp::Callback>> callbacks;
        // The session recorder goes first: it assigns ids to tool calls
        // whose chat template omitted them, and later callbacks (and the
        // conversation itself) should see those ids.
        if (!session_path.empty()) {
            callbacks.emplace_back(
                std::make_unique<agentfile::SessionRecorderCallback>(
                    session_path, model_name));
        }
        callbacks.emplace_back(
            std::make_unique<agentfile::DestructiveOpsConfirmationCallback>(
                always_yes, toolbox.write_tool_names(),
                /*audit=*/verbosity == 0));
        if (verbosity > 0) {
            callbacks.emplace_back(
                std::make_unique<agentfile::ProgressCallback>(verbosity > 1));
        }
        // The trace callback goes after the confirmation prompt so tool
        // spans measure execution, not the time the user spent deciding.
        if (!trace_path.empty()) {
            callbacks.emplace_back(
                std::make_unique<agentfile::OtlpTraceCallback>(trace_path,
                                                               model_name));
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

        // Follow-up turns reuse `messages`, so the KV-cache prefix carries
        // over and each turn only pays for what's new.
        while (interactive) {
            fflush(stdout);
            std::string followup = read_followup();
            if (followup.empty()) break;
            common_chat_msg msg;
            msg.role = "user";
            msg.content = followup;
            messages.push_back(std::move(msg));
            reply = agent.run_loop(messages);
            fputs(reply.c_str(), stdout);
            fputc('\n', stdout);
        }
    } catch (const agentfile::MaxIterationsExceeded &e) {
        fprintf(stderr, "%s\n", e.what());
        return 4;
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
