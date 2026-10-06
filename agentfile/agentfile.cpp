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

#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "llama.h"
#include "chat.h"
#include "common.h"
#include "log.h"

#include "agent.h"
#include "callbacks.h"
#include "error.h"
#include "model.h"
#include "tool.h"

#include "llamafile.h"

#include "callbacks/confirmation.h"
#include "callbacks/error_recovery.h"
#include "callbacks/max_iterations.h"
#include "callbacks/progress.h"
#include "callbacks/session_recorder.h"
#include "callbacks/trace.h"
#include "server_tools_adapter.h"

// Last, as in llama-server's server.cpp: cosmo.h defines a defer() macro
// that breaks server_queue::defer() in server-queue.h.
#ifdef COSMOCC
#include <cosmo.h>
#endif

namespace {

// Default context window: room for a few full http_fetch results (64 KB,
// roughly 16-24k tokens each) plus the rest of the conversation.
constexpr int kDefaultCtx = 64 * 1024;

// Lowest llama.cpp / ggml log level that reaches stderr: -v shows warnings
// and errors, -vv also info. Above GGML_LOG_LEVEL_ERROR (the default) is
// silent.
int g_llama_log_min = GGML_LOG_LEVEL_ERROR + 1;

// For llama_log_set. CONT carries on the previous message (a line printed
// in pieces, the loading dots), so it takes that message's level. GGUF
// metadata strings pass through here, so the text is escaped.
void llama_log_to_stderr(ggml_log_level level, const char *text, void *) {
    static thread_local ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level == GGML_LOG_LEVEL_CONT)
        level = last;
    else
        last = level;
    if (level >= g_llama_log_min && level <= GGML_LOG_LEVEL_ERROR)
        fputs(agentfile::terminal_text(text, /*keep_newlines=*/true).c_str(),
              stderr);
}

// The tools part of the help comes from the toolbox, so it always matches
// the registered tools. searxng_url is the value parsed so far, so
// web_search shows as available when it is.
void print_tools_help(const std::string &searxng_url) {
    try {
        agentfile::ServerToolbox toolbox(searxng_url, "");
        auto names = toolbox.tool_names();
        auto guarded = toolbox.write_tool_names(names);
        std::string ro, wr;
        for (const auto &name : names) {
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
                "  --tools-runtime SPEC Run tools in an isolate, as llama-server does\n"
                "                       (default: this host): \"docker-container:ID\"\n"
                "                       (a running container), \"docker:IMAGE\" (started\n"
                "                       here, stopped on exit; podman likewise) or\n"
                "                       \"ssh:TARGET\". http_fetch and web_search still\n"
                "                       connect from this host.\n"
                "  Tools run with this process's permissions; agentfile does not\n"
                "  sandbox itself. The confirmation prompt is the only gate (--yes\n"
                "  removes it); use --tools-runtime for isolation.\n");
    } catch (const std::exception &e) {
        fprintf(stderr, "Tools: unavailable in this build (%s)\n", e.what());
    }
}

void print_usage(const char *prog, const std::string &searxng_url) {
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
            "  -c, --ctx-size N     Context window in tokens (default: %d), never\n"
            "                       more than the model's maximum. 0 = the model's\n"
            "                       maximum — mind the KV-cache memory on\n"
            "                       long-context models\n"
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
            "  -v, --verbose        Also print truncated tool results, and\n"
            "                       llama.cpp warnings and errors (such as why a\n"
            "                       model fails to load) to stderr; chat parser\n"
            "                       warnings come unescaped\n"
            "  -vv                  Like -v, plus llama.cpp's info log (model\n"
            "                       metadata, GPU setup)\n"
            "  -h                   Show this help\n"
            "\n",
            prog, prog, kDefaultCtx);
    print_tools_help(searxng_url);
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

// --tools: "all", "read_only" (every tool without permission_write), or a
// comma-separated list of names, which main checks against the toolbox.
std::set<std::string> parse_tools_flag(const std::string &spec,
                                       const agentfile::ServerToolbox &toolbox) {
    std::set<std::string> names;
    if (spec == "all" || spec == "read_only") {
        names = toolbox.tool_names();
        if (spec == "read_only") {
            for (const auto &name : toolbox.write_tool_names(names))
                names.erase(name);
        }
        return names;
    }
    for (const auto &item : string_split<std::string>(spec, ',')) {
        std::string name = string_strip(item);
        if (!name.empty()) names.insert(name);
    }
    return names;
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
    int n_ctx = kDefaultCtx; // tokens; 0 = the model's maximum
    int verbosity = 1;       // 0 = --quiet, 1 = default, 2 = -v, 3 = -vv
    int max_iterations = 0;  // 0 = no cap
    std::string tools_spec = "all";
    std::string session_path;
    std::string trace_path;
    std::string searxng_url;
    std::string tools_runtime;
    if (const char *env = std::getenv("SEARXNG_URL")) searxng_url = env;

    // The last flag wins and every mode flag has an inverse, so a packaged
    // agent's defaults (/zip/.args, which cosmo_args puts first) can always
    // be overridden.
    auto need_value = [&](int &i) -> const char * {
        if (i + 1 >= argc) {
            fprintf(stderr, "agentfile: %s requires a value\n", argv[i]);
            exit(1);
        }
        return argv[++i];
    };
    // 0 means "no limit" for -c and --max-iterations, so a typo must fail
    // instead of becoming 0, as it would with atoi().
    auto need_count = [&](int &i) -> int {
        const char *flag = argv[i];
        const char *s = need_value(i);
        char *end;
        errno = 0;
        long v = std::strtol(s, &end, 10);
        if (end == s || *end || errno || v < 0 || v > INT_MAX) {
            fprintf(stderr,
                    "agentfile: %s: expected a non-negative integer, got "
                    "\"%s\"\n",
                    flag, s);
            exit(1);
        }
        return static_cast<int>(v);
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = need_value(i);
        } else if (std::strcmp(argv[i], "-p") == 0) {
            prompt = need_value(i);
        } else if (std::strcmp(argv[i], "-c") == 0 ||
                   std::strcmp(argv[i], "--ctx-size") == 0) {
            n_ctx = need_count(i);
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
        } else if (std::strcmp(argv[i], "-vv") == 0) {
            verbosity = 3;
        } else if (std::strcmp(argv[i], "--max-iterations") == 0) {
            max_iterations = need_count(i);
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0], searxng_url);
            return 0;
        } else {
            fprintf(stderr, "agentfile: unknown argument: %s\n\n", argv[i]);
            print_usage(argv[0], searxng_url);
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
        print_usage(argv[0], searxng_url);
        return 1;
    }

    // Logging, set before the backends start. llama.cpp's own log: silent
    // unless -v (warnings, errors) or -vv (also info).
    if (verbosity >= 2) {
        g_llama_log_min =
            verbosity >= 3 ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_WARN;
    }
    llama_log_set(llama_log_to_stderr, nullptr);
    // The GPU backends are separate libraries with their own logger. Their
    // callback runs as native code, where only llamafile's no-op is safe,
    // so they are either silent or (at -vv) print everything themselves.
    if (verbosity < 3) {
        llamafile_metal_log_set(llamafile_log_callback_null, nullptr);
        llamafile_cuda_log_set(llamafile_log_callback_null, nullptr);
        llamafile_vulkan_log_set(llamafile_log_callback_null, nullptr);
    }
    // common_log (llama.cpp common, agent.cpp) prints model text unescaped,
    // so only its errors show unless -v.
    if (verbosity < 2)
        common_log_set_verbosity_thold(LOG_LEVEL_ERROR);

    try {
        // Whatever a bad flag can break is set up before the slow part (GPU
        // init, model load): the toolset first, then the record files. A
        // requested tool that isn't available is an error, not a smaller
        // toolset the model would improvise with.
        agentfile::ServerToolbox toolbox(searxng_url, tools_runtime);
        auto keep = parse_tools_flag(tools_spec, toolbox);
        if (keep.empty()) {
            fprintf(stderr, "agentfile: --tools: no tool names in \"%s\"\n",
                    tools_spec.c_str());
            return 1;
        }
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

        // Model name for session/trace records: the GGUF basename.
        std::string model_name = model_path;
        if (auto slash = model_name.find_last_of('/');
            slash != std::string::npos)
            model_name.erase(0, slash + 1);

        // Order matters: the trace goes after the confirmation (tool spans
        // time the tool, not the user deciding) and after the iteration cap
        // (a refused call opens no span); error recovery goes last, so the
        // others record a failed call as failed.
        std::vector<std::unique_ptr<agent_cpp::Callback>> callbacks;
        if (!session_path.empty()) {
            callbacks.emplace_back(
                std::make_unique<agentfile::SessionRecorderCallback>(
                    session_path, model_name));
        }
        callbacks.emplace_back(
            std::make_unique<agentfile::DestructiveOpsConfirmationCallback>(
                always_yes, toolbox.write_tool_names(keep),
                /*audit=*/verbosity == 0));
        if (verbosity > 0) {
            callbacks.emplace_back(
                std::make_unique<agentfile::ProgressCallback>(verbosity > 1));
        }
        if (max_iterations > 0) {
            callbacks.emplace_back(
                std::make_unique<agentfile::MaxIterationsCallback>(
                    max_iterations));
        }
        if (!trace_path.empty()) {
            callbacks.emplace_back(
                std::make_unique<agentfile::OtlpTraceCallback>(trace_path,
                                                               model_name));
        }
        callbacks.emplace_back(
            std::make_unique<agentfile::ErrorRecoveryCallback>());

        // Initialize llamafile GPU backends (Metal, CUDA, Vulkan, ROCm),
        // which also registers them.
        llamafile_has_gpu();
        llama_backend_init();

        auto weights = agent_cpp::ModelWeights::create(model_path);
        agent_cpp::ModelConfig cfg;  // defaults: temp=0, top_p=1, top_k=0
        // agent.cpp's default (-1) wraps around in llama_context's unsigned
        // n_batch and breaks context creation. 2048 is llama.cpp's default.
        cfg.n_batch = 2048;
        // The context never exceeds what the model was trained on; -c 0
        // asks for exactly that.
        int n_ctx_train = llama_model_n_ctx_train(weights->get_model());
        if (n_ctx_train > 0 && (n_ctx == 0 || n_ctx > n_ctx_train))
            n_ctx = n_ctx_train;
        cfg.n_ctx = n_ctx;
        cfg.enable_thinking = think;
        auto model = agent_cpp::Model::create_with_weights(weights, cfg);

        agent_cpp::Agent agent(std::move(model), std::move(tools),
                               std::move(callbacks), instructions);

        std::vector<common_chat_msg> messages;
        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = prompt;
        messages.push_back(std::move(user_msg));

        // The answer goes to stdout as is when piped, and escaped on a
        // terminal, where it could otherwise rewrite the lines above it.
        const bool stdout_tty = isatty(STDOUT_FILENO);
        auto print_reply = [&](const std::string &reply) {
            std::string out = stdout_tty
                ? agentfile::terminal_text(reply, /*keep_newlines=*/true)
                : reply;
            out += '\n';
            fwrite(out.data(), 1, out.size(), stdout);
        };

        print_reply(agent.run_loop(messages));

        // Follow-up turns reuse `messages`, so the cached prefix carries
        // over. An empty line, EOF or no terminal ends the session.
        while (interactive) {
            fflush(stdout);
            std::string followup;
            if (!agentfile::ask_terminal(std::string(agentfile::dim()) +
                                             "\n> " + agentfile::dim_off(),
                                         followup) ||
                followup.empty())
                break;
            common_chat_msg msg;
            msg.role = "user";
            msg.content = followup;
            messages.push_back(std::move(msg));
            print_reply(agent.run_loop(messages));
        }
    } catch (const agentfile::MaxIterationsExceeded &e) {
        fprintf(stderr, "%s\n", e.what());
        return 4;
    } catch (const agent_cpp::Error &e) {
        // Error text can quote the model or a tool (an unknown tool name,
        // a tool's error), so it is escaped like the other stderr lines.
        fprintf(stderr, "agentfile error: %s\n",
                agentfile::terminal_text(e.what()).c_str());
        return 2;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n",
                agentfile::terminal_text(e.what()).c_str());
        return 3;
    }

    llama_backend_free();
    return 0;
}
