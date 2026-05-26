// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Helpers shared by the vendored llama.cpp server-tools (re-shaped to
// agent.cpp's Tool interface).
//

#pragma once

#include <sheredom/subprocess.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

namespace agentfile {
namespace tools {

using json = nlohmann::json;  // match agent_cpp::Tool::execute signature

// json_value: read body[key], fall back to default if missing/null or wrong type.
// Lifted from llama.cpp/tools/server/server-common.h:33.
template <typename T>
inline T json_value(const json &body, const std::string &key,
                    const T &default_value) {
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// Wrap a plain text payload in the {"plain_text_response": "..."} shape that
// llama.cpp's server-tools use, and serialize to a JSON string suitable for
// agent.cpp's Tool::execute return value.
inline std::string plain_text_response(const std::string &text) {
    return json{{"plain_text_response", text}}.dump();
}

// Wrap an error message in {"error": "..."} and serialize.
inline std::string error_response(const std::string &msg) {
    return json{{"error", msg}}.dump();
}

// Wrap a structured result (e.g. {"result": "...", "path": "...", "bytes": N}).
inline std::string result_response(const json &payload) { return payload.dump(); }

// Run a subprocess and collect (stdout+stderr) up to max_output bytes.
// Caps execution at timeout_secs seconds.
// Vendored from llama.cpp/tools/server/server-tools.cpp:37-103.
struct run_proc_result {
    std::string output;
    int exit_code = -1;
    bool timed_out = false;
};

inline run_proc_result run_process(const std::vector<std::string> &args,
                                   size_t max_output, int timeout_secs) {
    run_proc_result res;

    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto &s : args) argv.push_back(const_cast<char *>(s.c_str()));
    argv.push_back(nullptr);

    subprocess_s proc;
    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;
    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};
    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE *f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (std::fgets(buf, sizeof(buf), f) != nullptr) {
            if (truncated) continue;
            size_t len = std::strlen(buf);
            if (output.size() + len <= max_output) {
                output.append(buf, len);
            } else {
                output.append(buf, max_output - output.size());
                truncated = true;
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) timeout_thread.join();

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output = output;
    res.timed_out = timed_out.load();
    if (truncated) res.output += "\n[output truncated]";
    return res;
}

} // namespace tools
} // namespace agentfile
