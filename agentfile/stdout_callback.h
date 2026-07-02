// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Two helper callbacks for agentfile:
//
//   ProgressCallback — prints `[tool: NAME args] -> NB bytes` on stderr
//                       around each tool execution, keeping stdout clean.
//                       In verbose mode also prints a truncated result
//                       preview.
//
//   MaxIterationsCallback — throws MaxIterationsExceeded when the agent
//                            loop has produced N llm calls without
//                            terminating. Pair with --max-iterations N;
//                            main maps it to a distinct exit code.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace agentfile {

class ProgressCallback : public agent_cpp::Callback {
    bool verbose_;

  public:
    explicit ProgressCallback(bool verbose = false) : verbose_(verbose) {}

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        std::fprintf(stderr, "[tool: %s %s]\n",
                     tool_name.c_str(), arguments.c_str());
    }

    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        if (result.has_error()) {
            std::fprintf(stderr, "[tool: %s -> error: %s]\n",
                         tool_name.c_str(),
                         result.error().message.c_str());
        } else {
            const auto &out = result.output();
            std::fprintf(stderr, "[tool: %s -> %zu bytes]\n",
                         tool_name.c_str(), out.size());
            if (verbose_ && !out.empty()) {
                constexpr size_t kPreview = 512;
                std::fprintf(stderr, "  %.*s%s\n",
                             (int)std::min(out.size(), kPreview), out.c_str(),
                             out.size() > kPreview ? "…" : "");
            }
        }
    }
};

// Distinct type so main can map the iteration cap to its own exit code.
class MaxIterationsExceeded : public std::runtime_error {
  public:
    MaxIterationsExceeded()
        : std::runtime_error(
              "agentfile: agent loop exceeded --max-iterations cap") {}
};

class MaxIterationsCallback : public agent_cpp::Callback {
    int max_;
    int count_ = 0;

  public:
    explicit MaxIterationsCallback(int max) : max_(max) {}

    void before_llm_call(std::vector<common_chat_msg> & /*messages*/) override {
        if (max_ > 0 && ++count_ > max_) {
            throw MaxIterationsExceeded();
        }
    }
};

} // namespace agentfile
