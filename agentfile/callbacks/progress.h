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
// ProgressCallback — prints `[tool: NAME args] -> N bytes` on stderr
// around each tool execution, keeping stdout clean. In verbose mode also
// prints a truncated result preview.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include "util.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace agentfile {

class ProgressCallback : public agent_cpp::Callback {
    bool verbose_;

  public:
    explicit ProgressCallback(bool verbose = false) : verbose_(verbose) {}

    // Reasoning is commentary, not answer: show it dimmed on stderr and
    // keep stdout for the final result only.
    void after_llm_call(common_chat_msg &parsed_msg) override {
        if (!parsed_msg.reasoning_content.empty()) {
            std::fprintf(stderr, "%s[think] %s%s\n", dim(),
                         parsed_msg.reasoning_content.c_str(), dim_off());
        }
    }

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        std::fprintf(stderr, "%s[tool: %s %s]%s\n", dim(),
                     tool_name.c_str(), arguments.c_str(), dim_off());
    }

    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        if (result.has_error()) {
            std::fprintf(stderr, "%s[tool: %s -> error: %s]%s\n", dim(),
                         tool_name.c_str(),
                         result.error().message.c_str(), dim_off());
        } else {
            const auto &out = result.output();
            std::fprintf(stderr, "%s[tool: %s -> %zu bytes]%s\n", dim(),
                         tool_name.c_str(), out.size(), dim_off());
            if (verbose_ && !out.empty()) {
                constexpr size_t kPreview = 512;
                std::fprintf(stderr, "%s  %.*s%s%s\n", dim(),
                             (int)std::min(out.size(), kPreview), out.c_str(),
                             out.size() > kPreview ? "…" : "", dim_off());
            }
        }
    }
};

} // namespace agentfile
