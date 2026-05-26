// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Two helper callbacks for agentfile:
//
//   ProgressCallback — prints `[tool: NAME args] -> NB bytes` on stderr
//                       around each tool execution, keeping stdout clean.
//
//   MaxIterationsCallback — throws std::runtime_error when the agent loop
//                            has produced N llm calls without terminating.
//                            Pair with --max-iterations N.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace agentfile {

class ProgressCallback : public agent_cpp::Callback {
  public:
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
        }
    }
};

class MaxIterationsCallback : public agent_cpp::Callback {
    int max_;
    int count_ = 0;

  public:
    explicit MaxIterationsCallback(int max) : max_(max) {}

    void before_llm_call(std::vector<common_chat_msg> & /*messages*/) override {
        if (max_ > 0 && ++count_ > max_) {
            throw std::runtime_error(
                "agentfile: agent loop exceeded --max-iterations cap");
        }
    }
};

} // namespace agentfile
