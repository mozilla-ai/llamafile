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
// ErrorRecoveryCallback — hands a failed tool call (a tool the model doesn't
// have, arguments that aren't JSON) back to the model as {"error": true,
// "tool": ..., "message": ...}, so it can fix the call instead of the run
// ending. After kMaxConsecutiveFailures in a row, the next failure ends the
// run (main exits 2).
//

#pragma once

#include "callbacks.h"
#include "error.h"
#include "tool_result.h"

#include "server-common.h"  // json, safe_json_to_str

#include <string>
#include <vector>

namespace agentfile {

class ErrorRecoveryCallback : public agent_cpp::Callback {
    // Without a cap, a model that keeps calling a tool it doesn't have
    // loops until the context fills.
    static constexpr int kMaxConsecutiveFailures = 3;
    int failures_ = 0;

  public:
    void before_agent_loop(std::vector<common_chat_msg> &) override {
        failures_ = 0;
    }

    // A call whose arguments aren't JSON stays in the history, and chat
    // templates can't render it: store such arguments as a JSON string.
    void before_llm_call(std::vector<common_chat_msg> &messages) override {
        for (auto &msg : messages) {
            for (auto &tc : msg.tool_calls) {
                if (json::parse_no_throw(tc.arguments).is_discarded()) {
                    tc.arguments = safe_json_to_str(json(tc.arguments));
                }
            }
        }
    }

    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        if (!result.has_error()) {
            failures_ = 0;
            return;
        }
        if (++failures_ > kMaxConsecutiveFailures) {
            throw agent_cpp::Error("giving up after " +
                                   std::to_string(failures_) +
                                   " failed tool calls in a row; last: " +
                                   result.error().message);
        }
        json err = {
            {"error", true},
            {"tool", tool_name},
            {"message", result.error().message},
        };
        result.recover(safe_json_to_str(err));
    }
};

} // namespace agentfile
