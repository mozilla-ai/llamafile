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
// ErrorRecoveryCallback — hands a failed tool call back to the model as
// {"error": true, "tool": ..., "message": ...}, so it can fix the call
// (bad arguments, a tool it doesn't have) instead of run_loop ending the
// run. Same result shape as agent.cpp's
// examples/shared/error_recovery_callback.h, serialized with
// safe_json_to_str: the message can quote invalid UTF-8 from the model's
// arguments.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include "server-common.h"  // safe_json_to_str

#include <nlohmann/json.hpp>
#include <string>

namespace agentfile {

class ErrorRecoveryCallback : public agent_cpp::Callback {
  public:
    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        if (!result.has_error()) return;
        nlohmann::ordered_json err = {
            {"error", true},
            {"tool", tool_name},
            {"message", result.error().message},
        };
        result.recover(safe_json_to_str(err));
    }
};

} // namespace agentfile
