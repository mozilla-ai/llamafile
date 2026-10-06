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
// DestructiveOpsConfirmationCallback — asks "Allow? [y/N]" before each call
// to a guarded tool (writes, shell, network). --yes skips the question;
// with --quiet as well, a one-line record is printed instead, so a guarded
// call never runs without a trace.
//

#pragma once

#include "callbacks.h"
#include "error.h"

#include "server_tools_adapter.h"  // printable_arguments
#include "util.h"

#include <cctype>
#include <cstdio>
#include <set>
#include <string>

namespace agentfile {

// No terminal to ask on (CI, cron) or EOF at the prompt: every later call
// would go unanswered too, so the run ends (main exits 2).
class ConfirmationUnavailable : public agent_cpp::Error {
  public:
    explicit ConfirmationUnavailable(const std::string &tool_name)
        : agent_cpp::Error("cannot confirm " + tool_name +
                           ": no answer from a terminal (use --yes for "
                           "unattended runs)") {}
};

class DestructiveOpsConfirmationCallback : public agent_cpp::Callback {
    // A model can retry a declined call again and again; this many declines
    // in a row end the run.
    static constexpr int kMaxDeclines = 3;

    bool always_yes_;
    bool audit_;                       // print a record under --yes
    std::set<std::string> destructive_;  // tools with permission_write
    int declines_ = 0;                 // in a row, this turn

  public:
    DestructiveOpsConfirmationCallback(bool always_yes,
                                       std::set<std::string> destructive,
                                       bool audit = false)
        : always_yes_(always_yes), audit_(audit),
          destructive_(std::move(destructive)) {}

    void before_agent_loop(std::vector<common_chat_msg> &) override {
        declines_ = 0;
    }

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        if (!destructive_.count(tool_name)) return;
        std::string args = printable_arguments(arguments);
        if (always_yes_) {
            if (audit_) {
                std::fprintf(stderr, "[%s --yes] %s\n", tool_name.c_str(),
                             args.c_str());
            }
            return;
        }
        std::string answer;
        if (!ask_terminal("\n[" + tool_name + "] " + args +
                              "\nAllow? [y/N]: ",
                          answer, /*discard_typeahead=*/true)) {
            throw ConfirmationUnavailable(tool_name);
        }
        if (answer.empty() || std::tolower((unsigned char)answer[0]) != 'y') {
            if (++declines_ >= kMaxDeclines) {
                throw agent_cpp::Error(std::to_string(declines_) +
                                       " tool calls declined in a row");
            }
            throw agent_cpp::ToolExecutionSkipped(
                "declined by the user; do not retry this call");
        }
        declines_ = 0;
    }
};

} // namespace agentfile
