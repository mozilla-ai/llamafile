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
// DestructiveOpsConfirmationCallback — prompts the user on stderr before
// agentfile invokes a destructive tool. Read-only tools execute silently.
// Under --yes the prompt is skipped; when --quiet has also silenced the
// progress output, a one-line audit record is printed instead, so a
// write, shell command or network call never runs without a trace.
// Pattern inspired by agent.cpp/examples/shell/shell.cpp's
// ShellConfirmationCallback.
//

#pragma once

#include "callbacks.h"
#include "error.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>

namespace agentfile {

// The arguments are model-controlled text headed for the terminal. Show
// them re-serialized: that is what the tool will receive, and it cannot
// carry raw control bytes (a '\r' between JSON tokens would otherwise
// return the cursor and overprint the command being approved). Text that
// is not valid JSON, and so will be refused anyway, gets its control
// bytes escaped.
inline std::string printable_arguments(const std::string &arguments) {
    try {
        return nlohmann::json::parse(arguments).dump(
            -1, ' ', false, nlohmann::json::error_handler_t::replace);
    } catch (const std::exception &) {
    }
    std::string out;
    for (unsigned char c : arguments) {
        if (c < 0x20 || c == 0x7f) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
        } else {
            out += (char)c;
        }
    }
    return out;
}

class DestructiveOpsConfirmationCallback : public agent_cpp::Callback {
    bool always_yes_;
    // Log destructive calls under --yes when nothing else will (--quiet
    // disables the ProgressCallback that normally prints every call).
    bool audit_;
    // Tool names requiring confirmation. Comes from the tools' own
    // permission_write metadata (writes, shell, network access).
    std::set<std::string> destructive_;

  public:
    DestructiveOpsConfirmationCallback(bool always_yes,
                                       std::set<std::string> destructive,
                                       bool audit = false)
        : always_yes_(always_yes), audit_(audit),
          destructive_(std::move(destructive)) {}

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        if (!destructive_.count(tool_name)) return;
        if (always_yes_) {
            if (audit_) {
                std::fprintf(stderr, "[%s --yes] %s\n", tool_name.c_str(),
                             printable_arguments(arguments).c_str());
            }
            return;
        }
        std::fprintf(stderr, "\n[%s] %s\n", tool_name.c_str(),
                     printable_arguments(arguments).c_str());
        std::fprintf(stderr, "Allow? [y/N]: ");
        std::fflush(stderr);

        // When the prompt was piped in, stdin is consumed/EOF — ask on the
        // controlling terminal instead. No terminal at all (CI, cron) means
        // the answer is a decline; use --yes for unattended runs.
        std::string line;
        bool got = false;
        if (isatty(STDIN_FILENO)) {
            got = (bool)std::getline(std::cin, line);
        } else if (FILE *tty = std::fopen("/dev/tty", "r")) {
            char buf[256];
            if (std::fgets(buf, sizeof(buf), tty)) {
                line = buf;
                got = true;
            }
            std::fclose(tty);
        }
        if (!got) {
            throw agent_cpp::ToolExecutionSkipped(
                "user cancelled (no terminal to confirm on; use --yes for "
                "unattended runs)");
        }
        char c = line.empty() ? 'n' : (char)std::tolower((unsigned char)line[0]);
        if (c != 'y') {
            throw agent_cpp::ToolExecutionSkipped("user declined");
        }
    }
};

} // namespace agentfile
