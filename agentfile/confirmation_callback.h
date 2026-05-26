// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// DestructiveOpsConfirmationCallback — prompts the user on stderr before
// agentfile invokes a destructive tool. Read-only tools execute silently.
// Pattern inspired by agent.cpp/examples/shell/shell.cpp's
// ShellConfirmationCallback.
//

#pragma once

#include "callbacks.h"
#include "error.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

namespace agentfile {

class DestructiveOpsConfirmationCallback : public agent_cpp::Callback {
    bool always_yes_;

    static const std::set<std::string> &destructive() {
        static const std::set<std::string> s{
            "write_file",
            "edit_file",
            "apply_diff",
            "exec_shell_command",
        };
        return s;
    }

  public:
    explicit DestructiveOpsConfirmationCallback(bool always_yes)
        : always_yes_(always_yes) {}

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        if (!destructive().count(tool_name)) return;
        if (always_yes_) {
            std::fprintf(stderr, "[%s --yes] %s\n", tool_name.c_str(),
                         arguments.c_str());
            return;
        }
        std::fprintf(stderr, "\n[%s] %s\n", tool_name.c_str(),
                     arguments.c_str());
        std::fprintf(stderr, "Allow? [y/N]: ");
        std::fflush(stderr);

        std::string line;
        if (!std::getline(std::cin, line)) {
            throw agent_cpp::ToolExecutionSkipped(
                "user cancelled (stdin closed)");
        }
        char c = line.empty() ? 'n' : (char)std::tolower((unsigned char)line[0]);
        if (c != 'y') {
            throw agent_cpp::ToolExecutionSkipped("user declined");
        }
    }
};

} // namespace agentfile
