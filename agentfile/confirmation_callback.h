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
#include <unistd.h>

namespace agentfile {

class DestructiveOpsConfirmationCallback : public agent_cpp::Callback {
    bool always_yes_;
    // Tool names requiring confirmation. Comes from the tools' own
    // permission_write metadata (writes, shell, network access).
    std::set<std::string> destructive_;

  public:
    DestructiveOpsConfirmationCallback(bool always_yes,
                                       std::set<std::string> destructive)
        : always_yes_(always_yes), destructive_(std::move(destructive)) {}

    void before_tool_execution(std::string &tool_name,
                               std::string &arguments) override {
        if (!destructive_.count(tool_name)) return;
        if (always_yes_) {
            std::fprintf(stderr, "[%s --yes] %s\n", tool_name.c_str(),
                         arguments.c_str());
            return;
        }
        std::fprintf(stderr, "\n[%s] %s\n", tool_name.c_str(),
                     arguments.c_str());
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
