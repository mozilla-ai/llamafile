// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// exec_shell_command: vendored from llama.cpp/tools/server/server-tools.cpp:368-418.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace agentfile {
namespace tools {

class ExecShellCommandTool : public agent_cpp::Tool {
    static constexpr size_t kMaxOutput  = 16 * 1024;  // 16 KB
    static constexpr int    kMaxTimeout = 60;          // seconds

  public:
    std::string get_name() const override { return "exec_shell_command"; }

    common_chat_tool get_definition() const override {
        char timeout_desc[96];
        std::snprintf(timeout_desc, sizeof(timeout_desc),
                      "Timeout in seconds (default 10, max %d)", kMaxTimeout);
        char out_desc[96];
        std::snprintf(out_desc, sizeof(out_desc),
                      "Maximum output size in bytes (default %zu)", kMaxOutput);

        json schema = {
            {"type", "object"},
            {"properties", {
                {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                {"timeout",         {{"type", "integer"}, {"description", timeout_desc}}},
                {"max_output_size", {{"type", "integer"}, {"description", out_desc}}},
            }},
            {"required", json::array({"command"})},
        };
        return {"exec_shell_command",
                "Execute a shell command and return its output (stdout and "
                "stderr combined).",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        if (!arguments.contains("command")) {
            return error_response("missing required parameter: command");
        }
        std::string command = arguments.at("command").get<std::string>();
        int timeout         = json_value(arguments, "timeout", 10);
        size_t max_output   = (size_t)json_value(arguments, "max_output_size",
                                                 (int)kMaxOutput);
        timeout    = std::min(timeout,    kMaxTimeout);
        max_output = std::min(max_output, kMaxOutput);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text = res.output;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "\n[exit code: %d]", res.exit_code);
        text += buf;
        if (res.timed_out) text += " [exit due to timed out]";

        return plain_text_response(text);
    }
};

} // namespace tools
} // namespace agentfile
