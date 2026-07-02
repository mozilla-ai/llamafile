// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Convenience header: builds a vector of all v0 read-only tools.
// Destructive tools (write_file, edit_file, apply_diff, exec_shell_command)
// will be added in Step 5 alongside the confirmation callback.
//

#pragma once

#include "tool.h"

#include "apply_diff.h"
#include "edit_file.h"
#include "exec_shell_command.h"
#include "file_glob_search.h"
#include "get_datetime.h"
#include "grep_search.h"
#include "http_fetch.h"
#include "read_file.h"
#include "web_search.h"
#include "write_file.h"

#include <memory>
#include <vector>

namespace agentfile {
namespace tools {

// Build the read-only tool set (safe to execute silently).
inline std::vector<std::unique_ptr<agent_cpp::Tool>>
build_readonly_tools() {
    std::vector<std::unique_ptr<agent_cpp::Tool>> tools;
    tools.emplace_back(std::make_unique<ReadFileTool>());
    tools.emplace_back(std::make_unique<FileGlobSearchTool>());
    tools.emplace_back(std::make_unique<GrepSearchTool>());
    tools.emplace_back(std::make_unique<GetDatetimeTool>());
    return tools;
}

// Build the full default v0 tool set: read-only + destructive.
// Pair with DestructiveOpsConfirmationCallback for safety prompts.
// web_search is only registered when a SearXNG base URL is configured
// (--searxng-url / SEARXNG_URL) — the model never picks the instance.
inline std::vector<std::unique_ptr<agent_cpp::Tool>>
build_default_tools(const std::string &searxng_url = "") {
    auto tools = build_readonly_tools();
    tools.emplace_back(std::make_unique<WriteFileTool>());
    tools.emplace_back(std::make_unique<EditFileTool>());
    tools.emplace_back(std::make_unique<ApplyDiffTool>());
    tools.emplace_back(std::make_unique<ExecShellCommandTool>());
    tools.emplace_back(std::make_unique<HttpFetchTool>());
    if (!searxng_url.empty()) {
        tools.emplace_back(std::make_unique<WebSearchTool>(searxng_url));
    }
    return tools;
}

} // namespace tools
} // namespace agentfile
