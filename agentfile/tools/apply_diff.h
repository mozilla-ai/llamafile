// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// apply_diff: vendored from llama.cpp/tools/server/server-tools.cpp:645-695.
// Requires `git` in PATH at runtime.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>

namespace agentfile {
namespace tools {

class ApplyDiffTool : public agent_cpp::Tool {
  public:
    std::string get_name() const override { return "apply_diff"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
            }},
            {"required", json::array({"diff"})},
        };
        return {"apply_diff",
                "Apply a unified diff to one or more files via `git apply`. "
                "Requires git in PATH. Prefer this over edit_file for complex "
                "changes.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        namespace fs = std::filesystem;

        if (!arguments.contains("diff")) {
            return error_response("missing required parameter: diff");
        }
        std::string diff = arguments.at("diff").get<std::string>();

        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("agentfile_patch_" + std::to_string(++counter) + ".patch")).string();
        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) return error_response("failed to create temp patch file");
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return error_response("git apply failed (exit " +
                                  std::to_string(res.exit_code) + "): " +
                                  res.output);
        }
        return result_response({{"result", "patch applied successfully"}});
    }
};

} // namespace tools
} // namespace agentfile
