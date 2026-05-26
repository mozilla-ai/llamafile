// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// file_glob_search: vendored from llama.cpp/tools/server/server-tools.cpp:206-262,
// adapted to agent.cpp's Tool interface.
//

#pragma once

#include "chat.h"
#include "common.h"   // for glob_match
#include "tool.h"

#include "tools_common.h"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>

namespace agentfile {
namespace tools {

class FileGlobSearchTool : public agent_cpp::Tool {
    static constexpr size_t kMaxResults = 100;

  public:
    std::string get_name() const override { return "file_glob_search"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
            }},
            {"required", json::array({"path"})},
        };
        return {"file_glob_search",
                "Recursively search for files matching a glob pattern under a "
                "directory.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        namespace fs = std::filesystem;

        if (!arguments.contains("path")) {
            return error_response("missing required parameter: path");
        }
        std::string base    = arguments.at("path").get<std::string>();
        std::string include = json_value(arguments, "include", std::string("**"));
        std::string exclude = json_value(arguments, "exclude", std::string(""));

        std::ostringstream output;
        size_t count = 0;
        std::error_code ec;

        for (const auto &entry : fs::recursive_directory_iterator(
                 base, fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!glob_match(include, rel)) continue;
            if (!exclude.empty() && glob_match(exclude, rel)) continue;

            output << entry.path().string() << "\n";
            if (++count >= kMaxResults) break;
        }

        output << "\n---\nTotal matches: " << count << "\n";
        return plain_text_response(output.str());
    }
};

} // namespace tools
} // namespace agentfile
