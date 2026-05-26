// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// grep_search: vendored from llama.cpp/tools/server/server-tools.cpp:270-359,
// adapted to agent.cpp's Tool interface.
//

#pragma once

#include "chat.h"
#include "common.h"  // for glob_match
#include "tool.h"

#include "tools_common.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace agentfile {
namespace tools {

class GrepSearchTool : public agent_cpp::Tool {
    static constexpr size_t kMaxResults = 100;

  public:
    std::string get_name() const override { return "grep_search"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
            }},
            {"required", json::array({"path", "pattern"})},
        };
        return {"grep_search",
                "Search for a regex pattern in files under a path. Returns "
                "matching lines.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        namespace fs = std::filesystem;

        if (!arguments.contains("path") || !arguments.contains("pattern")) {
            return error_response("missing required parameter (need path and pattern)");
        }
        std::string path    = arguments.at("path").get<std::string>();
        std::string pat_str = arguments.at("pattern").get<std::string>();
        std::string include = json_value(arguments, "include", std::string("**"));
        std::string exclude = json_value(arguments, "exclude", std::string(""));
        bool show_lineno    = json_value(arguments, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error &e) {
            return error_response(std::string("invalid regex: ") + e.what());
        }

        std::ostringstream output;
        size_t total = 0;

        auto search_file = [&](const fs::path &fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < kMaxResults) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output << fpath.string() << ":";
                    if (show_lineno) output << lineno << ":";
                    output << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto &entry : fs::recursive_directory_iterator(
                     path, fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= kMaxResults) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return error_response("path does not exist: " + path);
        }

        output << "\n\n---\nTotal matches: " << total << "\n";
        return plain_text_response(output.str());
    }
};

} // namespace tools
} // namespace agentfile
