// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// edit_file: vendored from llama.cpp/tools/server/server-tools.cpp:479-639.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace agentfile {
namespace tools {

class EditFileTool : public agent_cpp::Tool {
  public:
    std::string get_name() const override { return "edit_file"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                {"changes", {
                    {"type", "array"},
                    {"description", "List of changes to apply"},
                    {"items", {
                        {"type", "object"},
                        {"properties", {
                            {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                            {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                            {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                            {"content",    {{"type", "string"},  {"description", "Content to insert; empty string for delete mode"}}},
                        }},
                        {"required", json::array({"mode", "line_start", "line_end", "content"})},
                    }},
                }},
            }},
            {"required", json::array({"path", "changes"})},
        };
        return {"edit_file",
                "Edit a file by applying a list of line-based changes. Each "
                "change targets a 1-based inclusive line range and has a mode: "
                "\"replace\", \"delete\", or \"append\". Set line_start to -1 "
                "to target the end of file.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        if (!arguments.contains("path") || !arguments.contains("changes")) {
            return error_response("missing required parameter (need path and changes)");
        }
        std::string path     = arguments.at("path").get<std::string>();
        const json &changes  = arguments.at("changes");
        if (!changes.is_array()) {
            return error_response("\"changes\" must be an array");
        }

        std::ifstream fin(path);
        if (!fin) return error_response("failed to open file: " + path);
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) lines.push_back(line);
        }
        fin.close();

        struct change_entry {
            std::string mode;
            int line_start;
            int line_end;
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto &ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return error_response("invalid mode \"" + e.mode +
                                       "\"; must be replace, delete, or append");
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return error_response("content must be empty string for delete mode");
            }
            int n = (int)lines.size();
            if (e.line_start == -1) {
                e.line_start = n + 1;
                e.line_end   = n + 1;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "invalid line range [%d, %d]",
                                  e.line_start, e.line_end);
                    return error_response(buf);
                }
                if (e.line_end > n) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf),
                                  "line_end %d exceeds file length %d",
                                  e.line_end, n);
                    return error_response(buf);
                }
            }
            entries.push_back(std::move(e));
        }

        std::sort(entries.begin(), entries.end(),
                  [](const change_entry &a, const change_entry &b) {
                      return a.line_start > b.line_start;
                  });

        for (const auto &e : entries) {
            int idx_start = e.line_start - 1;
            int idx_end   = e.line_end   - 1;

            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) new_lines.push_back(ln);
            }

            if (e.mode == "replace") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                lines.insert(lines.begin() + idx_end + 1, new_lines.begin(), new_lines.end());
            }
        }

        std::ofstream fout(path, std::ios::binary);
        if (!fout) return error_response("failed to open file for writing: " + path);
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) fout << "\n";
        }
        if (!lines.empty()) fout << "\n";
        if (!fout) return error_response("failed to write file: " + path);

        return result_response({{"result", "file edited successfully"},
                                {"path", path},
                                {"lines", (int)lines.size()}});
    }
};

} // namespace tools
} // namespace agentfile
