// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// read_file: vendored from llama.cpp/tools/server/server-tools.cpp:123-198,
// adapted to agent.cpp's Tool interface.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace agentfile {
namespace tools {

class ReadFileTool : public agent_cpp::Tool {
    static constexpr size_t kMaxSize = 16 * 1024;  // 16 KB

  public:
    std::string get_name() const override { return "read_file"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
            }},
            {"required", json::array({"path"})},
        };
        return {"read_file",
                "Read the contents of a file. Optionally specify a 1-based line "
                "range. If append_loc is true, each line is prefixed with its line "
                "number.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        namespace fs = std::filesystem;

        if (!arguments.contains("path")) {
            return error_response("missing required parameter: path");
        }
        std::string path  = arguments.at("path").get<std::string>();
        int start_line    = json_value(arguments, "start_line", 1);
        int end_line      = json_value(arguments, "end_line", -1);
        bool append_loc   = json_value(arguments, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return error_response("cannot stat file: " + ec.message());
        }
        if (file_size > kMaxSize && end_line == -1) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "file too large (%zu bytes, max %zu). Use start_line/end_line.",
                          (size_t)file_size, (size_t)kMaxSize);
            return error_response(buf);
        }

        std::ifstream f(path);
        if (!f) {
            return error_response("failed to open file: " + path);
        }

        std::string result;
        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\xe2\x86\x92 " + line + "\n";
            } else {
                out_line = line + "\n";
            }
            if (result.size() + out_line.size() > kMaxSize) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return plain_text_response(result);
    }
};

} // namespace tools
} // namespace agentfile
