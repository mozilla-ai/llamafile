// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// write_file: vendored from llama.cpp/tools/server/server-tools.cpp:424-473.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace agentfile {
namespace tools {

class WriteFileTool : public agent_cpp::Tool {
  public:
    std::string get_name() const override { return "write_file"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                {"content", {{"type", "string"}, {"description", "Content to write"}}},
            }},
            {"required", json::array({"path", "content"})},
        };
        return {"write_file",
                "Write content to a file, creating parent directories if needed. "
                "Overwrites if the file already exists.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        namespace fs = std::filesystem;

        if (!arguments.contains("path") || !arguments.contains("content")) {
            return error_response("missing required parameter (need path and content)");
        }
        std::string path    = arguments.at("path").get<std::string>();
        std::string content = arguments.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return error_response("failed to create directories: " + ec.message());
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) return error_response("failed to open file for writing: " + path);
        f << content;
        if (!f) return error_response("failed to write file: " + path);

        return result_response({{"result", "file written successfully"},
                                {"path", path},
                                {"bytes", content.size()}});
    }
};

} // namespace tools
} // namespace agentfile
