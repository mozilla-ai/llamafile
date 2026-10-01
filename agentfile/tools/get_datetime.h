// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// get_datetime: the current date and time in UTC. llama.cpp moved this
// tool out of the server into its web UI (#27255), where the client's
// clock lives; for agentfile the CLI is the client, so it keeps the tool.
// Same definition and output as the server tool it replaces.
//

#pragma once

#include "server-tools.h"

#include <chrono>
#include <ctime>
#include <string>

namespace agentfile {
namespace tools {

struct GetDatetimeTool : server_tool {
    GetDatetimeTool() {
        name = "get_datetime";
        display_name = "Get Date & Time";
        permission_write = false;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Returns the current date and time in UTC"},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"format", {
                            {"type", "string"},
                            {"description",
                             "strftime()-style format string for the output "
                             "(default: \"%Y-%m-%dT%H:%M:%SZ\", e.g. ISO "
                             "8601). Choose your own format if you need "
                             "something else, e.g. \"%A, %B %d %Y\" for a "
                             "human-readable date."},
                        }},
                    }},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        std::string format =
            json_value(params, "format", std::string("%Y-%m-%dT%H:%M:%SZ"));

        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_utc;
        gmtime_r(&time, &tm_utc);

        char buf[256];
        size_t len = std::strftime(buf, sizeof(buf), format.c_str(), &tm_utc);
        if (len == 0) {
            return {{"error", "invalid format string"}};
        }
        return {{"result", std::string(buf, len)}};
    }
};

} // namespace tools
} // namespace agentfile
