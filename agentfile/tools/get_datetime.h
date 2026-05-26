// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// get_datetime: vendored from llama.cpp/tools/server/server-tools.cpp:701-724,
// adapted to agent.cpp's Tool interface.
//

#pragma once

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <chrono>
#include <ctime>
#include <string>

namespace agentfile {
namespace tools {

class GetDatetimeTool : public agent_cpp::Tool {
  public:
    std::string get_name() const override { return "get_datetime"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", json::object()},
        };
        return {"get_datetime",
                "Returns the current date and time.",
                schema.dump()};
    }

    std::string execute(const json & /*arguments*/) override {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        return result_response({{"result", std::ctime(&time)}});
    }
};

} // namespace tools
} // namespace agentfile
