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
// SessionRecorderCallback — records the conversation in pi's session format
// (https://pi.dev, pi/packages/coding-agent/docs/session-format.md, v3): a
// header line, then one line per message. The history is linear, so each
// entry's parent is the previous one. Token counts are zero: agent.cpp
// doesn't report them.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include "server-common.h"  // json, safe_json_to_str
#include "util.h"

#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace agentfile {

class SessionRecorderCallback : public agent_cpp::Callback {

    FILE *file_;
    std::string model_name_;
    std::string parent_id_;               // last entry id; empty = root
    // Tool calls run in order, so the front id belongs to the next result.
    std::deque<std::string> pending_tool_calls_;
    size_t seen_messages_ = 0;            // messages already recorded

  public:
    SessionRecorderCallback(const std::string &path,
                            const std::string &model_name)
        : model_name_(model_name) {
        file_ = std::fopen(path.c_str(), "w");
        if (!file_) {
            throw std::runtime_error("--session: cannot open " + path);
        }
        char cwd[4096];
        json header = {
            {"type", "session"},
            {"version", 3},
            {"id", random_uuid()},
            {"timestamp", iso8601_now()},
            {"cwd", getcwd(cwd, sizeof(cwd)) ? cwd : ""},
        };
        write_line(header);
    }

    ~SessionRecorderCallback() override {
        if (file_) std::fclose(file_);
    }

    SessionRecorderCallback(const SessionRecorderCallback &) = delete;
    SessionRecorderCallback &operator=(const SessionRecorderCallback &) = delete;

    void before_agent_loop(std::vector<common_chat_msg> &messages) override {
        // New user messages; the other roles are recorded by their own
        // hooks, and pi has no system message.
        for (size_t i = seen_messages_; i < messages.size(); ++i) {
            if (messages[i].role != "user") continue;
            append_message({
                {"role", "user"},
                {"content", messages[i].content},
                {"timestamp", unix_ms_now()},
            });
        }
        seen_messages_ = messages.size();
    }

    void after_llm_call(common_chat_msg &parsed_msg) override {
        json content = json::array();
        if (!parsed_msg.reasoning_content.empty()) {
            content.push_back(
                {{"type", "thinking"}, {"thinking", parsed_msg.reasoning_content}});
        }
        if (!parsed_msg.content.empty()) {
            content.push_back({{"type", "text"}, {"text", parsed_msg.content}});
        }
        for (const auto &tc : parsed_msg.tool_calls) {
            json args;
            try {
                args = json::parse(tc.arguments);
            } catch (...) {
                args = {{"_raw", tc.arguments}};
            }
            content.push_back({{"type", "toolCall"},
                               {"id", tc.id},
                               {"name", tc.name},
                               {"arguments", args}});
            pending_tool_calls_.push_back(tc.id);
        }

        json zero_usage = {
            {"input", 0},      {"output", 0},     {"cacheRead", 0},
            {"cacheWrite", 0}, {"totalTokens", 0},
            {"cost",
             {{"input", 0.0},
              {"output", 0.0},
              {"cacheRead", 0.0},
              {"cacheWrite", 0.0},
              {"total", 0.0}}},
        };

        append_message({
            {"role", "assistant"},
            {"content", content},
            {"api", "agent.cpp"},
            {"provider", "agentfile"},
            {"model", model_name_},
            {"usage", zero_usage},
            {"stopReason", parsed_msg.tool_calls.empty() ? "stop" : "toolUse"},
            {"timestamp", unix_ms_now()},
        });
    }

    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        std::string call_id;
        if (!pending_tool_calls_.empty()) {
            call_id = pending_tool_calls_.front();
            pending_tool_calls_.pop_front();
        }
        bool is_error = result.has_error();
        const std::string &text =
            is_error ? result.error().message : result.output();
        append_message({
            {"role", "toolResult"},
            {"toolCallId", call_id},
            {"toolName", tool_name},
            {"content", json::array({{{"type", "text"}, {"text", text}}})},
            {"isError", is_error},
            {"timestamp", unix_ms_now()},
        });
    }

  private:
    void append_message(json message) {
        std::string id = random_hex(8);
        json entry = {
            {"type", "message"},
            {"id", id},
            {"parentId",
             parent_id_.empty() ? json(nullptr) : json(parent_id_)},
            {"timestamp", iso8601_now()},
            {"message", std::move(message)},
        };
        write_line(entry);
        parent_id_ = id;
    }

    // Tool output can be any bytes: safe_json_to_str replaces invalid UTF-8
    // where dump() would throw.
    void write_line(const json &j) {
        std::string line = safe_json_to_str(j);
        std::fwrite(line.data(), 1, line.size(), file_);
        std::fputc('\n', file_);
        std::fflush(file_);
    }
};

} // namespace agentfile
