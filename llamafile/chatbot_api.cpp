// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#include "chatbot.h"
#include "chatbot_backend.h"
#include "color.h"

#include <cstdio>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>

using json = nlohmann::ordered_json;

namespace lf {
namespace chatbot {

// ApiBackend: HTTP client that talks to a local llamafile server
// via /v1/chat/completions with SSE streaming.
class ApiBackend : public ChatBackend {
public:
    explicit ApiBackend(const std::string &server_url)
        : server_url_(server_url) {
        // Parse host and port from URL
        // Expected format: http://host:port
        std::string url = server_url;
        if (url.substr(0, 7) == "http://")
            url = url.substr(7);
        auto colon = url.find(':');
        if (colon != std::string::npos) {
            host_ = url.substr(0, colon);
            port_ = std::stoi(url.substr(colon + 1));
        } else {
            host_ = url;
            port_ = 8080;
        }
    }

    // Build the "content" field for a message, converting any embedded
    // data URIs into multimodal content parts (image_url).
    static json build_content(const std::string &text) {
        // Look for data:image/ URIs
        static const std::string prefix = "data:image/";
        auto pos = text.find(prefix);
        if (pos == std::string::npos) {
            return text;  // plain string, no images
        }

        // Split text into text parts and image_url parts
        json parts = json::array();
        size_t last = 0;

        while (pos != std::string::npos) {
            // Add preceding text
            if (pos > last) {
                std::string before = text.substr(last, pos - last);
                if (!before.empty())
                    parts.push_back({{"type", "text"}, {"text", before}});
            }

            // Find end of data URI (terminated by whitespace, quote, or end of string)
            size_t end = pos;
            while (end < text.size() && text[end] != ' ' && text[end] != '\n' &&
                   text[end] != '\r' && text[end] != '\t' && text[end] != '"')
                end++;

            std::string uri = text.substr(pos, end - pos);
            parts.push_back({
                {"type", "image_url"},
                {"image_url", {{"url", uri}}}
            });

            last = end;
            pos = text.find(prefix, last);
        }

        // Add trailing text
        if (last < text.size()) {
            std::string after = text.substr(last);
            if (!after.empty())
                parts.push_back({{"type", "text"}, {"text", after}});
        }

        return parts;
    }

    std::string complete(
        const std::vector<common_chat_msg> &messages,
        TokenCallback on_token) override
    {
        // Build request JSON
        json req_json;
        json msgs_json = json::array();
        for (const auto &msg : messages) {
            msgs_json.push_back({
                {"role", msg.role},
                {"content", build_content(msg.content)}
            });
        }
        req_json["messages"] = msgs_json;
        req_json["stream"] = true;
        req_json["stream_options"] = {{"include_usage", true}};

        std::string body = req_json.dump();
        std::string assistant_content;
        std::string sse_buffer;
        bool stopped = false;

        httplib::Client cli(host_, port_);
        cli.set_read_timeout(300);  // 5 minutes for long generations

        auto result = cli.Post(
            "/v1/chat/completions",
            httplib::Headers{},
            body,
            "application/json",
            [&](const char *data, size_t len) -> bool {
                if (stopped)
                    return false;

                sse_buffer.append(data, len);

                // Process complete SSE lines
                size_t pos;
                while ((pos = sse_buffer.find("\n")) != std::string::npos) {
                    std::string line = sse_buffer.substr(0, pos);
                    sse_buffer.erase(0, pos + 1);

                    // Skip empty lines
                    if (line.empty() || line == "\r")
                        continue;

                    // Strip trailing \r
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();

                    // Only process "data: " lines
                    if (line.substr(0, 6) != "data: ")
                        continue;

                    std::string payload = line.substr(6);

                    // Check for stream end
                    if (payload == "[DONE]")
                        return true;

                    // Parse JSON
                    try {
                        json chunk = json::parse(payload);

                        // Extract usage stats from final chunk
                        if (chunk.contains("usage")) {
                            auto &usage = chunk["usage"];
                            if (usage.contains("prompt_tokens"))
                                last_prompt_tokens_ = usage["prompt_tokens"].get<int>();
                            if (usage.contains("completion_tokens"))
                                last_completion_tokens_ = usage["completion_tokens"].get<int>();
                            if (usage.contains("total_tokens"))
                                total_tokens_ = usage["total_tokens"].get<int>();
                        }

                        if (!chunk.contains("choices") || chunk["choices"].empty())
                            continue;

                        auto &choice = chunk["choices"][0];
                        if (!choice.contains("delta"))
                            continue;

                        auto &delta = choice["delta"];

                        std::string content_delta;
                        std::string reasoning_delta;

                        if (delta.contains("content") && !delta["content"].is_null())
                            content_delta = delta["content"].get<std::string>();

                        if (delta.contains("reasoning_content") && !delta["reasoning_content"].is_null())
                            reasoning_delta = delta["reasoning_content"].get<std::string>();

                        if (!content_delta.empty())
                            assistant_content += content_delta;

                        if (!content_delta.empty() || !reasoning_delta.empty()) {
                            if (!on_token(content_delta, reasoning_delta)) {
                                stopped = true;
                                return false;  // close connection to cancel
                            }
                        }
                    } catch (const json::exception &) {
                        // Skip malformed JSON chunks
                        continue;
                    }
                }
                return true;
            });

        if (!result) {
            err("error: HTTP request failed: %s", httplib::to_string(result.error()).c_str());
        } else if (result->status != 200) {
            err("error: server returned HTTP %d", result->status);
        }

        return assistant_content;
    }

    int context_used() override {
        return total_tokens_;
    }

    int context_max() override {
        if (context_max_ <= 0)
            fetch_context_max();
        return context_max_;
    }

    void print_stats() override {
        printf("prompt tokens: %d\n"
               "completion tokens: %d\n"
               "total tokens: %d\n",
               last_prompt_tokens_, last_completion_tokens_, total_tokens_);
    }

    void on_clear() override {
        // Message list is managed by the REPL (g_messages)
        // Just reset our token counters
        last_prompt_tokens_ = 0;
        last_completion_tokens_ = 0;
        total_tokens_ = 0;
    }

    void on_push() override {
        message_stack_.push_back(g_messages);
        printf(FAINT "conversation state pushed (depth: %zu)" RESET "\n",
               message_stack_.size());
    }

    void on_pop() override {
        if (message_stack_.empty()) {
            err("error: conversation stack is empty");
            return;
        }
        g_messages = message_stack_.back();
        message_stack_.pop_back();
        printf(FAINT "conversation state restored (depth: %zu)" RESET "\n",
               message_stack_.size());
    }

    void on_undo() override {
        // Remove last assistant + user message pair
        while (!g_messages.empty() && g_messages.back().role == "assistant")
            g_messages.pop_back();
        if (!g_messages.empty() && g_messages.back().role == "user")
            g_messages.pop_back();
        printf(FAINT "last exchange undone (%zu messages remaining)" RESET "\n",
               g_messages.size());
    }

    void on_forget(int n) override {
        // Remove the oldest non-system message pair to free context
        auto it = g_messages.begin();
        while (it != g_messages.end() && it->role == "system")
            ++it;
        if (it == g_messages.end()) {
            err("error: nothing left to forget");
            return;
        }
        // Remove one user+assistant exchange
        auto start = it;
        ++it;  // skip user
        if (it != g_messages.end() && it->role == "assistant")
            ++it;  // skip assistant
        g_messages.erase(start, it);
        printf(FAINT "oldest exchange forgotten (%zu messages remaining)" RESET "\n",
               g_messages.size());
    }

private:
    std::string server_url_;
    std::string host_;
    int port_;

    // Token usage tracking (from SSE usage stats)
    int last_prompt_tokens_ = 0;
    int last_completion_tokens_ = 0;
    int total_tokens_ = 0;

    // Cached context size from server /props endpoint
    int context_max_ = 0;

    void fetch_context_max() {
        httplib::Client cli(host_, port_);
        cli.set_read_timeout(5);
        auto result = cli.Get("/props");
        if (result && result->status == 200) {
            try {
                json props = json::parse(result->body);
                if (props.contains("default_generation_settings")) {
                    auto &settings = props["default_generation_settings"];
                    if (settings.contains("n_ctx"))
                        context_max_ = settings["n_ctx"].get<int>();
                }
            } catch (const json::exception &) {
            }
        }
    }

    // Message stack for /push and /pop
    std::vector<std::vector<common_chat_msg>> message_stack_;
};

// Factory function
std::unique_ptr<ChatBackend> create_api_backend(const std::string &server_url) {
    return std::make_unique<ApiBackend>(server_url);
}

} // namespace chatbot
} // namespace lf
