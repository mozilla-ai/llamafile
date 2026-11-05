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

#include <algorithm>
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
        std::string raw_body;  // start of the response, for error messages
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

                if (raw_body.size() < 65536)
                    raw_body.append(data, std::min(len, 65536 - raw_body.size()));
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

                        // Extract timings from the final chunk. The server
                        // always attaches these once its stats are set, so no
                        // request flag is needed; without them /stats falls
                        // back to token counts alone.
                        if (chunk.contains("timings")) {
                            auto &t = chunk["timings"];
                            if (t.contains("prompt_n"))
                                last_prompt_n_ = t["prompt_n"].get<int>();
                            if (t.contains("prompt_ms"))
                                last_prompt_ms_ = t["prompt_ms"].get<double>();
                            if (t.contains("predicted_n"))
                                last_predicted_n_ = t["predicted_n"].get<int>();
                            if (t.contains("predicted_ms"))
                                last_predicted_ms_ = t["predicted_ms"].get<double>();
                            if (t.contains("cache_n"))
                                last_cache_n_ = t["cache_n"].get<int>();
                            if (t.contains("draft_n"))
                                last_draft_n_ = t["draft_n"].get<int>();
                            if (t.contains("draft_n_accepted"))
                                last_draft_accepted_ = t["draft_n_accepted"].get<int>();
                            have_timings_ = true;
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

        if (stopped) {
            // User interrupted generation with Ctrl+C. We closed the connection
            // ourselves by returning false from the receiver, so httplib reports
            // Error::Canceled here; that is expected, not a failure. Any partial
            // response was already streamed and is returned below.
        } else if (!result) {
            err("error: HTTP request failed: %s", httplib::to_string(result.error()).c_str());
        } else if (result->status != 200) {
            // show the server's reason, as llama-cli does (e.g. a decision
            // model, which serves /v1/systemone and can't chat)
            std::string message = error_message(result->body.empty() ? raw_body : result->body);
            if (message.empty())
                err("error: server returned HTTP %d", result->status);
            else
                err("error: %s", message.c_str());
        }

        return assistant_content;
    }

    // The "error.message" of an OpenAI-style error body, or "" if there is none
    static std::string error_message(const std::string &body) {
        try {
            json j = json::parse(body);
            if (j.contains("error") && j["error"].is_object() &&
                j["error"].contains("message") && j["error"]["message"].is_string())
                return j["error"]["message"].get<std::string>();
        } catch (const json::exception &) {
        }
        return "";
    }

    // False if the server's model can't generate text: a decision model, which
    // serves POST /v1/systemone instead. True if the server doesn't say.
    bool can_chat() {
        httplib::Client cli(host_, port_);
        cli.set_read_timeout(5);
        auto result = cli.Get("/v1/models");
        if (!result || result->status != 200)
            return true;
        try {
            json models = json::parse(result->body);
            const auto &out = models.at("data").at(0).at("architecture").at("output_modalities");
            for (const auto &m : out)
                if (m == "text")
                    return true;
            return out.empty();
        } catch (const json::exception &) {
            return true;
        }
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
        // Prefer the timings' own token counts: they are what the millisecond
        // figures were measured over, so the derived rates stay consistent.
        // usage.prompt_tokens also counts cache hits, which were never decoded.
        const int prompt_n = have_timings_ ? last_prompt_n_ : last_prompt_tokens_;
        const int gen_n = have_timings_ ? last_predicted_n_ : last_completion_tokens_;

        // The server's generation timer starts when the first token is
        // sampled -- that token comes free from the last prompt batch's
        // logits -- so predicted_ms covers predicted_n - 1 decode steps. This
        // is what the server's own n_gen_tps() divides by.
        const int gen_rate_n = gen_n > 0 ? gen_n - 1 : 0;

        if (!print_stats_core(prompt_n, last_prompt_ms_, gen_n,
                              last_predicted_ms_, gen_rate_n,
                              last_prompt_ms_ + last_predicted_ms_))
            return;

        // server only: the direct backend has no equivalent for these
        if (last_cache_n_ > 0)
            printf("  %-10s %6d tokens\n", "cached", last_cache_n_);
        if (last_draft_n_ > 0)
            printf("  %-10s %6d/%d accepted\n", "draft", last_draft_accepted_,
                   last_draft_n_);
    }

    void on_clear() override {
        // Message list is managed by the REPL (g_messages)
        // Just reset our token counters
        last_prompt_tokens_ = 0;
        last_completion_tokens_ = 0;
        total_tokens_ = 0;
        have_timings_ = false;
        last_prompt_n_ = 0;
        last_prompt_ms_ = 0;
        last_predicted_n_ = 0;
        last_predicted_ms_ = 0;
        last_cache_n_ = 0;
        last_draft_n_ = 0;
        last_draft_accepted_ = 0;
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

    // Perf tracking (from the SSE timings object on the final chunk)
    bool have_timings_ = false;
    int last_prompt_n_ = 0;
    double last_prompt_ms_ = 0;
    int last_predicted_n_ = 0;
    double last_predicted_ms_ = 0;
    int last_cache_n_ = 0;
    int last_draft_n_ = 0;
    int last_draft_accepted_ = 0;

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

bool api_server_can_chat(const std::string &server_url) {
    return ApiBackend(server_url).can_chat();
}

} // namespace chatbot
} // namespace lf
