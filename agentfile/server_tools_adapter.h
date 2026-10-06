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
// Bridges llama.cpp's server tools (tools/server/server-tools.h) into
// agent.cpp's Tool interface, so agentfile uses the upstream tools and
// their isolation runtimes instead of keeping its own copies.
//
//   ServerToolAdapter  wraps one server_tool as an agent_cpp::Tool
//   ServerToolbox      owns the tools, upstream and agentfile's own
//                      (get_datetime, http_fetch, web_search); must
//                      outlive the Agent that uses them
//
// Every call goes through server_tools::handle_post, the handler behind
// llama-server's POST /tools: it drops the keys the model must not set
// (cwd, runtime, resp_type), runs the tool in the --tools-runtime isolate
// if one is set, and turns exceptions into error responses.
//

#pragma once

#include "server-tools.h"   // server_tool, server_tools
#include "server-mcp.h"     // server_mcp (empty manager)

#include "chat.h"
#include "tool.h"
#include "util.h"

#include "tools/get_datetime.h"
#include "tools/http_fetch.h"
#include "tools/web_search.h"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace agentfile {

// Tool-call arguments as the confirmation prompt and progress line show
// them: as the tool will get them (re-serialized, without the keys
// handle_post drops) and escaped for the terminal, so model text can't
// hide or fake part of the line. A "url" also shows the host and port the
// request really goes to (http://a.example@b.example/ goes to b.example).
// Arguments that aren't JSON are shown escaped, as they are.
inline std::string printable_arguments(const std::string &arguments) {
    std::string text = arguments;
    std::string note;
    try {
        // Parsed the way the call parses them: agent.cpp's json, then the
        // server tools' json.
        auto args = json::parse(nlohmann::json::parse(arguments).dump());
        if (args.is_object()) {
            for (const char *key : {"cwd", "runtime", "resp_type"}) {
                if (args.contains(key)) args.erase(key);
            }
            if (args.contains("url") && args.at("url").is_string()) {
                try {
                    auto parts = common_http_parse_url(
                        args.at("url").get<std::string>());
                    note += " (connects to " + parts.scheme + "://" +
                            common_http_format_host(parts.host) + ":" +
                            std::to_string(parts.port) + ")";
                } catch (const std::exception &) {
                    // http_fetch reports the bad URL itself.
                }
            }
        }
        text = safe_json_to_str(args);
    } catch (const std::exception &) {
    }
    return terminal_text(text + note);
}

class ServerToolAdapter : public agent_cpp::Tool {
    // Both borrowed from ServerToolbox.
    const server_tool *tool_;
    const server_http_context::handler_t &handle_post_;

  public:
    ServerToolAdapter(const server_tool *tool,
                      const server_http_context::handler_t &handle_post)
        : tool_(tool), handle_post_(handle_post) {}

    std::string get_name() const override { return tool_->name; }

    common_chat_tool get_definition() const override {
        // server_tool definitions are OpenAI-shaped:
        // {"type":"function","function":{name,description,parameters}}
        json d = tool_->get_definition();
        const json &f = d.at("function");
        return {f.at("name").get<std::string>(),
                f.at("description").get<std::string>(),
                f.at("parameters").dump()};
    }

    // agent.cpp speaks nlohmann::json, server tools speak llama.cpp's json
    // (common_json): the request body is the boundary between them.
    std::string execute(const nlohmann::json &arguments) override {
        static const std::function<bool()> never_stop = [] { return false; };
        json body = {{"tool", tool_->name},
                     {"params", json::parse(arguments.dump())}};
        server_http_req req{.body = body.dump(), .should_stop = never_stop};
        auto res = handle_post_(req);
        // The handler serializes with safe_json_to_str, which also replaces
        // invalid UTF-8 (a Latin-1 file, a byte-truncated snippet).
        json result = json::parse(res->data);
        if (res->status != 200) {
            // A tool threw (bad arguments, say): pass the message on in the
            // {"error": msg} shape the tools use themselves.
            return safe_json_to_str(
                json{{"error", json_value(result, "message", res->data)}});
        }
        // Hand a {"plain_text_response": text} result on as the text
        // itself, not as escaped JSON.
        if (result.is_object() && !result.contains("error") &&
            result.contains("plain_text_response") &&
            result.at("plain_text_response").is_string()) {
            return result.at("plain_text_response").get<std::string>();
        }
        return res->data;
    }
};

class ServerToolbox {
    server_mcp mcp_;   // default-constructed: no MCP servers
    server_tools st_;
    // Tools that exist but could not be registered, with the reason —
    // shown by --tools validation and in the help text.
    std::map<std::string, std::string> missing_;

  public:
    // searxng_url empty = web_search not registered.
    // runtime_spec empty = tools run directly on the host; otherwise a
    // --tools-runtime spec, which setup() checks (throws if invalid) and,
    // for "docker:IMAGE", starts a container that lives as long as the
    // toolbox.
    ServerToolbox(const std::string &searxng_url,
                  const std::string &runtime_spec) {
        st_.setup({"all"}, mcp_, runtime_spec);
        st_.tools.push_back(std::make_unique<tools::GetDatetimeTool>());
        st_.tools.push_back(std::make_unique<tools::HttpFetchTool>());
        if (!searxng_url.empty()) {
            st_.tools.push_back(
                std::make_unique<tools::WebSearchTool>(searxng_url));
        } else {
            missing_["web_search"] =
                "requires a SearXNG instance: pass --searxng-url URL or "
                "set SEARXNG_URL";
        }
    }

    // name -> why it is not available in this configuration.
    const std::map<std::string, std::string> &missing() const {
        return missing_;
    }

    ServerToolbox(const ServerToolbox &) = delete;
    ServerToolbox &operator=(const ServerToolbox &) = delete;

    std::set<std::string> tool_names() const {
        std::set<std::string> names;
        for (const auto &t : st_.tools) names.insert(t->name);
        return names;
    }

    // The tools among `names` that mutate state or reach the network
    // (permission_write) — these get a confirmation prompt unless --yes is
    // given.
    std::set<std::string>
    write_tool_names(const std::set<std::string> &names) const {
        std::set<std::string> out;
        for (const auto &t : st_.tools)
            if (t->permission_write && names.count(t->name))
                out.insert(t->name);
        return out;
    }

    // Adapters for the tools among `names`. The toolbox must outlive them.
    std::vector<std::unique_ptr<agent_cpp::Tool>>
    make_adapters(const std::set<std::string> &names) const {
        std::vector<std::unique_ptr<agent_cpp::Tool>> out;
        for (const auto &t : st_.tools)
            if (names.count(t->name))
                out.push_back(std::make_unique<ServerToolAdapter>(
                    t.get(), st_.handle_post));
        return out;
    }
};

} // namespace agentfile
