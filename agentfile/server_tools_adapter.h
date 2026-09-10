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
// agent.cpp's Tool interface, so agentfile inherits the upstream tool
// implementations — including new tools and the isolation runtimes —
// instead of maintaining vendored copies.
//
//   ServerToolAdapter  one agent_cpp::Tool wrapping one server_tool
//   ServerToolbox      owns the upstream registry plus agentfile-native
//                      tools (http_fetch, web_search) and hands out
//                      adapters; must outlive the Agent using them
//
// Isolation: --tools-runtime makes tools run their file and shell
// operations inside a sandbox instead of on the host. The spec string is
// forwarded to every tool call as params["runtime"].
//
// Only sandboxes that already exist work here: start a container yourself,
// then pass e.g. "docker-container:<name>". llama-server can additionally
// CREATE a container on demand from a spec, but that creation logic is
// private to llama.cpp (server-tools.cpp), so agentfile cannot reuse it.
// Consequences: create-on-demand specs are unsupported, and a malformed
// spec is only detected at the first tool call, not at startup.
//

#pragma once

#include "server-tools.h"   // server_tool, server_tools
#include "server-mcp.h"     // server_mcp (empty manager)

#include "chat.h"
#include "tool.h"

#include "tools/http_fetch.h"
#include "tools/web_search.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace agentfile {

class ServerToolAdapter : public agent_cpp::Tool {
    const server_tool *tool_;   // borrowed from ServerToolbox
    std::string runtime_spec_;

  public:
    ServerToolAdapter(const server_tool *tool, std::string runtime_spec)
        : tool_(tool), runtime_spec_(std::move(runtime_spec)) {}

    std::string get_name() const override { return tool_->name; }

    common_chat_tool get_definition() const override {
        // server_tool definitions are OpenAI-shaped:
        // {"type":"function","function":{name,description,parameters}}
        nlohmann::ordered_json d = tool_->get_definition();
        const nlohmann::ordered_json &f = d.at("function");
        return {f.at("name").get<std::string>(),
                f.at("description").get<std::string>(),
                f.at("parameters").dump()};
    }

    // agent.cpp speaks nlohmann::json, server tools speak ordered_json —
    // convert at the boundary via dump/parse.
    std::string execute(const nlohmann::json &arguments) override {
        auto params = nlohmann::ordered_json::parse(arguments.dump());
        if (!runtime_spec_.empty()) {
            params["runtime"] = runtime_spec_;
        }
        try {
            return tool_->invoke(std::move(params), nullptr).dump();
        } catch (const std::exception &e) {
            // Tools normally report failures as {"error": ...} themselves;
            // this is the backstop for the ones that throw.
            return nlohmann::ordered_json{{"error", e.what()}}.dump();
        }
    }
};

class ServerToolbox {
    server_mcp mcp_;   // default-constructed: no MCP servers
    server_tools st_;
    std::string runtime_spec_;

  public:
    // searxng_url empty = web_search not registered.
    // runtime_spec empty = tools run directly on the host.
    ServerToolbox(const std::string &searxng_url,
                  const std::string &runtime_spec)
        : runtime_spec_(runtime_spec) {
        st_.setup({"all"}, mcp_, "");
        st_.tools.push_back(std::make_unique<tools::HttpFetchTool>());
        if (!searxng_url.empty()) {
            st_.tools.push_back(
                std::make_unique<tools::WebSearchTool>(searxng_url));
        }
    }

    ServerToolbox(const ServerToolbox &) = delete;
    ServerToolbox &operator=(const ServerToolbox &) = delete;

    std::set<std::string> tool_names() const {
        std::set<std::string> names;
        for (const auto &t : st_.tools) names.insert(t->name);
        return names;
    }

    // Tools that mutate state or reach the network (permission_write) —
    // these get a confirmation prompt unless --yes is given.
    std::set<std::string> write_tool_names() const {
        std::set<std::string> names;
        for (const auto &t : st_.tools)
            if (t->permission_write) names.insert(t->name);
        return names;
    }

    // Adapters for the tools in `keep` (empty = all). The toolbox must
    // outlive the returned adapters.
    std::vector<std::unique_ptr<agent_cpp::Tool>>
    make_adapters(const std::set<std::string> &keep) const {
        std::vector<std::unique_ptr<agent_cpp::Tool>> out;
        for (const auto &t : st_.tools) {
            if (!keep.empty() && !keep.count(t->name)) continue;
            out.push_back(
                std::make_unique<ServerToolAdapter>(t.get(), runtime_spec_));
        }
        return out;
    }
};

} // namespace agentfile
