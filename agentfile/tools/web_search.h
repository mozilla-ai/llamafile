// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// web_search: query a SearXNG instance (https://docs.searxng.org) via its
// JSON search API. Written as a server_tool (llama.cpp tools/server style)
// so it can be offered upstream as-is; agentfile runs it through
// ServerToolAdapter.
//
// The instance base URL is deliberately NOT a tool argument — the model
// must not choose where queries go. It comes from --searxng-url or the
// SEARXNG_URL environment variable; when neither is set the tool is not
// registered at all.
//
// The instance must enable `json` under `search: formats:` in its
// settings.yml (SearXNG answers 403 otherwise; most public instances
// don't — self-hosted is the expected deployment).
//

#pragma once

#include "server-tools.h"

#include <cpp-httplib/httplib.h>
#include "http.h"   // common_http_client, common_http_parse_url

#include <string>

namespace agentfile {
namespace tools {

struct WebSearchTool : server_tool {
    static constexpr int    kTimeoutSeconds = 30;
    // Snippet clip is input sanitation only: SearXNG engines send ~160
    // chars today, so this triggers only on unusual engines/instances.
    static constexpr size_t kMaxSnippet = 512;

    std::string base_url; // e.g. "http://localhost:8888", no trailing slash

    explicit WebSearchTool(std::string url) : base_url(std::move(url)) {
        while (!base_url.empty() && base_url.back() == '/')
            base_url.pop_back();
        name = "web_search";
        display_name = "Web search";
        // Network access: confirm before running (unless --yes).
        permission_write = true;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                 "Search the web via a SearXNG instance and return one page "
                 "of results as JSON (title, url, snippet). Use precise "
                 "queries; ask for the next page if the answer isn't there; "
                 "fetch promising URLs with other tools for more detail."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query",      {{"type", "string"},
                                        {"description", "The search query"}}},
                        {"page",       {{"type", "integer"},
                                        {"description", "Result page number (default 1)"}}},
                        {"time_range", {{"type", "string"},
                                        {"enum", json::array({"day", "month", "year"})},
                                        {"description", "Restrict results to this time range"}}},
                        {"categories", {{"type", "string"},
                                        {"description", "Comma-separated SearXNG categories, "
                                                        "e.g. \"general\", \"news\", \"it\""}}},
                    }},
                    {"required", json::array({"query"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        if (!params.contains("query")) {
            return {{"error", "missing required parameter: query"}};
        }
        std::string query = params.at("query").get<std::string>();

        std::string url = base_url + "/search?format=json&q=" +
                          httplib::encode_query_component(query);
        int page = json_value(params, "page", 1);
        if (page > 1) url += "&pageno=" + std::to_string(page);
        std::string time_range = json_value(params, "time_range", std::string{});
        if (time_range == "day" || time_range == "month" ||
            time_range == "year") {
            url += "&time_range=" + time_range;
        }
        std::string categories = json_value(params, "categories", std::string{});
        if (!categories.empty())
            url += "&categories=" + httplib::encode_query_component(categories);

        try {
            auto [cli, parts] = common_http_client(url);
            cli.set_read_timeout(kTimeoutSeconds, 0);
            cli.set_connection_timeout(kTimeoutSeconds, 0);

            auto res = cli.Get(parts.path,
                               httplib::Headers{{"Accept", "application/json"}});
            if (!res) {
                int err = errno;
                return {{"error", "request failed: " +
                                      httplib::to_string(res.error()) +
                                      " (errno " + std::to_string(err) +
                                      ": " + strerror(err) + ")"}};
            }
            if (res->status == 403) {
                return {{"error",
                         "SearXNG returned 403: the instance at " + base_url +
                             " does not allow the JSON API. Enable `json` "
                             "under `search: formats:` in its settings.yml."}};
            }
            if (res->status != 200) {
                return {{"error", "SearXNG returned HTTP " +
                                      std::to_string(res->status)}};
            }

            json body = json::parse(res->body);
            json results = json::array();
            if (body.contains("results") && body["results"].is_array()) {
                for (const auto &r : body["results"]) {
                    std::string snippet = r.value("content", "");
                    if (snippet.size() > kMaxSnippet) {
                        snippet.resize(kMaxSnippet);
                        snippet += "…";
                    }
                    results.push_back({
                        {"title", r.value("title", "")},
                        {"url", r.value("url", "")},
                        {"snippet", snippet},
                        {"engine", r.value("engine", "")},
                    });
                }
            }
            return {
                {"query", query},
                {"results", results},
            };
        } catch (const json::exception &e) {
            return {{"error",
                     std::string("SearXNG returned unparseable JSON: ") +
                         e.what()}};
        } catch (const std::exception &e) {
            return {{"error", std::string("web_search failed: ") + e.what()}};
        }
    }
};

} // namespace tools
} // namespace agentfile
