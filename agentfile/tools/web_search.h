// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// web_search: query a SearXNG instance (https://docs.searxng.org) via its
// JSON search API and return the top results.
//
// The instance base URL is deliberately NOT a tool argument — the model
// must not choose where queries go. It comes from --searxng-url or the
// SEARXNG_URL environment variable; when neither is set the tool is not
// registered at all.
//
// Notes:
//   - The instance must enable `json` under `search: formats:` in its
//     settings.yml; SearXNG answers 403 otherwise (most public instances
//     do not enable it — self-hosted is the expected deployment).
//   - Built on common_http_client like http_fetch; http:// and https://
//     instances both work (cpp-httplib is compiled with the Mbed TLS
//     backend, PR #1011).
//

#pragma once

#include <cpp-httplib/httplib.h>
#include "http.h"   // common_http_client, common_http_parse_url

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <cerrno>
#include <cstring>
#include <string>

namespace agentfile {
namespace tools {

class WebSearchTool : public agent_cpp::Tool {
    static constexpr int    kTimeoutSeconds = 30;
    static constexpr size_t kMaxResults = 8;
    static constexpr size_t kMaxSnippet = 400; // chars per result snippet

    std::string base_url_; // e.g. "http://localhost:8888", no trailing slash

  public:
    explicit WebSearchTool(std::string base_url)
        : base_url_(std::move(base_url)) {
        while (!base_url_.empty() && base_url_.back() == '/')
            base_url_.pop_back();
    }

    std::string get_name() const override { return "web_search"; }

    common_chat_tool get_definition() const override {
        json schema = {
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
        };
        return {"web_search",
                "Search the web via a SearXNG instance and return the top "
                "results as JSON (title, url, snippet). Use precise queries; "
                "fetch promising URLs with other tools if more detail is "
                "needed.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        if (!arguments.contains("query")) {
            return error_response("missing required parameter: query");
        }
        std::string query = arguments.at("query").get<std::string>();

        std::string url = base_url_ + "/search?format=json&q=" + url_encode(query);
        int page = json_value(arguments, "page", 1);
        if (page > 1) url += "&pageno=" + std::to_string(page);
        std::string time_range = json_value(arguments, "time_range", std::string{});
        if (time_range == "day" || time_range == "month" || time_range == "year") {
            url += "&time_range=" + time_range;
        }
        std::string categories = json_value(arguments, "categories", std::string{});
        if (!categories.empty()) url += "&categories=" + url_encode(categories);

        try {
            auto [cli, parts] = common_http_client(url);
            cli.set_read_timeout(kTimeoutSeconds, 0);
            cli.set_connection_timeout(kTimeoutSeconds, 0);

            auto res = cli.Get(parts.path.c_str(),
                               httplib::Headers{{"Accept", "application/json"}});
            if (!res) {
                int err = errno;
                return error_response("request failed: " +
                                      httplib::to_string(res.error()) +
                                      " (errno " + std::to_string(err) + ": " +
                                      strerror(err) + ")");
            }
            if (res->status == 403) {
                return error_response(
                    "SearXNG returned 403: the instance at " + base_url_ +
                    " does not allow the JSON API. Enable `json` under "
                    "`search: formats:` in its settings.yml.");
            }
            if (res->status != 200) {
                return error_response("SearXNG returned HTTP " +
                                      std::to_string(res->status));
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
                    if (results.size() >= kMaxResults) break;
                }
            }
            return result_response({
                {"query", query},
                {"results", results},
                {"total_available",
                 body.contains("results") && body["results"].is_array()
                     ? body["results"].size()
                     : 0},
            });
        } catch (const json::exception &e) {
            return error_response(
                std::string("SearXNG returned unparseable JSON: ") + e.what());
        } catch (const std::exception &e) {
            return error_response(std::string("web_search failed: ") + e.what());
        }
    }

  private:
    static std::string url_encode(const std::string &s) {
        static const char hex[] = "0123456789ABCDEF";
        std::string out;
        out.reserve(s.size() * 3);
        for (unsigned char c : s) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
                out += (char)c;
            } else {
                out += '%';
                out += hex[c >> 4];
                out += hex[c & 0xf];
            }
        }
        return out;
    }
};

} // namespace tools
} // namespace agentfile
