// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// http_fetch: HTTP(S) GET built on llama.cpp's common_http_client.
// Written as a server_tool (llama.cpp tools/server style) so it can be
// offered upstream as-is; agentfile runs it through ServerToolAdapter.
//
// Limits: GET only; body capped at 16 KB (truncated + marked); 30s timeout.
//

#pragma once

#include "server-tools.h"

#include <cpp-httplib/httplib.h>
#include "http.h"   // common_http_client, common_http_parse_url

#include <string>

namespace agentfile {
namespace tools {

struct HttpFetchTool : server_tool {
    static constexpr size_t kMaxBody = 16 * 1024;
    static constexpr int    kTimeoutSeconds = 30;

    HttpFetchTool() {
        name = "http_fetch";
        display_name = "Fetch URL";
        // Network access: confirm before running (unless --yes).
        permission_write = true;
    }

    json get_definition() const override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                 "Fetch a URL with HTTP GET and return status + headers + "
                 "body. Response body is capped at 16 KB. Use sparingly — "
                 "this tool grants the model network access."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"url",     {{"type", "string"},
                                     {"description", "HTTP or HTTPS URL to fetch"}}},
                        {"headers", {{"type", "object"},
                                     {"description", "Optional request headers as name->value map"}}},
                    }},
                    {"required", json::array({"url"})},
                }},
            }},
        };
    }

    json invoke(json params, server_tool::stream *) const override {
        if (!params.contains("url")) {
            return {{"error", "missing required parameter: url"}};
        }
        std::string url = params.at("url").get<std::string>();

        try {
            auto [cli, parts] = common_http_client(url);
            cli.set_read_timeout(kTimeoutSeconds, 0);
            cli.set_connection_timeout(kTimeoutSeconds, 0);

            httplib::Headers req_headers;
            if (params.contains("headers") && params.at("headers").is_object()) {
                for (const auto &[k, v] : params.at("headers").items()) {
                    if (v.is_string()) {
                        req_headers.emplace(k, v.get<std::string>());
                    }
                }
            }

            auto res = cli.Get(parts.path, req_headers);
            if (!res) {
                return {{"error", "request failed: " +
                                      httplib::to_string(res.error())}};
            }

            bool truncated = false;
            std::string body = res->body;
            if (body.size() > kMaxBody) {
                body.resize(kMaxBody);
                truncated = true;
            }

            json resp_headers = json::object();
            for (const auto &h : res->headers) {
                resp_headers[h.first] = h.second;
            }

            return {
                {"status", res->status},
                {"headers", resp_headers},
                {"body", body},
                {"truncated", truncated},
            };
        } catch (const std::exception &e) {
            return {{"error", std::string("http_fetch failed: ") + e.what()}};
        }
    }
};

} // namespace tools
} // namespace agentfile
