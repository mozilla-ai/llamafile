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
// http_fetch: HTTP(S) GET built on llama.cpp's common_http_client.
// Written as a server_tool (llama.cpp tools/server style) so it can be
// offered upstream as-is; agentfile runs it through ServerToolAdapter.
//
// Limits: GET only; body capped at 64 KB (truncated + marked); 30s timeout;
// redirects are reported (redirect_to), not followed.
//

#pragma once

#include "server-tools.h"

#include <cpp-httplib/httplib.h>
#include "http.h"   // common_http_client, common_http_parse_url

#include <string>

namespace agentfile {
namespace tools {

struct HttpFetchTool : server_tool {
    static constexpr size_t kMaxBody = 64 * 1024;
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
                 "body. Response body is capped at 64 KB. Redirects are not "
                 "followed automatically: a 3xx response carries redirect_to; "
                 "call http_fetch again with that URL to retrieve the content. "
                 "Use sparingly — this tool grants the model network access."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"url", {{"type", "string"},
                                 {"description", "HTTP or HTTPS URL to fetch"}}},
                    }},
                    {"required", json::array({"url"})},
                }},
            }},
        };
    }

    // Resolve a Location header against the request URL (RFC 3986-lite:
    // absolute, host-relative, or path-relative).
    static std::string resolve_location(const common_http_url &parts,
                                        const std::string &loc) {
        if (loc.find("://") != std::string::npos) return loc;
        std::string origin = parts.scheme + "://" +
                             common_http_format_host(parts.host) + ":" +
                             std::to_string(parts.port);
        if (!loc.empty() && loc[0] == '/') return origin + loc;
        std::string dir = parts.path.substr(0, parts.path.rfind('/') + 1);
        if (dir.empty()) dir = "/";
        return origin + dir + loc;
    }

    json invoke(json params, server_tool::stream *) const override {
        if (!params.contains("url")) {
            return {{"error", "missing required parameter: url"}};
        }
        std::string url = params.at("url").get<std::string>();

        try {
            auto [cli, parts] = common_http_client(url);
            // Report redirects instead of following them, so every URL that
            // gets fetched is one the model asked for by name — and one the
            // confirmation prompt showed the user.
            cli.set_follow_location(false);
            cli.set_read_timeout(kTimeoutSeconds, 0);
            cli.set_connection_timeout(kTimeoutSeconds, 0);

            auto res = cli.Get(parts.path);
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

            json out = json::object();
            // Lead with the redirect so the model sees it before the
            // headers/body noise.
            if (res->status >= 300 && res->status < 400 &&
                res->has_header("Location")) {
                out["redirect_to"] =
                    resolve_location(parts, res->get_header_value("Location"));
                out["note"] = "Redirect, not followed automatically. The "
                              "content is at redirect_to; call http_fetch "
                              "with that URL to retrieve it.";
            }
            out["status"] = res->status;
            out["headers"] = resp_headers;
            out["body"] = body;
            out["truncated"] = truncated;
            return out;
        } catch (const std::exception &e) {
            return {{"error", std::string("http_fetch failed: ") + e.what()}};
        }
    }
};

} // namespace tools
} // namespace agentfile
