// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// http_fetch: an HTTP GET tool built on llama.cpp's common_http_client()
// helper (which wraps cpp-httplib). Not in llama.cpp's server-tools set;
// added for agentfile to give the agent a basic "fetch a URL" primitive
// in the spirit of "if we added an HTTP client tool we'd already go a
// long way" (project brief).
//
// HTTPS is supported since llamafile compiles cpp-httplib with its
// Mbed TLS backend (CPPHTTPLIB_MBEDTLS_SUPPORT, PR #1011); agentfile's
// BUILD.mk defines the same macro so the class layouts match.
//
// v0.5 limitations:
//   - GET only. POST/PUT/etc. deferred.
//   - Response body capped at 16 KB; rest is truncated and marked.
//   - Single global timeout (30s).
//

#pragma once

#include <cpp-httplib/httplib.h>
#include "http.h"   // common_http_client, common_http_parse_url

#include "chat.h"
#include "tool.h"

#include "tools_common.h"

#include <string>

namespace agentfile {
namespace tools {

class HttpFetchTool : public agent_cpp::Tool {
    static constexpr size_t kMaxBody = 16 * 1024;       // 16 KB
    static constexpr int    kTimeoutSeconds = 30;

  public:
    std::string get_name() const override { return "http_fetch"; }

    common_chat_tool get_definition() const override {
        json schema = {
            {"type", "object"},
            {"properties", {
                {"url",     {{"type", "string"},
                             {"description", "HTTP or HTTPS URL to fetch"}}},
                {"headers", {{"type", "object"},
                             {"description", "Optional request headers as name->value map"}}},
            }},
            {"required", json::array({"url"})},
        };
        return {"http_fetch",
                "Fetch a URL with HTTP GET and return status + headers + body. "
                "Response body is capped at 16 KB. "
                "Use sparingly — this tool grants the model network access.",
                schema.dump()};
    }

    std::string execute(const json &arguments) override {
        if (!arguments.contains("url")) {
            return error_response("missing required parameter: url");
        }
        std::string url = arguments.at("url").get<std::string>();

        try {
            auto [cli, parts] = common_http_client(url);

            cli.set_read_timeout(kTimeoutSeconds, 0);
            cli.set_connection_timeout(kTimeoutSeconds, 0);

            httplib::Headers req_headers;
            if (arguments.contains("headers") &&
                arguments.at("headers").is_object()) {
                for (const auto &[k, v] : arguments.at("headers").items()) {
                    if (v.is_string()) {
                        req_headers.emplace(k, v.get<std::string>());
                    }
                }
            }

            auto res = cli.Get(parts.path.c_str(), req_headers);
            if (!res) {
                return error_response("request failed: " +
                                      httplib::to_string(res.error()));
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

            return result_response({
                {"status", res->status},
                {"headers", resp_headers},
                {"body", body},
                {"truncated", truncated},
            });
        } catch (const std::exception &e) {
            return error_response(std::string("http_fetch failed: ") + e.what());
        }
    }
};

} // namespace tools
} // namespace agentfile
