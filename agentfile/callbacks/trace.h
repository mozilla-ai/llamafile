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
// OtlpTraceCallback — writes spans in the OTLP/JSON encoding, one
// ExportTraceServiceRequest per line. This is the standard OTLP wire format
// in its JSON flavor: the OpenTelemetry Collector ingests these files
// directly via the `otlpjsonfile` receiver, no SDK required on our side.
//
// Span structure and GenAI semconv attributes mirror agent.cpp's
// examples/tracing OpenTelemetryCallbacks:
//
//   invoke_agent <agent>            (root)
//   ├── chat <model>                per LLM call
//   └── execute_tool <tool>         per tool execution
//
// A span's request line is written when the span ends; the root span is
// written by after_agent_loop, or by the destructor (with error status) if
// the loop terminated by exception.
//

#pragma once

#include "callbacks.h"
#include "tool_result.h"

#include "server-common.h"  // safe_json_to_str
#include "util.h"

#include <cstdio>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace agentfile {

class OtlpTraceCallback : public agent_cpp::Callback {
    using json = nlohmann::ordered_json;  // what safe_json_to_str takes

    struct OpenSpan {
        std::string span_id;
        int64_t start_ns = 0;
        bool open = false;
    };

    FILE *file_;
    std::string model_name_;
    std::string agent_name_;
    std::string provider_name_;
    std::string trace_id_;
    OpenSpan agent_, chat_, tool_;
    std::string tool_name_;

  public:
    OtlpTraceCallback(const std::string &path, const std::string &model_name,
                      const std::string &agent_name = "agentfile",
                      const std::string &provider_name = "llamafile")
        : model_name_(model_name), agent_name_(agent_name),
          provider_name_(provider_name) {
        file_ = std::fopen(path.c_str(), "w");
        if (!file_) {
            throw std::runtime_error("--trace: cannot open " + path);
        }
    }

    ~OtlpTraceCallback() override {
        // Loop terminated by exception: close whatever is still open so the
        // trace file stays a complete, ingestible record.
        if (chat_.open) end_chat("error", "aborted", "aborted");
        if (tool_.open) {
            end_span(tool_, "execute_tool " + tool_name_, tool_attributes(),
                     "aborted", "aborted");
        }
        if (agent_.open) {
            end_span(agent_, "invoke_agent " + agent_name_,
                     agent_attributes(), "aborted", "aborted");
        }
        if (file_) std::fclose(file_);
    }

    OtlpTraceCallback(const OtlpTraceCallback &) = delete;
    OtlpTraceCallback &operator=(const OtlpTraceCallback &) = delete;

    void before_agent_loop(std::vector<common_chat_msg> &) override {
        trace_id_ = random_hex(32);
        start(agent_);
    }

    void after_agent_loop(std::vector<common_chat_msg> &,
                          std::string &) override {
        end_span(agent_, "invoke_agent " + agent_name_, agent_attributes());
    }

    void before_llm_call(std::vector<common_chat_msg> &) override {
        start(chat_);
    }

    void after_llm_call(common_chat_msg &parsed_msg) override {
        end_chat(parsed_msg.tool_calls.empty() ? "stop" : "tool_calls");
    }

    void before_tool_execution(std::string &tool_name, std::string &) override {
        tool_name_ = tool_name;
        start(tool_);
    }

    void after_tool_execution(std::string &tool_name,
                              agent_cpp::ToolResult &result) override {
        // If an earlier callback skipped the tool, before_tool_execution
        // never ran here; synthesize a zero-length span.
        if (!tool_.open) {
            tool_name_ = tool_name;
            start(tool_);
        }
        end_span(tool_, "execute_tool " + tool_name_, tool_attributes(),
                 result.has_error() ? result.error().message : "");
    }

  private:
    void start(OpenSpan &s) {
        s.span_id = random_hex(16);
        s.start_ns = unix_ns_now();
        s.open = true;
    }

    static json attr(const std::string &key, const std::string &value) {
        return {{"key", key}, {"value", {{"stringValue", value}}}};
    }

    json agent_attributes() const {
        return json::array({
            attr("gen_ai.operation.name", "invoke_agent"),
            attr("gen_ai.provider.name", provider_name_),
            attr("gen_ai.agent.name", agent_name_),
            attr("gen_ai.request.model", model_name_),
        });
    }

    json tool_attributes() const {
        return json::array({
            attr("gen_ai.operation.name", "execute_tool"),
            attr("gen_ai.tool.name", tool_name_),
            attr("gen_ai.tool.type", "function"),
        });
    }

    void end_chat(const std::string &finish_reason,
                  const std::string &error_message = "",
                  const std::string &error_type = "") {
        json attrs = json::array({
            attr("gen_ai.operation.name", "chat"),
            attr("gen_ai.provider.name", provider_name_),
            attr("gen_ai.request.model", model_name_),
            attr("gen_ai.output.type", "text"),
        });
        attrs.push_back(
            {{"key", "gen_ai.response.finish_reasons"},
             {"value",
              {{"arrayValue",
                {{"values", json::array({{{"stringValue", finish_reason}}})}}}}}});
        end_span(chat_, "chat " + model_name_, attrs, error_message,
                 error_type);
    }

    // Ends the span and writes its ExportTraceServiceRequest line.
    // A non-empty error_message sets STATUS_CODE_ERROR per semconv.
    void end_span(OpenSpan &s, const std::string &name, json attributes,
                  const std::string &error_message = "",
                  const std::string &error_type = "tool_execution_error") {
        if (!s.open) return;
        s.open = false;

        json span = {
            {"traceId", trace_id_},
            {"spanId", s.span_id},
            {"name", name},
            {"kind", 1}, // SPAN_KIND_INTERNAL
            {"startTimeUnixNano", std::to_string(s.start_ns)},
            {"endTimeUnixNano", std::to_string(unix_ns_now())},
            {"attributes", std::move(attributes)},
        };
        if (&s != &agent_ && agent_.open) {
            span["parentSpanId"] = agent_.span_id;
        }
        if (!error_message.empty()) {
            span["attributes"].push_back(attr("error.type", error_type));
            span["status"] = {{"code", 2}, // STATUS_CODE_ERROR
                              {"message", error_message}};
        } else {
            span["status"] = json::object();
        }

        json request = {
            {"resourceSpans",
             json::array(
                 {{{"resource",
                    {{"attributes",
                      json::array({attr("service.name", "agentfile")})}}},
                   {"scopeSpans",
                    json::array({{{"scope", {{"name", "agentfile"}}},
                                  {"spans", json::array({span})}}})}}})},
        };

        // safe_json_to_str never throws on invalid UTF-8 (a model path or
        // error message can carry it): this also runs from the destructor,
        // where a throw is std::terminate.
        std::string line = safe_json_to_str(request);
        std::fwrite(line.data(), 1, line.size(), file_);
        std::fputc('\n', file_);
        std::fflush(file_);
    }
};

} // namespace agentfile
