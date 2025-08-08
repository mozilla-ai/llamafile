// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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

#pragma once
#include "buffer.h"
#include <ctime>
#include <libc/fmt/itoa.h>
#include <libc/str/slice.h>
#include <net/http/http.h>
#include <net/http/url.h>
#include <optional>
#include <string>
#include <sys/resource.h>

#define HasHeader(H) (!!msg_.headers[H].a)
#define HeaderData(H) (ibuf_.p + msg_.headers[H].a)
#define HeaderLength(H) (msg_.headers[H].b - msg_.headers[H].a)
#define HeaderEqual(H, S) \
    SlicesEqual(S, strlen(S), HeaderData(H), HeaderLength(H))
#define HeaderEqualCase(H, S) \
    SlicesEqualCase(S, strlen(S), HeaderData(H), HeaderLength(H))

struct llama_model;
struct llama_lora_adapter;

namespace jt {
struct Json;
}

namespace lf {
namespace server {

struct Cleanup;
struct Slot;
struct Worker;
struct TokenizeParams;
struct EmbeddingParams;
struct V1CompletionParams;
struct V1ChatCompletionParams;

struct Client
{
    int fd_ = -1;
    unsigned client_ip_ = 0;
    unsigned effective_ip_ = 0;
    bool client_ip_trusted_ = false;
    bool effective_ip_trusted_ = false;
    bool close_connection_ = false;
    bool should_send_error_if_canceled_;
    size_t unread_ = 0;
    Worker* worker_; // borrowed
    Slot* slot_ = nullptr; // owned or null
    llama_model* model_; // borrowed
    timespec message_started_;
    HttpMessage msg_;
    Url url_ = {};
    char* url_memory_ = nullptr;
    char* params_memory_ = nullptr;
    std::string_view payload_;
    std::string resolved_;
    std::string dump_;
    Cleanup* cleanups_;
    Buffer ibuf_;
    Buffer obuf_;

    explicit Client(llama_model*);

    void run();
    int close();
    void clear();
    void cleanup();
    bool transport() __wur;
    bool synchronize() __wur;
    bool read_payload() __wur;
    bool read_request() __wur;
    bool read_content() __wur;
    bool send_continue() __wur;
    bool send(const std::string_view) __wur;
    bool send_binary(const void*, size_t) __wur;
    void defer_cleanup(void (*)(void*), void*);
    bool send_error(int, const char* = nullptr);
    char* append_http_response_message(char*, int, const char* = nullptr);
    bool send_response(char*, char*, const std::string_view) __wur;
    bool send_response_start(char*, char*) __wur;
    bool send_response_chunk(const std::string_view) __wur;
    bool send_response_finish() __wur;
    bool send2(const std::string_view, const std::string_view) __wur;
    char* append_header(const std::string_view, const std::string_view);
    bool has_at_most_this_element(int, const std::string_view);
    std::string_view get_header(const std::string_view&);
    bool fun() __wur;

    std::string_view path();
    std::optional<std::string_view> param(std::string_view);

    bool dispatch() __wur;
    bool dispatcher() __wur;

    bool tokenize() __wur;
    bool get_tokenize_params(TokenizeParams*) __wur;

    bool embedding() __wur;
    bool get_embedding_params(EmbeddingParams*) __wur;

    bool v1_completions() __wur;
    bool get_v1_completions_params(V1CompletionParams*) __wur;

    bool v1_chat_completions() __wur;
    bool get_v1_chat_completions_params(V1ChatCompletionParams*) __wur;

    bool v1_models() __wur;

    bool slotz() __wur;
    bool flagz() __wur;
    bool lora_adapters() __wur;
    bool handle_apply_adapters(jt::Json&) __wur;
    bool handle_load_adapter(jt::Json&) __wur;
    bool handle_clear_adapters() __wur;
    bool handle_upstream_lora_apply(jt::Json&) __wur;
    bool db_chat(int64_t) __wur;
    bool db_chats() __wur;
    bool db_message(int64_t) __wur;
    bool db_messages(int64_t) __wur;
};

} // namespace server
} // namespace lf

// Global LoRA adapter storage - extern declarations (outside namespace to match definitions in prog.cpp)
#define MAX_LORA_ADAPTERS 8
struct lora_adapter_container {
    struct llama_lora_adapter* adapter;
    float scale;
    std::string name;  // Model/adapter name for identification
    bool applied;      // Whether this adapter is currently applied to slots
};

extern struct lora_adapter_container g_lora_adapters[MAX_LORA_ADAPTERS];
extern int g_lora_adapters_count;
