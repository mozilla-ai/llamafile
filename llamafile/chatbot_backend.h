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

#pragma once
#include <functional>
#include <string>
#include <vector>

#include "chat.h"

namespace lf {
namespace chatbot {

// Callback for streaming tokens. Return false to stop generation.
using TokenCallback = std::function<bool(const std::string &content_delta,
                                         const std::string &reasoning_delta)>;

// Abstract inference backend for the chatbot REPL.
// DirectBackend: wraps llama_decode (used by --chat mode)
// ApiBackend: HTTP client to /v1/chat/completions (used by combined mode)
class ChatBackend {
public:
    virtual ~ChatBackend() = default;

    // Send messages and stream the response.
    // Calls on_token for each streamed chunk.
    // Returns the full assistant content (no reasoning).
    virtual std::string complete(
        const std::vector<common_chat_msg> &messages,
        TokenCallback on_token) = 0;

    // Context info
    virtual int context_used() = 0;
    virtual int context_max() = 0;

    // Stats
    virtual void print_stats() = 0;

    // History management
    virtual void on_clear() = 0;
    virtual void on_push() = 0;
    virtual void on_pop() = 0;
    virtual void on_undo() = 0;
    virtual void on_forget(int n) = 0;

    // Whether this backend supports token-level dump
    virtual bool supports_dump() { return false; }
    virtual void on_dump(int fd) {}

    // Whether this backend supports manual mode (role cycling)
    virtual bool supports_manual_mode() { return false; }
};

} // namespace chatbot
} // namespace lf
