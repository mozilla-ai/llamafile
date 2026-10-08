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
// MaxIterationsCallback — --max-iterations: ends the run when a turn makes
// more than N LLM calls.
//

#pragma once

#include "callbacks.h"

#include <stdexcept>

namespace agentfile {

// Distinct type so main can map the iteration cap to its own exit code.
class MaxIterationsExceeded : public std::runtime_error {
  public:
    MaxIterationsExceeded()
        : std::runtime_error(
              "agentfile: agent loop exceeded --max-iterations cap") {}
};

class MaxIterationsCallback : public agent_cpp::Callback {
    int max_;
    int count_ = 0;

  public:
    explicit MaxIterationsCallback(int max) : max_(max) {}

    // The cap is per turn: each -i follow-up starts again from zero.
    void before_agent_loop(std::vector<common_chat_msg> & /*messages*/) override {
        count_ = 0;
    }

    void before_llm_call(std::vector<common_chat_msg> & /*messages*/) override {
        if (max_ > 0 && ++count_ > max_) {
            throw MaxIterationsExceeded();
        }
    }
};

} // namespace agentfile
