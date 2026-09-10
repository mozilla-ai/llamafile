// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// MaxIterationsCallback — throws MaxIterationsExceeded when a turn makes
// more than N LLM calls. Pairs with --max-iterations; main maps the
// exception to its own exit code.
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

    // The cap is per user turn: each run_loop invocation (one-shot run, or
    // one --interactive follow-up) gets a fresh budget.
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
