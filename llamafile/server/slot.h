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
#include <cosmo.h>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

#define SLOT(e) DLL_CONTAINER(Slot, elem_, e)

struct llama_context;
struct llama_model;
struct llama_lora_adapter;
struct clip_ctx;

// Function to get the global LoRA adapter
extern "C" struct llama_lora_adapter* llamafiler_get_lora_adapter();

// Function to get multiple LoRA adapters with their scales
extern "C" int llamafiler_get_lora_adapters(struct llama_lora_adapter** adapters, float* scales, int max_adapters);

namespace lf {
namespace server {

using ProgressCallback = std::function<void(int processed, int total)>;

struct Atom;
struct Image;

struct Slot
{
    enum
    {
        uninitialized = -4096,
        out_of_context,
        no_vision_model,
        decode_token_failed,
        decode_image_failed,
        encode_image_failed,
    };

    static const char* describe_error(int);

    int id_;
    Dll elem_;
    time_t last_used_;
    llama_model* model_;
    clip_ctx* clip_ctx_ = nullptr;
    llama_context* ctx_ = nullptr;
    std::vector<Atom> history_;
    std::string system_fingerprint_;
    bool needs_refresh_ = false;

    ~Slot();
    Slot(int, llama_model*);
    int ctx_size() const;
    int ctx_used() const;
    bool start();
    int eval_token(int);
    int eval_tokens(const std::vector<int>&, const ProgressCallback& = nullptr);
    int eval_image(const std::string_view&, const ProgressCallback& = nullptr);
    int eval_atoms(const std::vector<Atom>&, const ProgressCallback& = nullptr);
    int prefill(const std::vector<Atom>&, const ProgressCallback& = nullptr);
    void tokenize(std::vector<Atom>*, std::string_view, bool);
    void dump(std::string*);
    void mark_for_refresh();
};

} // namespace server
} // namespace lf
