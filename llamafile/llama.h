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
#include <string>
#include <string_view>
#include <vector>

// Many llama.cpp APIs take boolean parameters at the end. Please favor
// passing these constants as arguments instead, for better readability

#define ADD_SPECIAL true
#define DONT_ADD_SPECIAL false

#define PARSE_SPECIAL true
#define DONT_PARSE_SPECIAL false

#define ADD_ASSISTANT true
#define DONT_ADD_ASSISTANT false

#define APPLY_GRAMMAR true
#define DONT_APPLY_GRAMMAR false

#define RENDER_SPECIAL_TOKENS true
#define DONT_RENDER_SPECIAL_TOKENS false

struct llama_model;
struct llama_context;

int llamafile_token_eot(llama_model *);

std::string llamafile_token_to_piece(const llama_context *, int, bool);
std::vector<int> llamafile_tokenize(const llama_model *, const std::string_view &, bool, bool);
