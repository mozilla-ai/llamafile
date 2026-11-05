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

// Decision models ("system one" models) answer typed questions about a state
// with one probability per option, in one forward pass, instead of generating
// text. llama.cpp serves them on POST /v1/systemone (see docs/api.md).

#pragma once
#include <string>

struct common_params;
struct llama_model;

namespace lf {

// What a model file can do, read from its GGUF metadata without loading it.
// Mirrors the "output_modalities" that the server reports in /v1/models.
struct ModelOutputs {
    bool text = true;       // text generation (chat, completions)
    bool decisions = false; // POST /v1/systemone
};
ModelOutputs model_outputs(const std::string &model_path);
// same, for a loaded model
ModelOutputs model_outputs(const llama_model *model);

// The error llama.cpp gives when a decision-only model is asked to chat
extern const char kDecisionNoChat[];

// True if --cli should answer a /v1/systemone request instead of generating
// text: always for a decision-only model; for a model that does both, when
// the prompt (-p or -f) is a JSON object with "questions", or comes from a pipe.
bool decision_cli_wanted(const ModelOutputs &outputs, const common_params &params, char **argv);

// --cli with a decision model: reads one /v1/systemone request (-p, -f or
// stdin), adds the --image/--audio files to it, answers it in-process and
// prints the response JSON on stdout. Returns the process exit code.
// The caller enters the --cli sandbox first: this needs only stdio and rpath.
int decision_cli_main(common_params &params, char **argv);

} // namespace lf
