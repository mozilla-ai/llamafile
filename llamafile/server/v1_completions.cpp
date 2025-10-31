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

#include "client.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/sampling.h"
#include "llamafile/llamafile.h"
#include "llamafile/json.h"
#include "llamafile/llama.h"
#include "llamafile/macros.h"
#include "llamafile/server/atom.h"
#include "llamafile/server/cleanup.h"
#include "llamafile/server/fastjson.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/slot.h"
#include "llamafile/server/slots.h"
#include "llamafile/server/utils.h"
#include "llamafile/server/worker.h"
#include "llamafile/string.h"
#include "llamafile/vector.h"
#include <cmath>
#include <cstring>
#include <sys/resource.h>
#include <vector>

using jt::Json;

namespace lf {
namespace server {

struct V1CompletionParams
{
    bool echo = false;
    bool stream = false;
    bool stream_include_usage = false;
    long max_tokens = -1;
    long seed = FLAG_seed;
    double top_p = FLAG_top_p;
    double min_p = FLAG_min_p;
    long top_k = FLAG_top_k;
    double temperature = FLAG_temperature;
    double presence_penalty = 0;
    double frequency_penalty = 0;
    std::string user;
    std::string model;
    std::string prompt;
    std::vector<std::vector<Atom>> stop;

    void add_stop(llama_model* model, const std::string& text)
    {
        stop.emplace_back();
        atomize(model, &stop.back(), text, DONT_PARSE_SPECIAL);
    }

    bool should_stop(const std::vector<Atom>& history)
    {
        for (const auto& s : stop)
            if (vector_ends_with(history, s))
                return true;
        return false;
    }
};

struct V1CompletionState
{
    std::vector<Atom> atoms;
    std::string piece = "";
};

struct V1CompletionResponse
{
    std::string content;
    Json json;
};

static void
cleanup_params(void* arg)
{
    delete (V1CompletionParams*)arg;
}

static void
cleanup_state(void* arg)
{
    delete (V1CompletionState*)arg;
}

static void
cleanup_response(void* arg)
{
    delete (V1CompletionResponse*)arg;
}

static void
cleanup_sampler(void* arg)
{
    llama_sampling_free((llama_sampling_context*)arg);
}

static void
cleanup_slot(void* arg)
{
    Client* client = (Client*)arg;
    if (client->slot_) {
        client->worker_->server_->slots_->give(client->slot_);
        client->slot_ = nullptr;
    }
}

static bool
is_legal_role(const std::string_view& role)
{
    return role == "system" || //
           role == "user" || //
           role == "assistant";
}

static std::string
generate_id()
{
    std::string b = "cmpl-";
    for (int i = 0; i < 2; ++i) {
        uint64_t w = _rand64();
        for (int j = 0; j < 64 / 5; ++j) {
            b += "abcdefghijklmnopqrstuvwxyz012345"[w & 31];
            w >>= 5;
        }
    }
    return b;
}

static llama_sampling_context*
create_sampler(const V1CompletionParams* params)
{
    llama_sampling_params sparams;
    sparams.temp = params->temperature;
    sparams.top_p = params->top_p;
    sparams.min_p = params->min_p;
    sparams.top_k = params->top_k;
    sparams.penalty_freq = params->frequency_penalty;
    sparams.penalty_present = params->presence_penalty;
    sparams.seed = params->seed;
    return llama_sampling_init(sparams);
}

static std::string
make_event(const Json& json)
{
    std::string s = "data: ";
    s += json.toString();
    s += "\n\n";
    return s;
}

bool
Client::get_v1_completions_params(V1CompletionParams* params)
{
    // must be json post request
    if (msg_.method != kHttpPost)
        return send_error(405);
    if (!HasHeader(kHttpContentType) ||
        !IsMimeType(HeaderData(kHttpContentType),
                    HeaderLength(kHttpContentType),
                    "application/json")) {
        return send_error(501, "Content Type Not Implemented");
    }
    if (!read_payload())
        return false;

    // object<model, messages, ...>
    auto [status, json] = Json::parse(std::string(payload_));
    if (status != Json::success)
        return send_error(400, Json::StatusToString(status));
    if (!json.isObject())
        return send_error(400, "JSON body must be an object");

    // fields openai documents that we don't support
    if (json.contains("suffix"))
        return send_error(400, "OpenAI suffix field not supported");

    // model: string
    Json& model = json["model"];
    if (!model.isString())
        return send_error(400, "JSON missing model string");
    params->model = model.getString();

    // prompt: string
    if (!json["prompt"].isString())
        return send_error(400, "JSON missing prompt string");
    params->prompt = json["prompt"].getString();

    // n: integer|null
    //
    // How many chat completion choices to generate for each input
    // message. Note that you will be charged based on the number of
    // generated tokens across all of the choices. Keep n as 1 to
    // minimize costs.
    Json& n = json["n"];
    if (!n.isNull()) {
        if (!n.isLong())
            return send_error(400, "n field must be integer");
        if (n.getLong() != 1)
            return send_error(400, "n field must be 1 if specified");
    }

    // best_of: integer|null
    //
    // Generates best_of completions server-side and returns the "best"
    // (the one with the highest log probability per token). Results
    // cannot be streamed. When used with n, best_of controls the number
    // of candidate completions and n specifies how many to return –
    // best_of must be greater than n.
    Json& best_of = json["best_of"];
    if (!best_of.isNull()) {
        if (!best_of.isLong())
            return send_error(400, "best_of field must be integer");
        if (best_of.getLong() != 1)
            return send_error(400, "best_of field must be 1 if specified");
    }

    // echo: bool|null
    //
    // Echo back the prompt in addition to the completion.
    Json& echo = json["echo"];
    if (!echo.isNull()) {
        if (!echo.isBool())
            return send_error(400, "echo field must be boolean");
        params->echo = echo.getBool();
    }

    // stream: bool|null
    //
    // If set, partial message deltas will be sent, like in ChatGPT.
    // Tokens will be sent as data-only server-sent events as they
    // become available, with the stream terminated by a data: [DONE]
    // message.
    Json& stream = json["stream"];
    if (!stream.isNull()) {
        if (!stream.isBool())
            return send_error(400, "stream field must be boolean");
        params->stream = stream.getBool();

        // stream_options: object|null
        //
        // Options for the streaming response.
        Json& stream_options = json["stream_options"];
        if (!stream_options.isNull()) {
            if (!stream_options.isObject())
                return send_error(400, "stream_options field must be object");

            // include_usage: bool|null
            //
            // Include usage also for streaming responses. The actual usage will be reported before
            // the [DONE] message, but all chunks contain an empty usage field.
            Json& include_usage = stream_options["include_usage"];
            if (!include_usage.isNull()) {
                if (!include_usage.isBool())
                    return send_error(400, "include_usage field must be boolean");
                params->stream_include_usage = include_usage.getBool();
            }
        }
    }

    // max_tokens: integer|null
    //
    // An upper bound for the number of tokens that can be generated for
    // a completion. This can be used to control compute costs.
    Json& max_tokens = json["max_tokens"];
    if (!max_tokens.isNull()) {
        if (!max_tokens.isLong())
            return send_error(400, "max_tokens must be integer");
        params->max_tokens = max_tokens.getLong();
    }
    Json& max_completion_tokens = json["max_completion_tokens"];
    if (!max_completion_tokens.isNull()) {
        if (!max_completion_tokens.isLong())
            return send_error(400, "max_completion_tokens must be integer");
        params->max_tokens = max_completion_tokens.getNumber();
    }

    // top_p: number|null
    //
    // An alternative to sampling with temperature, called nucleus
    // sampling, where the model considers the results of the tokens
    // with top_p probability mass. So 0.1 means only the tokens
    // comprising the top 10% probability mass are considered.
    //
    // We generally recommend altering this or temperature but not both.
    Json& top_p = json["top_p"];
    if (!top_p.isNull()) {
        if (!top_p.isNumber())
            return send_error(400, "top_p must be number");
        params->top_p = top_p.getNumber();
        if (!(0 <= params->top_p && params->top_p <= 1))
            return send_error(400, "top_p must be between 0 and 1");
    }

    // min_p: number|null
    Json& min_p = json["min_p"];
    if (!min_p.isNull()) {
        if (!min_p.isNumber())
            return send_error(400, "min_p must be number");
        params->min_p = min_p.getNumber();
        if (!(0 <= params->min_p && params->min_p <= 1))
            return send_error(400, "min_p must be between 0 and 1");
    }

    // top_k: integer|null
    Json& top_k = json["top_k"];
    if (!top_k.isNull()) {
        if (!top_k.isNumber())
            return send_error(400, "top_k must be number");
        params->top_k = top_k.getNumber();
        if (params->top_k < 0)
            return send_error(400, "top_k must be non-negative");
    }

    // temperature: number|null
    //
    // What sampling temperature to use, between 0 and 2. Higher values
    // like 0.8 will make the output more random, while lower values
    // like 0.2 will make it more focused and deterministic.
    //
    // We generally recommend altering this or top_p but not both.
    Json& temperature = json["temperature"];
    if (!temperature.isNull()) {
        if (!temperature.isNumber())
            return send_error(400, "temperature must be number");
        params->temperature = temperature.getNumber();
        if (!(0 <= params->temperature && params->temperature <= 2))
            return send_error(400, "temperature must be between 0 and 2");
    }

    // seed: integer|null
    //
    // If specified, our system will make a best effort to sample
    // deterministically, such that repeated requests with the same seed
    // and parameters should return the same result. Determinism is not
    // guaranteed, and you should refer to the system_fingerprint
    // response parameter to monitor changes in the backend.
    Json& seed = json["seed"];
    if (!seed.isNull()) {
        if (!seed.isLong())
            return send_error(400, "seed must be integer");
        params->seed = seed.getLong();
    }

    // presence_penalty: number|null
    //
    // Number between -2.0 and 2.0. Positive values penalize new tokens
    // based on whether they appear in the text so far, increasing the
    // model's likelihood to talk about new topics.
    Json& presence_penalty = json["presence_penalty"];
    if (!presence_penalty.isNull()) {
        if (!presence_penalty.isNumber())
            return send_error(400, "presence_penalty must be number");
        params->presence_penalty = presence_penalty.getNumber();
        if (!(-2 <= params->presence_penalty && params->presence_penalty <= 2))
            return send_error(400, "presence_penalty must be between -2 and 2");
    }

    // frequency_penalty: number|null
    //
    // Number between -2.0 and 2.0. Positive values penalize new tokens
    // based on their existing frequency in the text so far, decreasing
    // the model's likelihood to repeat the same line verbatim.
    Json& frequency_penalty = json["frequency_penalty"];
    if (!frequency_penalty.isNull()) {
        if (!frequency_penalty.isNumber())
            return send_error(400, "frequency_penalty must be number");
        params->frequency_penalty = frequency_penalty.getNumber();
        if (!(-2 <= params->frequency_penalty &&
              params->frequency_penalty <= 2))
            return send_error(400, "frequency_penalty must be -2 through 2");
    }

    // user: string|null
    //
    // A unique identifier representing your end-user, which can help
    // llamafiler to monitor and detect abuse.
    Json& user = json["user"];
    if (!user.isNull()) {
        if (!user.isString())
            return send_error(400, "JSON missing user string");
        params->user = user.getString();
    }

    // stop: string|array<string>|null
    //
    // Up to 4 sequences where the API will stop generating further tokens.
    Json& stop = json["stop"];
    if (!stop.isNull()) {
        if (stop.isString()) {
            params->add_stop(model_, stop.getString());
        } else if (stop.isArray()) {
            std::vector<Json>& stops = stop.getArray();
            if (stops.size() > 4)
                return send_error(400, "stop array must have 4 items or fewer");
            for (Json& stop2 : stops) {
                if (!stop2.isString())
                    return send_error(400, "stop array item must be string");
                if (stop2.getString().size() > 50)
                    return send_error(400, "stop array string too long");
                params->add_stop(model_, stop2.getString());
            }
        } else {
            return send_error(400, "stop field must be string or string array");
        }
    }

    return true;
}

bool
Client::v1_completions()
{
    // get parameters
    auto params = new V1CompletionParams;
    defer_cleanup(cleanup_params, params);
    if (!get_v1_completions_params(params))
        return false;

    // create state and response objects
    V1CompletionState* state = new V1CompletionState;
    defer_cleanup(cleanup_state, state);
    V1CompletionResponse* response = new V1CompletionResponse;
    defer_cleanup(cleanup_response, response);

    // add bos token if it's needed
    if (llama_should_add_bos_token(model_))
        state->atoms.emplace_back(llama_token_bos(model_));

    // turn text into tokens
    atomize(model_, &state->atoms, params->prompt, PARSE_SPECIAL);

    // we don't support multiple images yet
    state->atoms = remove_old_image_atoms(state->atoms);

    // find appropriate slot
    slot_ = worker_->server_->slots_->take(state->atoms);
    defer_cleanup(cleanup_slot, this);

    // init sampling
    llama_sampling_context* sampler = create_sampler(params);
    if (!sampler)
        return send_error(500, "failed to create sampler");
    defer_cleanup(cleanup_sampler, sampler);

    // prefill time
    int prompt_tokens = 0;
    if ((prompt_tokens = slot_->prefill(state->atoms)) < 0) {
        SLOG("slot prefill failed: %s", Slot::describe_error(prompt_tokens));
        return send_error(500, Slot::describe_error(prompt_tokens));
    }

    // setup response json
    response->json["id"] = generate_id();
    response->json["object"] = "text_completion";
    response->json["model"] = params->model;
    response->json["system_fingerprint"] = slot_->system_fingerprint_;
    Json& choice = response->json["choices"][0];
    choice["index"] = 0;
    choice["logprobs"] = nullptr;
    choice["finish_reason"] = nullptr;

    // initialize response
    if (params->stream) {
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: text/event-stream\r\n");
        if (!send_response_start(obuf_.p, p))
            return false;
        choice["delta"]["role"] = "assistant";
        choice["delta"]["content"] = "";
        response->json["created"] = timespec_real().tv_sec;
        if (params->stream_include_usage)
            response->json["usage"] = nullptr;
        response->content = make_event(response->json);
        choice.getObject().erase("delta");
        if (!send_response_chunk(response->content))
            return false;
    }

    // prediction time
    int completion_tokens = 0;
    const char* finish_reason = "length";
    for (;;) {
        if (params->max_tokens >= 0 &&
            completion_tokens >= params->max_tokens) {
            slot_->eval_token(llamafile_token_eot(model_));
            break;
        }
        llama_token id = llama_sampling_sample(sampler, slot_->ctx_, NULL);
        llama_sampling_accept(sampler, slot_->ctx_, id, DONT_APPLY_GRAMMAR);
        ++completion_tokens;
        if (slot_->eval_token(id) < 0) {
            SLOG("ran out of context window");
            break;
        }
        if (llama_token_is_eog(model_, id)) {
            finish_reason = "stop";
            break;
        }
        if (params->should_stop(slot_->history_)) {
            slot_->eval_token(llamafile_token_eot(model_));
            finish_reason = "stop";
            break;
        }
        state->piece +=
          llamafile_token_to_piece(slot_->ctx_, id, DONT_RENDER_SPECIAL_TOKENS);
        if (!state->piece.empty()) {
            if (params->stream) {
                if (!ends_with_incomplete_utf8(state->piece)) {
                    char* p = append_http_response_message(obuf_.p, 200);
                    choice["text"] = state->piece;
                    response->json["created"] = timespec_real().tv_sec;
                    response->content = make_event(response->json);
                    if (!send_response_chunk(response->content))
                        return false;
                    state->piece.clear();
                }
            } else {
                response->content += state->piece;
                state->piece.clear();
            }
        }
    }
    choice["finish_reason"] = finish_reason;

    // finalize response
    cleanup_slot(this);
    if (params->stream) {
        choice["text"] = "";
        response->json["created"] = timespec_real().tv_sec;
        if (params->stream_include_usage) {
            Json& usage = response->json["usage"];
            usage["prompt_tokens"] = prompt_tokens;
            usage["completion_tokens"] = completion_tokens;
            usage["total_tokens"] = completion_tokens + prompt_tokens;
        }
        response->content = make_event(response->json);
        if (!send_response_chunk(response->content))
            return false;
        if (!send_response_chunk("data: [DONE]\n\n"))
            return false;
        return send_response_finish();
    } else {
        Json& usage = response->json["usage"];
        usage["prompt_tokens"] = prompt_tokens;
        usage["completion_tokens"] = completion_tokens;
        usage["total_tokens"] = completion_tokens + prompt_tokens;
        choice["text"] = std::move(response->content);
        response->json["created"] = timespec_real().tv_sec;
        char* p = append_http_response_message(obuf_.p, 200);
        p = stpcpy(p, "Content-Type: application/json\r\n");
        response->content = response->json.toStringPretty();
        response->content += '\n';
        return send_response(obuf_.p, p, response->content);
    }
}

} // namespace server
} // namespace lf
