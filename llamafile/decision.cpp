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

#include "decision.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

#include "base64.hpp"
#include "common.h"
#include "log.h"
#include "server-common.h"

#include "llamafile.h"

// llama.cpp/tools/server/server.cpp (llamafile patch): answers one
// /v1/systemone request in-process, as the server would
int llamafile_server_decide(common_params &params, const std::string &request, std::string &response);

namespace lf {

const char kDecisionNoChat[] =
    "This model is a decision model without text generation, use /v1/systemone";

ModelOutputs model_outputs(const std::string &model_path) {
    // a file that can't be read gives COMMON_DECISION_TYPE_UNKNOWN, which the
    // server reports as text: the normal load error is left to the caller
    const common_gguf_info info = common_get_gguf_info(model_path);
    const std::vector<std::string> out =
        server_model_output_modalities(info.decision_type, /*has_cap_chat=*/true, /*has_cap_tts=*/false);
    ModelOutputs res;
    res.text = std::find(out.begin(), out.end(), "text") != out.end();
    res.decisions = std::find(out.begin(), out.end(), "decisions") != out.end();
    return res;
}

static bool is_decision_request(const std::string &text) {
    const json body = json::parse_no_throw(text);
    return !body.is_discarded() && body.is_object() && body.contains("questions");
}

// The request text of -p or -f as the user wrote it. common_params_parse()
// rewrites the escape sequences of the prompt (\n, \", \\, ...; params.escape,
// on by default), which breaks a JSON request. As in common_params_parse(),
// the last -p or -f wins; empty means neither was given.
static std::string request_text(const common_params &params, char **argv) {
    if (!params.escape)
        return params.prompt;
    std::string text = params.prompt;
    for (int i = 1; argv[i] && argv[i + 1]; ++i) {
        if (!strcmp(argv[i], "-p") || !strcmp(argv[i], "--prompt")) {
            text = argv[++i];
        } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--file")) {
            std::ifstream f(argv[++i], std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
    }
    return text;
}

bool decision_cli_wanted(const ModelOutputs &outputs, const common_params &params, char **argv) {
    if (!outputs.decisions)
        return false;
    if (!outputs.text)
        return true;
    // a model that also generates text: decide when given a request
    const std::string text = request_text(params, argv);
    return text.empty() ? !isatty(STDIN_FILENO) : is_decision_request(text);
}

static bool read_stream(FILE *f, std::string &out) {
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);
    return !ferror(f);
}

// the server only reads data URLs; mtmd tells images from audio by content,
// so the MIME type is informative only
static const char *media_mime_type(const std::string &path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "png") return "image/png";
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "gif") return "image/gif";
    if (ext == "webp") return "image/webp";
    if (ext == "bmp") return "image/bmp";
    if (ext == "wav") return "audio/wav";
    if (ext == "mp3") return "audio/mpeg";
    if (ext == "flac") return "audio/flac";
    return "application/octet-stream";
}

static bool media_data_url(const std::string &path, std::string &url) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad())
        return false;
    url = std::string("data:") + media_mime_type(path) + ";base64," + base64::encode(bytes);
    return true;
}

static std::string error_message(const std::string &response) {
    const json body = json::parse_no_throw(response);
    if (!body.is_discarded() && body.is_object() && body.contains("error")) {
        const json &e = body.at("error");
        if (e.is_object() && e.contains("message") && e.at("message").is_string())
            return e.at("message").get<std::string>();
        if (e.is_string())
            return e.get<std::string>();
    }
    return response;
}

int decision_cli_main(common_params &params, char **argv) {
    // the request: -p TEXT, -f FILE, or stdin
    std::string request = request_text(params, argv);
    if (request.empty()) {
        if (isatty(STDIN_FILENO)) {
            fprintf(stderr,
                    "error: %s is a decision model: give it a /v1/systemone request as JSON,\n"
                    "       with -p '{...}', -f request.json, or on stdin:\n"
                    "         echo '{\"state\": \"...\", \"questions\": {...}}' | llamafile -m MODEL --cli\n",
                    params.model.path.c_str());
            return 1;
        }
        if (!read_stream(stdin, request)) {
            fprintf(stderr, "error: failed to read the request from stdin\n");
            return 1;
        }
    }

    // --image/--audio FILE: appended to the "files" of the request as data URLs
    if (!params.image.empty()) {
        json body = json::parse_no_throw(request);
        if (body.is_discarded() || !body.is_object()) {
            fprintf(stderr, "error: the request must be a JSON object\n");
            return 1;
        }
        if (!body.contains("files") || body.at("files").is_null())
            body["files"] = json::array();
        if (!body.at("files").is_array()) {
            fprintf(stderr, "error: \"files\" must be an array\n");
            return 1;
        }
        for (const std::string &path : params.image) {
            std::string url;
            if (!media_data_url(path, url)) {
                fprintf(stderr, "error: failed to read %s\n", path.c_str());
                return 1;
            }
            body["files"].push_back(url);
        }
        request = body.dump();
        params.image.clear();
    }

    // Same sandbox as the text --cli: the request is answered in-process, so
    // no network is needed. Must come before the untrusted GGUF is parsed.
    common_log_pause(common_log_main());
    int sandbox = llamafile_sandbox_enter("stdio rpath tty", FLAG_verbose);
    common_log_resume(common_log_main());
    if (sandbox == LLAMAFILE_SANDBOX_FAILED)
        return 1;

    // The text --cli defaults (set before parsing) don't apply here: keep the
    // server's, so the answers match llamafile --server. Decisions read whole
    // prompts in one batch (laya, clef, ...), and their questions share the
    // state across slots.
    if (!llamafile_has(argv, "-b") && !llamafile_has(argv, "--batch-size") && !getenv("LLAMA_ARG_BATCH"))
        params.n_batch = common_params{}.n_batch;
    if (!llamafile_has(argv, "-np") && !llamafile_has(argv, "--parallel") && !getenv("LLAMA_ARG_N_PARALLEL"))
        params.n_parallel = -1; // auto, as the server parses it

    std::string response;
    const int status = llamafile_server_decide(params, request, response);
    if (status < 0) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.path.c_str());
        return 2;
    }
    if (status != 200) {
        fprintf(stderr, "error: %s\n", error_message(response).c_str());
        if (!is_decision_request(request))
            fprintf(stderr, "       a decision model answers a /v1/systemone request, e.g.\n"
                            "       {\"state\": \"...\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"...\"}}}\n");
        return 1;
    }
    fwrite(response.data(), 1, response.size(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return 0;
}

} // namespace lf
