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

#include "chatbot.h"

#include <string>
#include <sys/stat.h>
#include <vector>

#include "common.h"
#include "llama.h"
#include "color.h"
#include "image.h"
#include "llama.h"  // llamafile wrapper
#include "string.h"

namespace lf {
namespace chatbot {

static bool has_binary(const std::string_view s) {
    return s.find('\0') != std::string_view::npos;
}

// Helper to apply chat template for a single system message
static std::string apply_system_template(const std::string &content) {
    const char *tmpl = g_params->chat_template.empty()
                       ? llama_model_chat_template(g_model, nullptr)
                       : g_params->chat_template.c_str();

    llama_chat_message chat[] = {{"system", content.c_str()}};
    int len = llama_chat_apply_template(tmpl, chat, 1, false, nullptr, 0);
    if (len < 0) {
        return "";
    }

    std::string result(len, '\0');
    llama_chat_apply_template(tmpl, chat, 1, false, &result[0], result.size());
    return result;
}

void on_upload(const std::vector<std::string> &args) {
    if (args.size() < 2) {
        err("error: missing file path" RESET "\n"
            "usage: /upload PATH");
        return;
    }
    if (args.size() > 2) {
        err("error: too many arguments" RESET "\n"
            "usage: /upload PATH");
        return;
    }
    const char *path = args[1].c_str();
    struct stat st;
    if (stat(path, &st) || !S_ISREG(st.st_mode)) {
        err("%s: file does not exist", path);
        return;
    }
    int tokens_used_before = tokens_used();
    std::string content;
    if (!slurp(&content, path)) {
        err("%s: failed to slurp file", path);
        return;
    }
    std::string markdown;
    markdown += "- **Filename**: `";
    markdown += path;
    markdown += "`\n- **Last modified**: ";
    markdown += iso8601(st.st_mtim);
    markdown += "\n\n";
    if (is_image(content)) {
        if (!g_mtmd) {
            err("%s: need --mmproj model to process images", path);
            return;
        }
        print_image(1, content, 80);
        convert_image_to_uri(&markdown, content);
    } else {
        if (has_binary(content)) {
            err("%s: binary file type not supported", path);
            return;
        }
        markdown += "``````";
        markdown += extname(path);
        markdown += '\n';
        markdown += content;
        if (markdown.back() != '\n')
            markdown += '\n';
        markdown += "``````";
    }
    std::string formatted = apply_system_template(markdown);
    if (formatted.empty()) {
        err("failed to apply chat template");
        rewind(tokens_used_before);
        return;
    }
    if (!eval_string(formatted, DONT_ADD_SPECIAL, PARSE_SPECIAL)) {
        rewind(tokens_used_before);
        return;
    }
    llama_synchronize(g_ctx);
}

} // namespace chatbot
} // namespace lf
