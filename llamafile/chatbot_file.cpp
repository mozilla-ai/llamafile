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
        // In direct mode, need multimodal context loaded locally.
        // In API mode (g_model==null), the server handles multimodal.
        if (g_model && !g_mtmd) {
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
    // Store content for inclusion with next user message.
    // This avoids template validation errors in models like Qwen3.5 that
    // require user messages to be present when applying the template.
    if (!g_pending_file_content.empty()) {
        g_pending_file_content += "\n\n";
    }
    g_pending_file_content += markdown;
}

} // namespace chatbot
} // namespace lf
