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

#include <cstring>
#include <glob.h>
#include <string>
#include <sys/stat.h>

#include "bestline.h"

namespace lf {
namespace chatbot {

static bool starts_with(const char *str, const char *prefix) {
    return strncmp(str, prefix, strlen(prefix)) == 0;
}

static bool is_directory(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode);
}

void on_completion(const char *line, int pos, bestlineCompletions *comp) {
    if (starts_with(line, "/upload ")) {
        std::string pattern(line + strlen("/upload "));
        pattern += '*';
        glob_t gl;
        if (!glob(pattern.c_str(), GLOB_TILDE, 0, &gl)) {
            for (size_t i = 0; i < gl.gl_pathc; ++i) {
                std::string completion = "/upload ";
                completion += gl.gl_pathv[i];
                if (is_directory(gl.gl_pathv[i]))
                    completion += '/';
                bestlineAddCompletion(comp, completion.c_str());
            }
            globfree(&gl);
        }
    } else {
        static const char *const kCompletions[] = {
            "/clear", // usage: /clear
            "/context", // usage: /context
            "/dump", // usage: /dump [FILE]
            "/exit", // usage: /exit
            "/forget", // usage: /forget
            "/help", // usage: /help [COMMAND]
            "/manual", // usage: /manual [on|off]
            "/pop", // usage: /pop
            "/push", // usage: /push
            "/stack", // usage: /stack
            "/stats", // usage: /stats
            "/undo", // usage: /undo
            "/upload", // usage: /upload FILE
        };
        for (int i = 0; i < sizeof(kCompletions) / sizeof(*kCompletions); ++i)
            if (starts_with(kCompletions[i], line))
                bestlineAddCompletion(comp, kCompletions[i]);
    }
}

} // namespace chatbot
} // namespace lf
