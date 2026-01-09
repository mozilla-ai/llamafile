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

#include "color.h"

namespace lf {
namespace chatbot {

static const char *on_hint_impl(const char *line) {
    if (!*line && g_manual_mode)
        return get_role_name(g_role);
    if (!*line && !g_manual_mode && !g_said_something) {
        if (is_base_model()) {
            return "type text to be completed (or /help for help)";
        } else {
            return "say something (or type /help for help)";
        }
    }
    static const char *const kHints[] = {
        "/clear", //
        "/context", //
        "/dump", //
        "/exit", //
        "/forget", //
        "/help", //
        "/manual", //
        "/pop", //
        "/push", //
        "/stack", //
        "/stats", //
        "/undo", //
        "/upload", //
    };
    int z = strlen(line);
    int n = sizeof(kHints) / sizeof(kHints[0]);
    int l = 0;
    int r = n - 1;
    int i = -1;
    while (l <= r) {
        int m = (l & r) + ((l ^ r) >> 1); // floor((a+b)/2)
        int c = strncmp(line, kHints[m], z);
        if (!c) {
            i = m;
            r = m - 1;
        } else if (c < 0) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    if (i == -1 || (i + 1 < n && !strncmp(line, kHints[i + 1], z)))
        return "";
    return kHints[i] + z;
}

char *on_hint(const char *line, const char **ansi1, const char **ansi2) {
    *ansi1 = FAINT;
    *ansi2 = UNBOLD;
    return strdup(on_hint_impl(line));
}

} // namespace chatbot
} // namespace lf
