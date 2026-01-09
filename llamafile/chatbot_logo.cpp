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

#include <cstdio>

#include "color.h"
#include "llamafile.h"

namespace lf {
namespace chatbot {

static void print_logo(const char16_t *s) {
    for (int i = 0; s[i]; ++i) {
        switch (s[i]) {
        case u'█':
            printf(GREEN "█" UNFOREGROUND);
            break;
        case u'╚':
        case u'═':
        case u'╝':
        case u'╗':
        case u'║':
        case u'╔':
            printf(FAINT "%C" UNBOLD, s[i]);
            break;
        default:
            printf("%C", s[i]);
            break;
        }
    }
}

void logo(char **argv) {
    if (llamafile_has(argv, "--nologo"))
        return;
    if (llamafile_has(argv, "--ascii")) {
        printf("\
 _ _                        __ _ _\n\
| | | __ _ _ __ ___   __ _ / _(_) | ___\n\
| | |/ _` | '_ ` _ \\ / _` | |_| | |/ _ \\\n\
| | | (_| | | | | | | (_| |  _| | |  __/\n\
|_|_|\\__,_|_| |_| |_|\\__,_|_| |_|_|\\___|\n");
    } else {
        print_logo(u"\n\
██╗     ██╗      █████╗ ███╗   ███╗ █████╗ ███████╗██╗██╗     ███████╗\n\
██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║██║     ██╔════╝\n\
██║     ██║     ███████║██╔████╔██║███████║█████╗  ██║██║     █████╗\n\
██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══╝  ██║██║     ██╔══╝\n\
███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║     ██║███████╗███████╗\n\
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝\n");
    }
}

} // namespace chatbot
} // namespace lf
