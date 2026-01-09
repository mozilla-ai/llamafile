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

#include "xterm.h"

#include <cosmo.h>
#include <stdlib.h>
#include <string.h>

#define SQR(X) ((X) * (X))
#define UNCUBE(x) x < 48 ? 0 : x < 115 ? 1 : (x - 35) / 40

/**
 * Detect if terminal supports 24-bit color.
 */
bool is_rgb_terminal(void) {

    if (IsWindows())
        return true;

    const char *colorterm = getenv("COLORTERM");
    if (colorterm && (!strcasecmp(colorterm, "truecolor") || //
                      !strcasecmp(colorterm, "24bit"))) {
        return true;
    }

    const char *term = getenv("TERM");
    if (term && (strstr(term, "24bit") || //
                 strstr(term, "truecolor") || //
                 strstr(term, "iterm") || //
                 strstr(term, "kitty") || //
                 strstr(term, "wezterm"))) {
        return true;
    }

    return false;
}

/**
 * Quantizes 24-bit RGB to xterm256 code range [16,256).
 */
int rgb2xterm256(int rgb) {
    int av, ir, ig, ib, il, qr, qg, qb, ql, r, g, b;
    unsigned char cube[] = {0, 0137, 0207, 0257, 0327, 0377};
    r = (rgb & 0xff0000) >> 020;
    g = (rgb & 0x00ff00) >> 010;
    b = (rgb & 0x0000ff) >> 000;
    av = r * .299 + g * .587 + b * .114 + .5;
    ql = (il = av > 238 ? 23 : (av - 3) / 10) * 10 + 8;
    qr = cube[(ir = UNCUBE(r))];
    qg = cube[(ig = UNCUBE(g))];
    qb = cube[(ib = UNCUBE(b))];
    if (SQR(qr - r) + SQR(qg - g) + SQR(qb - b) <= //
        SQR(ql - r) + SQR(ql - g) + SQR(ql - b))
        return ir * 36 + ig * 6 + ib + 020;
    return il + 0350;
}
