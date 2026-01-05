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

#include "highlight.h"
#include "util.h"

#include <cosmo.h>
#include <libc/intrin/bsr.h>
#include <libc/serialize.h>

// UTF-8 Thompson-Pike macros
#ifndef ThomPikeCont
#define ThomPikeCont(x)     (0200 == (0300 & (x)))
#define ThomPikeByte(x)     ((x) & (((1 << ThomPikeMsb(x)) - 1) | 3))
#define ThomPikeLen(x)      (7 - ThomPikeMsb(x))
#define ThomPikeMsb(x)      ((255 & (x)) < 252 ? bsr(255 & ~(x)) : 1)
#define ThomPikeMerge(x, y) ((x) << 6 | (077 & (y)))
#endif

// UTF-8 encode a codepoint into a packed uint64 (little-endian)
static inline uint64_t tpenc(uint32_t c) {
    if (c < 0x80) {
        return c;
    } else if (c < 0x800) {
        return (0xC0 | (c >> 6)) | ((0x80 | (c & 0x3F)) << 8);
    } else if (c < 0x10000) {
        return (0xE0 | (c >> 12)) | ((0x80 | ((c >> 6) & 0x3F)) << 8) |
               ((0x80 | (c & 0x3F)) << 16);
    } else {
        return (0xF0 | (c >> 18)) | ((0x80 | ((c >> 12) & 0x3F)) << 8) |
               ((0x80 | ((c >> 6) & 0x3F)) << 16) | ((0x80 | (c & 0x3F)) << 24);
    }
}

enum
{
    NORMAL,
    TICK,
    TICK_TICK,
    LANG,
    CODE,
    CODE_TICK,
    CODE_TICK_TICK,
    STAR,
    STRONG,
    STRONG_BACKSLASH,
    STRONG_STAR,
    BACKSLASH,
    INCODE,
    INCODE2,
    INCODE2_TICK,
    INCODE2_TICK2,
    EMPHASIS,
    EMPHASIS_BACKSLASH,
};

HighlightMarkdown::HighlightMarkdown()
{
}

HighlightMarkdown::~HighlightMarkdown()
{
}

void
HighlightMarkdown::feed(std::string* r, std::string_view input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        wchar_t c;
        int b = input[i] & 255;
        if (!u_) {
            if (b < 0300) {
                c = b;
            } else {
                c_ = ThomPikeByte(b);
                u_ = ThomPikeLen(b) - 1;
                continue;
            }
        } else if (ThomPikeCont(b)) {
            c = c_ = ThomPikeMerge(c_, b);
            if (--u_)
                continue;
        } else {
            u_ = 0;
            c = b;
        }
        if (c == 0xFEFF)
            continue; // utf-8 bom
        if (!c)
            c = 0xfffd; // replacement character
        switch (t_) {

        Normal:
        case NORMAL:
            if (c == '`') {
                t_ = TICK;
                break;
            } else if (c == '*') {
                t_ = STAR;
                break;
            } else if (c == '\\') {
                // handle \*\*not bold\*\* etc.
                t_ = BACKSLASH;
                *r += '\\';
                bol_ = false;
            } else {
                lf::append_wchar(r, c);
            }
            if (c == '\n') {
                bol_ = true;
                tail_ = false;
            } else {
                tail_ = true;
                if (!isblank(c))
                    bol_ = false;
            }
            break;

        case BACKSLASH:
            lf::append_wchar(r, c);
            t_ = NORMAL;
            break;

        case STAR:
            if (c == '*') {
                // handle **strong** text
                t_ = STRONG;
                *r += HI_BOLD;
                *r += "**";
            } else if (bol_ && isblank(c)) {
                *r += '*';
                lf::append_wchar(r, c);
                t_ = NORMAL;
            } else {
                // handle *emphasized* text
                // inverted because \e[3m has a poorly supported western
                // bias
                *r += '*';
                *r += HI_ITALIC;
                lf::append_wchar(r, c);
                t_ = EMPHASIS;
                if (c == '\\')
                    t_ = EMPHASIS_BACKSLASH;
            }
            bol_ = false;
            break;

        case EMPHASIS:
            // this is for *emphasized* text
            if (c == '*') {
                t_ = NORMAL;
                *r += HI_RESET;
                *r += '*';
            } else if (c == '\\') {
                t_ = EMPHASIS_BACKSLASH;
                *r += '\\';
            } else {
                lf::append_wchar(r, c);
            }
            break;

        case EMPHASIS_BACKSLASH:
            // so we can say *unbroken \* italic* and have it work
            lf::append_wchar(r, c);
            t_ = EMPHASIS;
            break;

        case STRONG:
            lf::append_wchar(r, c);
            if (c == '*') {
                t_ = STRONG_STAR;
            } else if (c == '\\') {
                t_ = STRONG_BACKSLASH;
            }
            break;

        case STRONG_BACKSLASH:
            // so we can say **unbroken \*\* bold** and have it work
            lf::append_wchar(r, c);
            t_ = STRONG;
            break;

        case STRONG_STAR:
            lf::append_wchar(r, c);
            if (c == '*' || // handle **bold** ending
                (c == '\n' && !tail_)) { // handle *** line break
                t_ = NORMAL;
                *r += HI_RESET;
            } else if (c == '\\') {
                t_ = STRONG_BACKSLASH;
            } else {
                t_ = STRONG;
            }
            break;

        case TICK:
            if (c == '`') {
                if (bol_) {
                    t_ = TICK_TICK;
                } else {
                    *r += HI_INCODE;
                    *r += "``";
                    t_ = INCODE2;
                }
            } else {
                *r += HI_INCODE;
                *r += '`';
                lf::append_wchar(r, c);
                t_ = INCODE;
            }
            bol_ = false;
            break;

        case INCODE:
            // this is for `inline code` like that
            // no backslash escapes are supported here
            lf::append_wchar(r, c);
            if (c == '`') {
                *r += HI_RESET;
                t_ = NORMAL;
            }
            break;

        case INCODE2:
            // this is for ``inline ` code`` like that
            // it lets you put backtick inside the code
            lf::append_wchar(r, c);
            if (c == '`') {
                t_ = INCODE2_TICK;
            }
            break;

        case INCODE2_TICK:
            lf::append_wchar(r, c);
            if (c == '`') {
                t_ = INCODE2_TICK2;
            } else {
                t_ = INCODE2;
            }
            break;

        case INCODE2_TICK2:
            if (c == '`') {
                *r += '`';
            } else {
                *r += HI_RESET;
                t_ = NORMAL;
                goto Normal;
            }
            break;

        case TICK_TICK:
            if (c == '`') {
                t_ = LANG;
                *r += "```";
            } else {
                *r += HI_INCODE;
                *r += "``";
                lf::append_wchar(r, c);
                t_ = INCODE2;
            }
            break;

        case LANG:
            if (!isascii(c) || !isspace(c)) {
                lf::append_wchar(r, c);
                lang_ += std::tolower(c);
            } else {
                if (!(highlighter_ = Highlight::create(lang_)))
                    highlighter_ = new HighlightTxt;
                lang_.clear();
                t_ = CODE;
                goto Code;
            }
            break;

        Code:
        case CODE:
            if (c == '`') {
                t_ = CODE_TICK;
            } else {
                char cs[8];
                uint64_t w = tpenc(c);
                WRITE64LE(cs, w);
                highlighter_->feed(r, cs);
            }
            break;

        case CODE_TICK:
            if (c == '`') {
                t_ = CODE_TICK_TICK;
            } else {
                char cs[9];
                uint64_t w = tpenc(c);
                cs[0] = '`';
                WRITE64LE(cs + 1, w);
                highlighter_->feed(r, cs);
                t_ = CODE;
            }
            break;

        case CODE_TICK_TICK:
            if (c == '`') {
                t_ = NORMAL;
                highlighter_->flush(r);
                delete highlighter_;
                highlighter_ = nullptr;
                *r += "```";
            } else {
                char cs[10];
                uint64_t w = tpenc(c);
                cs[0] = '`';
                cs[1] = '`';
                WRITE64LE(cs + 2, w);
                highlighter_->feed(r, cs);
                t_ = CODE;
            }
            break;

        default:
            __builtin_unreachable();
        }
    }
}

void
HighlightMarkdown::flush(std::string* r)
{
    switch (t_) {
    case LANG:
        lang_.clear();
        break;
    case STAR:
        *r += '*';
        break;
    case TICK:
        *r += '`';
        break;
    case TICK_TICK:
        *r += "``";
        break;
    case INCODE:
    case INCODE2:
    case INCODE2_TICK:
    case INCODE2_TICK2:
    case STRONG:
    case STRONG_BACKSLASH:
    case STRONG_STAR:
    case EMPHASIS:
    case EMPHASIS_BACKSLASH:
        *r += HI_RESET;
        break;
    case CODE:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        break;
    case CODE_TICK:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        *r += '`';
        break;
    case CODE_TICK_TICK:
        highlighter_->flush(r);
        delete highlighter_;
        highlighter_ = nullptr;
        *r += "``";
        break;
    default:
        break;
    }
    c_ = 0;
    u_ = 0;
    t_ = NORMAL;
    bol_ = true;
    tail_ = false;
}
