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

#include "datauri.h"
#include "base64.hpp"
#include "string.h"
#include <cctype>

// See RFC2045 (MIME)
static const char kMimeToken[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, //  ! #$%&'  *+ -.
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, // 0123456789
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //  ABCDEFGHIJKLMNO
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, // PQRSTUVWXYZ   ^_
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // `abcdefghijklmno
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, // pqrstuvwxyz{|}~
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
};

// See RFC2397 ("data" URL scheme) which imports `urlchar` a.k.a. `uric`
// from RFC2396 (URI obsolete) with a design finalized by RFC3986 (URI).
static const char kUrlChar[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, //     $%&    +,-./
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, // 0123456789:; = ?
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // @ABCDEFGHIJKLMNO
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, // PQRSTUVWXYZ    _
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //  abcdefghijklmno
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, // pqrstuvwxyz   ~
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
};

alignas(signed char) static const signed char kHexToInt[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x00
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x10
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x20
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  -1, -1, -1, -1, -1, -1, // 0x30
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x40
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x50
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x60
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x70
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x80
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0x90
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xa0
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xb0
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xc0
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xd0
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xe0
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, // 0xf0
};

static std::string percent_decode(std::string_view data) {
    std::string r;
    enum {
        NORMAL,
        PERCENT1,
        PERCENT2,
    } t = NORMAL;
    int b, a = 0, ac = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int c = data[i] & 255;
        switch (t) {
        case NORMAL:
            if (c == '%') {
                t = PERCENT1;
            } else {
                r += c;
            }
            break;
        case PERCENT1:
            if ((a = kHexToInt[(ac = c)]) != -1) {
                t = PERCENT2;
            } else if (c == '%') {
                r += '%';
            } else {
                t = NORMAL;
                r += '%';
                r += c;
            }
            break;
        case PERCENT2:
            if ((b = kHexToInt[c]) != -1) {
                t = NORMAL;
                r += a << 4 | b;
            } else if (c == '%') {
                t = PERCENT1;
                r += '%';
                r += ac;
            } else {
                t = NORMAL;
                r += '%';
                r += ac;
                r += c;
            }
            break;
        default:
            __builtin_unreachable();
        }
    }
    switch (t) {
    case PERCENT1:
        r += '%';
        break;
    case PERCENT2:
        r += '%';
        r += ac;
        break;
    default:
        break;
    }
    return r;
}

DataUri::DataUri() {
}

DataUri::~DataUri() {
}

// parses "data" uri scheme, where `s` has everything after "data:".
// returns index of where data uri ends or npos if the parser failed
size_t DataUri::parse(std::string_view s) {
    enum {
        BEGIN,
        MIME_TYPE,
        MIME_SLASH,
        MIME_SUBTYPE,
        PARAMETER,
        PARAMETER_ATTRIBUTE,
        PARAMETER_VALUE,
        PAYLOAD,
    } t = BEGIN;
    size_t a = 0;
    std::string_view k;
    for (size_t i = 0; i < s.size(); ++i) {
        int c = s[i] & 255;
        switch (t) {
        case BEGIN:
            if (c == ';') {
                t = PARAMETER;
                a = i + 1;
                break;
            } else if (c == ',') {
                mime = "text/plain";
                params.emplace_back("charset", "US-ASCII");
                t = PAYLOAD;
                a = i + 1;
                break;
            } else if (kMimeToken[c]) {
                t = MIME_TYPE;
            } else {
                return std::string_view::npos;
            }
            break;
        case MIME_TYPE:
            if (c == '/') {
                t = MIME_SLASH;
            } else if (!kMimeToken[c]) {
                return std::string_view::npos;
            }
            break;
        case MIME_SLASH:
            if (kMimeToken[c]) {
                t = MIME_SUBTYPE;
            } else {
                return std::string_view::npos;
            }
            break;
        case MIME_SUBTYPE:
            if (c == ';') {
                mime = s.substr(a, i - a);
                t = PARAMETER;
                a = i + 1;
            } else if (c == ',') {
                mime = s.substr(a, i - a);
                t = PAYLOAD;
                a = i + 1;
            } else if (!kMimeToken[c]) {
                return std::string_view::npos;
            }
            break;
        case PARAMETER:
            if (kMimeToken[c]) {
                t = PARAMETER_ATTRIBUTE;
            } else {
                return std::string_view::npos;
            }
            break;
        case PARAMETER_ATTRIBUTE:
            if (c == ';') {
                params.emplace_back(s.substr(a, i - a), "");
                t = PARAMETER;
                a = i + 1;
            } else if (c == '=') {
                k = s.substr(a, i - a);
                t = PARAMETER_VALUE;
                a = i + 1;
            } else if (c == ',') {
                params.emplace_back(s.substr(a, i - a), "");
                t = PAYLOAD;
                a = i + 1;
            } else if (!kMimeToken[c]) {
                return std::string_view::npos;
            }
            break;
        case PARAMETER_VALUE:
            if (c == ';') {
                params.emplace_back(k, s.substr(a, i - a));
                t = PARAMETER;
                a = i + 1;
            } else if (c == ',') {
                params.emplace_back(k, s.substr(a, i - a));
                t = PAYLOAD;
                a = i + 1;
            } else if (!kMimeToken[c]) {
                return std::string_view::npos;
            }
            break;
        case PAYLOAD:
            if (!kUrlChar[c]) {
                data = s.substr(a, i - a);
                return i;
            }
            break;
        default:
            __builtin_unreachable();
        }
    }
    switch (t) {
    case PAYLOAD:
        data = s.substr(a);
        return s.size();
    default:
        return std::string_view::npos;
    }
}

std::string DataUri::decode() {
    if (has_param("base64"))
        return base64::decode(std::string(data));
    return percent_decode(data);
}

bool DataUri::has_param(std::string_view attribute) {
    for (const auto &param : params)
        if (!lf::strcasecmp(param.first, attribute))
            return true;
    return false;
}

std::string_view DataUri::get_param(std::string_view attribute) {
    for (const auto &param : params)
        if (!lf::strcasecmp(param.first, attribute))
            return param.second;
    return "";
}
