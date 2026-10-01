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
//
// Small shared helpers for agentfile's callbacks: terminal output
// (dim chrome, escaping model text), random hex identifiers and
// wall-clock timestamps in the encodings used by the pi session format
// (ISO 8601 + Unix ms) and OTLP (Unix nanoseconds).
//

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <unistd.h>

namespace agentfile {

// ANSI dim on/off codes for stderr chrome (progress lines, prompts), so it
// reads visually distinct from the model's answer on stdout. These wrap the
// text but never replace it: when stderr is redirected to a file/pipe or
// NO_COLOR is set, both return "" and the exact same text prints unstyled.
inline const char *dim() {
    static const bool on = isatty(STDERR_FILENO) && !std::getenv("NO_COLOR");
    return on ? "\033[2m" : "";
}
inline const char *dim_off() {
    static const bool on = isatty(STDERR_FILENO) && !std::getenv("NO_COLOR");
    return on ? "\033[0m" : "";
}

// Model or tool text headed for the terminal. Escapes what a terminal
// acts on or reorders instead of printing (C0 controls, DEL, C1 controls,
// bidi controls) and bytes that are not UTF-8, so the text cannot
// overprint, restyle or reorder what is shown around it. '\t' always
// passes; '\n' passes only for multi-line text.
inline std::string terminal_text(const std::string &s,
                                 bool keep_newlines = false) {
    std::string out;
    out.reserve(s.size());
    char buf[12];
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3
                 : (c >> 3) == 0x1e ? 4 : 0;
        uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1f : n == 3 ? c & 0x0f
                    : c & 0x07;
        bool ok = n > 0 && i + n <= s.size();
        for (size_t k = 1; ok && k < n; ++k) {
            unsigned char cc = s[i + k];
            ok = (cc & 0xc0) == 0x80;
            cp = (cp << 6) | (cc & 0x3f);
        }
        // Overlong forms, surrogates and code points past U+10FFFF are
        // not UTF-8 either.
        ok = ok && !(n == 2 && cp < 0x80) && !(n == 3 && cp < 0x800) &&
             !(n == 4 && (cp < 0x10000 || cp > 0x10ffff)) &&
             !(cp >= 0xd800 && cp <= 0xdfff);
        if (!ok) {
            std::snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
            ++i;
            continue;
        }
        bool control = (cp < 0x20 && cp != '\t' &&
                        !(keep_newlines && cp == '\n')) ||
                       (cp >= 0x7f && cp <= 0x9f) || cp == 0x061c ||
                       cp == 0x200e || cp == 0x200f ||
                       (cp >= 0x202a && cp <= 0x202e) ||
                       (cp >= 0x2066 && cp <= 0x2069);
        if (control) {
            std::snprintf(buf, sizeof(buf),
                          cp < 0x80 ? "\\x%02x" : "\\u%04x", (unsigned)cp);
            out += buf;
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    return out;
}

// n random lowercase hex characters (n/2 random bytes).
inline std::string random_hex(size_t n) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char digits[] = "0123456789abcdef";
    std::string out(n, '0');
    for (size_t i = 0; i < n; ++i)
        out[i] = digits[rng() & 0xf];
    return out;
}

// RFC 4122-shaped random UUID (version 4).
inline std::string random_uuid() {
    std::string h = random_hex(32);
    h[12] = '4';
    static const char variant[] = "89ab";
    h[16] = variant[h[16] & 0x3];
    return h.substr(0, 8) + "-" + h.substr(8, 4) + "-" + h.substr(12, 4) +
           "-" + h.substr(16, 4) + "-" + h.substr(20);
}

inline int64_t unix_ms_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

inline int64_t unix_ns_now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ISO 8601 UTC with millisecond precision, e.g. "2026-07-02T14:00:01.234Z".
inline std::string iso8601_now() {
    int64_t ms = unix_ms_now();
    time_t secs = (time_t)(ms / 1000);
    struct tm tm_utc;
#ifdef _WIN32
    gmtime_s(&tm_utc, &secs);
#else
    gmtime_r(&secs, &tm_utc);
#endif
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                  tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                  tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec,
                  (int)(ms % 1000));
    return buf;
}

} // namespace agentfile
