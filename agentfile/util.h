// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Small shared helpers for agentfile's recording callbacks: random hex
// identifiers and wall-clock timestamps in the encodings used by the pi
// session format (ISO 8601 + Unix ms) and OTLP (Unix nanoseconds).
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
