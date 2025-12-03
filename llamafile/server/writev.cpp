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

#include "llamafile/server/log.h"
#include "llamafile/llamafile.h"
#include "utils.h"
#include <cerrno>
#include <poll.h>
#include <string_view>
#include <vector>

namespace lf {
namespace server {

ssize_t
safe_writev(int fd, const iovec* iov, int iovcnt)
{
    // Security check for binary content in headers
    for (int i = 0; i < iovcnt; ++i) {
        bool has_binary = false;
        size_t n = iov[i].iov_len;
        unsigned char* p = (unsigned char*)iov[i].iov_base;
        for (size_t j = 0; j < n; ++j) {
            has_binary |= p[j] < 7;
        }
        if (has_binary) {
            SLOG("safe_writev() detected binary server is compromised");
            errno = EINVAL;
            return -1;
        }
    }

    ssize_t total = 0;
    // Create a mutable copy of iovecs to track progress
    std::vector<iovec> copy(iov, iov + iovcnt);
    int i = 0; // Current iovec index

    while (i < iovcnt) {
        ssize_t sent = writev(fd, copy.data() + i, iovcnt - i);
        if (sent == -1) {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = { .fd = fd, .events = POLLOUT };
                int rc = poll(&pfd, 1, FLAG_http_write_timeout);
                if (rc == 0) {
                    errno = ETIMEDOUT;
                    return -1;
                }
                if (rc == -1) {
                    if (errno == EINTR)
                        continue;
                    return -1;
                }
                continue;
            }
            return -1;
        }

        total += sent;
        size_t got = sent;

        // Advance the iovecs based on bytes written
        while (got > 0 && i < iovcnt) {
            if (got >= copy[i].iov_len) {
                got -= copy[i].iov_len;
                ++i;
            } else {
                copy[i].iov_base = (char*)copy[i].iov_base + got;
                copy[i].iov_len -= got;
                got = 0;
            }
        }
    }
    return total;
}

} // namespace server
} // namespace lf
