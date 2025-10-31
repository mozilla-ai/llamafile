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

#include "client.h"
#include "llama.cpp/llama.h"
#include "llamafile/flags.h"
#include "llamafile/llamafile.h"
#include "llamafile/server/cleanup.h"
#include "llamafile/server/log.h"
#include "llamafile/server/server.h"
#include "llamafile/server/time.h"
#include "llamafile/server/tokenbucket.h"
#include "llamafile/server/utils.h"
#include "llamafile/server/worker.h"
#include "llamafile/string.h"
#include "llamafile/threadlocal.h"
#include "llamafile/trust.h"
#include "llamafile/version.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define STANDARD_RESPONSE_HEADERS \
    "Server: llamafile/" LLAMAFILE_VERSION_STRING "\r\n" \
    "Referrer-Policy: origin\r\n" \
    "Cache-Control: private; max-age=0\r\n"

namespace lf {
namespace server {

static int64_t
atoi(std::string_view s)
{
    int64_t n = 0;
    for (char c : s) {
        if (c < '0' || c > '9')
            return -1;
        n *= 10;
        n += c - '0';
    }
    return n;
}

static void
on_http_cancel(Client* client)
{
    if (client->should_send_error_if_canceled_) {
        fcntl(client->fd_, F_SETFL, fcntl(client->fd_, F_GETFL) | O_NONBLOCK);
        client->send_error(503);
    }
}

static ThreadLocal<Client> g_http_cancel(on_http_cancel);

static const char*
pick_content_type(const std::string_view& path)
{
    const char* ct = FindContentType(path.data(), path.size());
    if (!ct)
        ct = "application/octet-stream";
    return ct;
}

Client::Client(llama_model* model)
  : model_(model)
  , cleanups_(nullptr)
  , ibuf_(FLAG_http_ibuf_size)
  , obuf_(FLAG_http_obuf_size)
{
    InitHttpMessage(&msg_, 0);
    url_.params.p = nullptr;
}

int
Client::close()
{
    int rc = 0;
    clear();
    DestroyHttpMessage(&msg_);
    if (fd_ != -1) {
        if (FLAG_verbose >= 2)
            SLOG("close");
        rc = ::close(fd_);
        fd_ = -1;
    }
    return rc;
}

void
Client::cleanup()
{
    Cleanup* clean;
    while ((clean = cleanups_)) {
        cleanups_ = clean->next;
        clean->func(clean->arg);
        delete clean;
    }
}

void
Client::clear()
{
    cleanup();
    free(url_memory_);
    url_memory_ = nullptr;
    free(params_memory_);
    params_memory_ = nullptr;
    free(url_.params.p);
    url_.params.p = nullptr;
    close_connection_ = false;
    payload_ = "";
    unread_ = 0;
}

void
Client::defer_cleanup(void (*func)(void*), void* arg)
{
    Cleanup* clean = new Cleanup;
    clean->next = cleanups_;
    clean->func = func;
    clean->arg = arg;
    cleanups_ = clean;
}

void
Client::run()
{
    ibuf_.n = 0;
    for (;;) {

        // read headers
        clear();
        if (!read_request())
            break;

        // process message
        if (!transport())
            break;

        // synchronize message stream
        if (close_connection_)
            break;
        if (!read_payload())
            break;

        // move pipelined bytes back to beginning
        if (ibuf_.n == ibuf_.i) {
            ibuf_.n = 0;
        } else {
            memmove(ibuf_.p, ibuf_.p + ibuf_.i, ibuf_.n - ibuf_.i);
            ibuf_.n -= ibuf_.i;
        }
    }
}

bool
Client::read_request()
{
    int inmsglen;
    ResetHttpMessage(&msg_, kHttpRequest);
    for (;;) {
        inmsglen = ParseHttpMessage(&msg_, ibuf_.p, ibuf_.n, ibuf_.c);
        if (inmsglen > 0) {
            message_started_ = timespec_real();
            ibuf_.i = inmsglen;
            return true;
        }
        if (inmsglen == -1) {
            SLOG("bad message %m");
            return false;
        }
        if (ibuf_.n)
            SLOG("fragmented message with %zu bytes", ibuf_.n);
        ssize_t got;
        got = read(fd_, ibuf_.p + ibuf_.n, ibuf_.c - ibuf_.n);
        if (!got && ibuf_.n)
            SLOG("unexpected eof after %zu bytes", ibuf_.n);
        if (got == -1 && (ibuf_.n || (errno != EAGAIN && errno != ECONNRESET)))
            SLOG("read failed %m");
        if (got <= 0)
            return false;
        ibuf_.n += got;
    }
}

bool
Client::transport()
{
    effective_ip_ = client_ip_;
    effective_ip_trusted_ = client_ip_trusted_;
    if (FLAG_ip_header) {
        if (is_loopback_ip(client_ip_) || client_ip_trusted_) {
            std::string_view ip_header = get_header(FLAG_ip_header);
            if (!ip_header.empty()) {
                long ip;
                if ((ip = parse_ip(ip_header)) == -1) {
                    effective_ip_ = ip;
                    effective_ip_trusted_ = is_trusted_ip(ip);
                } else {
                    SLOG("client's --ip-header wasn't a single ipv4 address");
                    effective_ip_trusted_ = false;
                }
            }
        } else {
            SLOG("received direct connect from untrusted ip");
            effective_ip_trusted_ = false;
        }
    }

    if (effective_ip_ != client_ip_) {
        char name[17];
        snprintf(name,
                 sizeof(name),
                 "%hhu.%hhu.%hhu.%hhu",
                 effective_ip_ >> 24,
                 effective_ip_ >> 16,
                 effective_ip_ >> 8,
                 effective_ip_);
        set_thread_name(name);
    }

    if (get_header("X-Priority") == "batch") {
        worker_->deprioritize();
    } else if (!effective_ip_trusted_) {
        if (tokenbucket_acquire(client_ip_) > FLAG_token_burst) {
            SLOG("deprioritizing");
            worker_->deprioritize();
        }
    }

    if (msg_.version > 11) {
        close_connection_ = true;
        return send_error(505);
    }

    if (msg_.method == kHttpConnect) {
        close_connection_ = true;
        return send_error(501);
    }

    if (!has_at_most_this_element(kHttpExpect, "100-continue")) {
        close_connection_ = true;
        return send_error(417);
    }

    if (HasHeader(kHttpTransferEncoding))
        if (!HeaderEqualCase(kHttpTransferEncoding, "identity")) {
            close_connection_ = true;
            return send_error(501, "Transfer-Encoding Not Implemented");
        }

    if (HasHeader(kHttpContentLength)) {
        long cl;
        cl = ParseContentLength(HeaderData(kHttpContentLength),
                                HeaderLength(kHttpContentLength));
        if (cl == -1) {
            close_connection_ = true;
            return send_error(400, "Bad Content-Length");
        }
        if (cl > ibuf_.c - ibuf_.i) {
            close_connection_ = true;
            return send_error(413);
        }
        unread_ = cl;
    } else if (msg_.method == kHttpPost || msg_.method == kHttpPut) {
        close_connection_ = true;
        return send_error(411);
    }

    if (FLAG_verbose >= 1)
        SLOG("get %#.*s", msg_.uri.b - msg_.uri.a, ibuf_.p + msg_.uri.a);

    if (msg_.version >= 11)
        if (HeaderEqualCase(kHttpExpect, "100-continue"))
            if (!send("HTTP/1.1 100 Continue\r\n\r\n"))
                return false;

    url_memory_ = ParseUrl(ibuf_.p + msg_.uri.a,
                           msg_.uri.b - msg_.uri.a,
                           &url_,
                           kUrlPlus | kUrlLatin1);
    if (!url_memory_)
        __builtin_trap();

    return dispatch();
}

// sends http error response message and body in one shot
//
// after this function is called, the handler must return control.
//
// @param code must be a number between 400 and 999
// @param reason must be a sanitized http token, or null for default
bool
Client::send_error(int code, const char* reason)
{
    if (!reason)
        reason = GetHttpReason(code);
    SLOG("error %d %s", code, reason);
    char* p = append_http_response_message(obuf_.p, code, reason);
    (void)!send_response(obuf_.p, p, std::string(reason) + "\r\n");
    return false;
}

// appends start of http response message to `p`
//
// after this function is called, more header lines may be appended.
// afterwards, either send_response() or send_response_start() should be
// called to transmit the message over the wire.
//
// @param p is a page guarded buffer
// @param code must be a number between 200 and 999
// @param reason must be a sanitized http token, or null for default
// @return p + len(appended content)
char*
Client::append_http_response_message(char* p, int code, const char* reason)
{
    // generate http message starting line
    *p++ = 'H';
    *p++ = 'T';
    *p++ = 'T';
    *p++ = 'P';
    *p++ = '/';
    *p++ = '1';
    *p++ = '.';
    *p++ = '0' + (msg_.version & 1);
    *p++ = ' ';
    *p++ = '0' + code / 100;
    *p++ = '0' + code / 10 % 10;
    *p++ = '0' + code % 10;
    *p++ = ' ';
    if (!reason)
        reason = GetHttpReason(code);
    p = stpcpy(p, reason);
    *p++ = '\r';
    *p++ = '\n';

    // append standard headers
    p = stpcpy(p, STANDARD_RESPONSE_HEADERS);

    // append date header
    tm tm;
    p = stpcpy(p, "Date: ");
    gmtime_lockless(message_started_.tv_sec, &tm);
    p = FormatHttpDateTime(p, &tm);
    *p++ = '\r';
    *p++ = '\n';

    // inform client of close() intent
    if (msg_.version < 11)
        close_connection_ = true;
    if (HeaderEqualCase(kHttpConnection, "close"))
        close_connection_ = true;
    if (close_connection_)
        p = stpcpy(p, "Connection: close\r\n");

    // send user supplied headers
    for (const auto& h : FLAG_headers) {
        p = (char*)mempcpy(p, h.data(), h.size());
        p = stpcpy(p, "\r\n");
    }

    return p;
}

// sends http response message and body in one shot
//
// after this function is called, the handler must return control.
//
// @param p0 points to start of http response message
// @param p points to end of http response message headers; we assume
//     that we can keep appending to `p` which must be page guarded
bool
Client::send_response(char* p0, char* p, std::string_view content)
{
    pthread_testcancel();
    should_send_error_if_canceled_ = false;

    // append content length
    p = stpcpy(p, "Content-Length: ");
    p = FormatInt64(p, content.size());
    *p++ = '\r';
    *p++ = '\n';

    // finish message
    *p++ = '\r';
    *p++ = '\n';

    return send2(std::string_view(p0, p - p0), content);
}

// sends http response message, but not its body.
//
// after this function is called, send_response_chunk() may be called to
// stream individual pieces of the response body. when you're done, you
// must call send_response_finish() to complete the response.
//
// here's an example handler that uses this:
//
//     bool
//     Client::fun()
//     {
//         char* message = obuf_.p;
//         char* p = append_http_response_message(message, 200);
//         if (!send_response_start(message, p))
//             return false;
//         sleep(1);
//         if (!send_response_chunk("hello\r\n"))
//             return false;
//         sleep(1);
//         if (!send_response_chunk("world\r\n"))
//             return false;
//         sleep(1);
//         return send_response_finish();
//     }
//
// @param p0 points to start of http response message
// @param p points to end of http response message headers; we assume
//     that we can keep appending to `p` which must be page guarded
bool
Client::send_response_start(char* p0, char* p)
{
    // use chunked transfer encoding if http/1.1
    if (msg_.version >= 11)
        p = stpcpy(p, "Transfer-Encoding: chunked\r\n");

    // finish message
    *p++ = '\r';
    *p++ = '\n';

    should_send_error_if_canceled_ = false;
    return send(std::string_view(p0, p - p0));
}

// finishes sending chunked http response body.
//
// once you are finished sending chunks, call send_response_finish().
bool
Client::send_response_chunk(const std::string_view content)
{
    // don't encode chunk boundaries for simple http client
    // it will need to rely on read() doing the right thing
    if (msg_.version < 11)
        return send(content);

    // sent in three pieces
    iovec iov[3];
    size_t bytes = 0;

    // 1. send "%zx\r\n" % (len(content))
    char start[32];
    char* p = start;
    p = FormatHex64(p, content.size(), 0);
    *p++ = '\r';
    *p++ = '\n';
    iov[0].iov_base = start;
    iov[0].iov_len = p - start;
    bytes += iov[0].iov_len;

    // 2. send content
    iov[1].iov_base = (void*)content.data();
    iov[1].iov_len = content.size();
    bytes += iov[1].iov_len;

    // 3. send newline
    iov[2].iov_base = (void*)"\r\n";
    iov[2].iov_len = 2;
    bytes += iov[2].iov_len;

    // perform send system call
    ssize_t sent;
    if ((sent = safe_writev(fd_, iov, 3)) != bytes) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("writev failed %m");
        close_connection_ = true;
        return false;
    }
    return true;
}

// finishes sending chunked http response body.
//
// after this function is called, the handler must return control.
bool
Client::send_response_finish()
{
    cleanup();

    // don't encode chunk boundaries for simple http client
    // it will need to rely on read() doing the right thing
    if (msg_.version < 11)
        return true;

    // send terminating chunk
    return send("0\r\n\r\n");
}

// writes any old data to socket
//
// unlike send() this won't fail if binary content is detected.
bool
Client::send_binary(const void* p, size_t n) {
    const char* buf = (const char*)p;
    size_t written = 0;
    while (written < n) {
        ssize_t sent = write(fd_, buf + written, n - written);
        if (sent == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // no data can be written right now; retry
                continue;
            }
            if (errno != ECONNRESET)
                SLOG("write failed %m");
            close_connection_ = true;
            return false;
        }
        // sent ≥ 0
        written += sent;
    }
    return true;
}

// writes non-binary data to socket
//
// consider using the higher level methods like send_error(),
// send_response(), send_response_start(), etc.
bool
Client::send(const std::string_view s)
{
    iovec iov[1];
    ssize_t sent;
    iov[0].iov_base = (void*)s.data();
    iov[0].iov_len = s.size();
    if ((sent = safe_writev(fd_, iov, 1)) != s.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("write failed %m");
        close_connection_ = true;
        return false;
    }
    return true;
}

// writes two pieces of non-binary data to socket in single system call
//
// consider using the higher level methods like send_error(),
// send_response(), send_response_start(), etc.
bool
Client::send2(const std::string_view s1, const std::string_view s2)
{
    iovec iov[2];
    ssize_t sent;
    iov[0].iov_base = (void*)s1.data();
    iov[0].iov_len = s1.size();
    iov[1].iov_base = (void*)s2.data();
    iov[1].iov_len = s2.size();
    if ((sent = safe_writev(fd_, iov, 2)) != s1.size() + s2.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            SLOG("writev failed %m");
        close_connection_ = true;
        return false;
    }
    return true;
}

bool
Client::has_at_most_this_element(int h, const std::string_view s)
{
    if (!HasHeader(h))
        return true;
    if (!SlicesEqualCase(s.data(), s.size(), HeaderData(h), HeaderLength(h)))
        return false;
    struct HttpHeader* x;
    for (unsigned i = 0; i < msg_.xheaders.n; ++i) {
        x = msg_.xheaders.p + i;
        if (GetHttpHeader(ibuf_.p + x->k.a, x->k.b - x->k.a) == h &&
            !SlicesEqualCase(
              ibuf_.p + x->v.a, x->v.b - x->v.a, s.data(), s.size())) {
            return false;
        }
    }
    return true;
}

std::string_view
Client::get_header(const std::string_view& key)
{
    int h;
    size_t i, keylen;
    if ((h = GetHttpHeader(key.data(), key.size())) != -1) {
        if (msg_.headers[h].a)
            return std::string_view(ibuf_.p + msg_.headers[h].a,
                                    msg_.headers[h].b - msg_.headers[h].a);
    } else {
        for (i = 0; i < msg_.xheaders.n; ++i)
            if (SlicesEqualCase(key.data(),
                                key.size(),
                                ibuf_.p + msg_.xheaders.p[i].k.a,
                                msg_.xheaders.p[i].k.b -
                                  msg_.xheaders.p[i].k.a))
                return std::string_view(ibuf_.p + msg_.xheaders.p[i].v.a,
                                        msg_.xheaders.p[i].v.b -
                                          msg_.xheaders.p[i].v.a);
    }
    return std::string_view();
}

bool
Client::read_payload()
{
    while (ibuf_.n - ibuf_.i < unread_) {
        ssize_t got;
        if ((got = read(fd_, ibuf_.p + ibuf_.n, ibuf_.c - ibuf_.n)) <= 0) {
            if (!got)
                SLOG("unexpected eof");
            if (got == -1)
                SLOG("read failed %m");
            return false;
        }
        ibuf_.n += got;
    }
    payload_ = std::string_view(ibuf_.p + ibuf_.i, unread_);
    ibuf_.i += unread_;
    unread_ = 0;
    if (msg_.method == kHttpPost && //
        HasHeader(kHttpContentType) &&
        IsMimeType(HeaderData(kHttpContentType),
                   HeaderLength(kHttpContentType),
                   "application/x-www-form-urlencoded")) {
        params_memory_ =
          ParseParams(payload_.data(), payload_.size(), &url_.params);
    }
    return true;
}

bool
Client::dispatch()
{
    bool res;
    should_send_error_if_canceled_ = true;
    g_http_cancel.set(this);
    res = dispatcher();
    g_http_cancel.set(nullptr);
    return res;
}

bool
Client::dispatcher()
{
    if (msg_.method == kHttpOptions) {
        char* p = obuf_.p;
        char* headers = p;
        p = append_http_response_message(p, 204);
        p = stpcpy(p, "Accept: */*\r\n");
        p = stpcpy(p, "Accept-Charset: utf-8\r\n");
        p = stpcpy(p, "Allow: GET, POST, OPTIONS\r\n");
        for (const auto& h : FLAG_headers) {
            p = (char*)mempcpy(p, h.data(), h.size());
            p = stpcpy(p, "\r\n");
        }
        return send_response(headers, p, "");
    }

    // get request-uri path
    char method[9] = { 0 };
    std::string_view p1 = path();
    WRITE64LE(method, msg_.method);
    SLOG("%s %.*s", method, (int)p1.size(), p1.data());
    if (!p1.starts_with(FLAG_url_prefix)) {
        SLOG("path prefix mismatch");
        return send_error(404);
    }
    p1 = p1.substr(strlen(FLAG_url_prefix));
    if (!p1.starts_with("/") || !IsAcceptablePath(p1.data(), p1.size())) {
        SLOG("unacceptable path");
        return send_error(400);
    }
    p1 = p1.substr(1);

    // look for dynamic endpoints
    if (p1 == "tokenize")
        return tokenize();
    if (p1 == "embedding")
        return embedding();
    if (p1 == "v1/embeddings")
        return embedding();
    if (p1 == "v1/completions")
        return v1_completions();
    if (p1 == "v1/chat/completions")
        return v1_chat_completions();
    if (p1 == "v1/models")
        return v1_models();
    if (p1 == "slotz")
        return slotz();
    if (p1 == "flagz")
        return flagz();

#if 0
    // TODO: implement frontend for database
    if (p1 == "db/chats" || p1 == "db/chats/")
        return db_chats();
    if (p1.starts_with("db/chat/")) {
        int64_t id = atoi(p1.substr(strlen("db/chat/")));
        if (id != -1)
            return db_chat(id);
    }
    if (p1.starts_with("db/messages/")) {
        int64_t id = atoi(p1.substr(strlen("db/messages/")));
        if (id != -1)
            return db_messages(id);
    }
    if (p1.starts_with("db/message/")) {
        int64_t id = atoi(p1.substr(strlen("db/message/")));
        if (id != -1)
            return db_messages(id);
    }
#endif

    // serve static endpoints
    int infd;
    size_t size;
    resolved_ = resolve(FLAG_www_root, p1);
    for (;;) {
        infd = open(resolved_.c_str(), O_RDONLY);
        if (infd == -1) {
            if (errno == ENOENT || errno == ENOTDIR) {
                SLOG("path not found: %s", resolved_.c_str());
                return send_error(404);
            } else if (errno == EPERM || errno == EACCES) {
                SLOG("path not authorized: %s", resolved_.c_str());
                return send_error(401);
            } else {
                SLOG("%s: %s", strerror(errno), resolved_.c_str());
                return send_error(500);
            }
        }
        struct stat st;
        if (fstat(infd, &st)) {
            SLOG("%s: %s", strerror(errno), resolved_.c_str());
            ::close(infd);
            return send_error(500);
        }
        size = st.st_size;
        if (S_ISREG(st.st_mode)) {
            break;
        } else if (S_ISDIR(st.st_mode)) {
            ::close(infd);
            resolved_ = resolve(resolved_, "index.html");
        } else {
            ::close(infd);
            SLOG("won't serve special file: %s", resolved_.c_str());
            return send_error(500);
        }
    }
    defer_cleanup(cleanup_fildes, (void*)(intptr_t)infd);
    char* p = append_http_response_message(obuf_.p, 200, "OK");
    p = stpcpy(p, "Content-Type: ");
    p = stpcpy(p, pick_content_type(resolved_));
    p = stpcpy(p, "\r\n");
    p = stpcpy(p, "Content-Length: ");
    p = FormatInt64(p, size);
    p = stpcpy(p, "\r\n");
    p = stpcpy(p, "\r\n");
    should_send_error_if_canceled_ = false;
    if (!send(std::string_view(obuf_.p, p - obuf_.p)))
        return false;
    char buf[16384];
    size_t i, chunk;
    for (i = 0; i < size; i += chunk) {
        chunk = size - i;
        if (chunk > sizeof(buf))
            chunk = sizeof(buf);
        ssize_t got = pread(infd, buf, chunk, i);
        if (got == -1) {
            SLOG("static asset pread failed: %s", strerror(errno));
            close_connection_ = true;
            return false;
        }
        if (got != chunk) {
            SLOG("couldn't read full amount reading static asset");
            close_connection_ = true;
            return false;
        }
        if (!send_binary(buf, chunk)) {
            close_connection_ = true;
            return false;
        }
    }
    if (FLAG_verbose >= 1)
        SLOG("served %s", resolved_.c_str());
    cleanup();
    return true;
}

std::string_view
Client::path()
{
    if (!url_.path.n)
        return "/";
    return { url_.path.p, url_.path.n };
}

std::optional<std::string_view>
Client::param(std::string_view key)
{
    for (size_t i = 0; i < url_.params.n; ++i)
        if (key.size() == url_.params.p[i].key.n)
            if (!memcmp(key.data(), url_.params.p[i].key.p, key.size()))
                return std::optional(std::string_view(url_.params.p[i].val.p,
                                                      url_.params.p[i].val.n));
    return {};
}

} // namespace server
} // namespace lf
