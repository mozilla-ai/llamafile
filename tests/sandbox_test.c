// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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
// Verifies pledge()/SECCOMP sandboxing (llamafile/sandbox.c, issue #930).
//
// Each case forks a child that installs one of the promise sets llamafile
// actually uses, then checks from inside the sandbox that:
//   - what must keep working still works (read-only opens under rpath,
//     accept() under anet, thread creation under stdio), and
//   - what must be blocked fails with EPERM instead of killing the
//     process (PLEDGE_PENALTY_RETURN_EPERM), exactly what the server
//     and CLI rely on.
//
// pledge() can only be enforced on Linux (SECCOMP) and OpenBSD. On other
// platforms this test reports SKIP and exits 0.
//

#include "llamafile/llamafile.h"

#include <arpa/inet.h>
#include <cosmo.h>
#include <errno.h>
#include <fcntl.h>
#include <libc/calls/calls.h>  // unveil
#include <libc/dce.h>          // IsLinux
#include <limits.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

// The policy wrapper llamafile_sandbox() consults llamafile_has_gpu(),
// which lives in llamafile.c and drags in the GPU loader stack. This
// test only exercises the enforcement primitives, so a stub keeps the
// link small.
bool llamafile_has_gpu(void) {
    return false;
}

#define CHECK(ok, what) \
    do { \
        if (!(ok)) { \
            fprintf(stderr, "FAIL(%s:%d): %s (errno=%s)\n", __FILE__, \
                    __LINE__, what, strerror(errno)); \
            _exit(1); \
        } \
    } while (0)

static void *thread_fn(void *arg) {
    return arg;
}

// what the --cli / --chat modes rely on
static void child_rpath_promises(void) {
    CHECK(llamafile_sandbox_apply("stdio rpath") == LLAMAFILE_SANDBOX_ACTIVE,
          "apply stdio rpath");

    // reads must keep working: model weights are opened after pledge
    int fd = open("/dev/null", O_RDONLY);
    CHECK(fd != -1, "open O_RDONLY allowed under rpath");
    close(fd);

    // threads must keep working: ggml spawns its compute pool after pledge
    pthread_t th;
    CHECK(!pthread_create(&th, 0, thread_fn, 0), "pthread_create under stdio");
    pthread_join(th, 0);

    // no filesystem writes
    errno = 0;
    CHECK(open("sandbox_test.tmp", O_WRONLY | O_CREAT | O_TRUNC, 0644) == -1,
          "open O_WRONLY|O_CREAT blocked");
    CHECK(errno == EPERM, "write open fails with EPERM, not a kill");

    // no network at all
    errno = 0;
    CHECK(socket(AF_INET, SOCK_STREAM, 0) == -1, "socket() blocked");
    CHECK(errno == EPERM, "socket() fails with EPERM");

    _exit(0);
}

// without rpath even read-only opens must be denied (embedded-weights
// server: filesystem access disabled entirely)
static void child_stdio_only(void) {
    CHECK(llamafile_sandbox_apply("stdio") == LLAMAFILE_SANDBOX_ACTIVE,
          "apply stdio");
    errno = 0;
    CHECK(open("/dev/null", O_RDONLY) == -1, "open O_RDONLY blocked");
    CHECK(errno == EPERM, "read open fails with EPERM");
    _exit(0);
}

// what --server relies on: accept incoming connections, never initiate
static void child_anet_promises(void) {
    // pre-pledge: a second socket to attempt the outbound connect() with
    int censored = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(censored != -1, "pre-pledge socket");

    CHECK(llamafile_sandbox_apply("stdio anet rpath") == LLAMAFILE_SANDBOX_ACTIVE,
          "apply stdio anet rpath");

    // the listening side of the HTTP server must keep working
    int server = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(server != -1, "socket() allowed under anet");

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; // ephemeral
    CHECK(!bind(server, (struct sockaddr *)&addr, sizeof(addr)),
          "bind() allowed under anet");
    CHECK(!listen(server, 1), "listen() allowed under anet");

    socklen_t addrlen = sizeof(addr);
    CHECK(!getsockname(server, (struct sockaddr *)&addr, &addrlen),
          "getsockname()");

    // accept() is permitted: with no pending client it must report
    // EAGAIN (not EPERM, which would mean the filter rejected it)
    int flags = fcntl(server, F_GETFL);
    CHECK(flags != -1 && fcntl(server, F_SETFL, flags | O_NONBLOCK) != -1,
          "set O_NONBLOCK");
    errno = 0;
    CHECK(accept(server, 0, 0) == -1, "accept with no client");
    CHECK(errno == EAGAIN || errno == EWOULDBLOCK,
          "accept() allowed by filter (EAGAIN, not EPERM)");

    // outbound connections must be denied, even to ourselves
    errno = 0;
    CHECK(connect(censored, (struct sockaddr *)&addr, sizeof(addr)) == -1,
          "connect() blocked under anet");
    CHECK(errno == EPERM, "connect() fails with EPERM");

    _exit(0);
}

// unveil() confinement: reads under an unveiled directory succeed while
// the rest of the filesystem is denied. Landlock accepts rules on any
// filesystem but enforces on only some (see llamafile_sandbox_server's
// governability probe), so if a read inside the unveiled dir is itself
// denied we treat the filesystem as ungovernable and skip -- matching the
// production fallback -- rather than failing.
static void child_unveil(void) {
    // Two files we own: one inside a directory we will unveil (must stay
    // readable), one outside it (a self-made canary that must become denied
    // -- a sibling under /tmp, never a system file).
    char dir[] = "/tmp/lf_uv_XXXXXX";
    CHECK(mkdtemp(dir), "mkdtemp");
    char inside[PATH_MAX];
    snprintf(inside, sizeof(inside), "%s/weights.bin", dir);
    int fd = open(inside, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd != -1, "create file inside temp dir");
    close(fd);

    char canary[] = "/tmp/lf_canary_XXXXXX";
    fd = mkstemp(canary);  // outside `dir`, so confinement must hide it
    CHECK(fd != -1, "mkstemp canary");
    close(fd);

    unveil(dir, "r");
    unveil(0, 0);  // lock

    int in = open(inside, O_RDONLY);
    if (in == -1) {
        // filesystem accepts unveil() but denies everything: over-denies
        fprintf(stderr, "sandbox_test: unveil: SKIP (filesystem not "
                        "Landlock-governable)\n");
        _exit(0);
    }
    close(in);

    // If the canary outside the unveiled dir is still readable, unveil()
    // installed nothing (kernel without Landlock, or cosmo's silent no-op):
    // there is no confinement to assert. Same check the production probe
    // makes, on a file we created rather than a system file.
    errno = 0;
    int c = open(canary, O_RDONLY);
    if (c != -1) {
        close(c);
        fprintf(stderr, "sandbox_test: unveil: SKIP (Landlock unavailable; "
                        "confinement not installed)\n");
        _exit(0);
    }
    CHECK(errno == EACCES || errno == EPERM,
          "unveil denial is EACCES/EPERM, not a kill");
    _exit(0);
}

static int run_case(const char *name, void (*fn)(void)) {
    pid_t pid = fork();
    if (!pid)
        fn(); // never returns
    int ws;
    if (waitpid(pid, &ws, 0) != pid || !WIFEXITED(ws) || WEXITSTATUS(ws)) {
        fprintf(stderr, "sandbox_test: %s: FAILED%s\n", name,
                WIFSIGNALED(ws) ? " (killed by signal)" : "");
        return 1;
    }
    printf("sandbox_test: %s: OK\n", name);
    return 0;
}

// Pure promise-string derivation: keep the server policy unit-testable
// rather than buried in the vendored server.cpp patch. Covers the
// accept-only default, the wpath/cpath (has_rw) and inet (needs_outbound)
// relaxations, and OpenBSD's lack of "anet".
static int test_promises(void) {
    struct {
        bool openbsd, has_rw, outbound;
        const char *want;
    } cases[] = {
        {false, false, false, "stdio anet rpath"},
        {false, true, false, "stdio anet rpath wpath cpath"},
        {false, false, true, "stdio inet rpath"},              // --rpc etc.
        {false, true, true, "stdio inet rpath wpath cpath"},
        {true, false, false, "stdio inet rpath"},              // OpenBSD: no anet
        {true, true, false, "stdio inet rpath wpath cpath"},
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i) {
        char got[64];
        llamafile_sandbox_server_promises(got, sizeof(got), cases[i].openbsd,
                                          cases[i].has_rw, cases[i].outbound);
        if (strcmp(got, cases[i].want)) {
            fprintf(stderr, "sandbox_test: promises: FAILED want=\"%s\" got=\"%s\"\n",
                    cases[i].want, got);
            rc = 1;
        }
    }
    if (!rc)
        printf("sandbox_test: server promise derivation: OK\n");
    return rc;
}

int main(void) {
    int rc = 0;

    // Pure logic, runs on every platform.
    rc |= test_promises();

    // The enforcement cases assert Linux SECCOMP semantics: the "anet"
    // promise is a cosmo Linux extension OpenBSD's pledge(2) rejects, and
    // the EPERM assertions assume PLEDGE_PENALTY_RETURN_EPERM (Linux only;
    // OpenBSD kills violators). So gate them to Linux.
    if (!IsLinux() || !llamafile_sandbox_supported()) {
        printf("sandbox_test: SKIP enforcement: needs Linux with SECCOMP "
               "(anet/EPERM semantics are Linux-specific)\n");
        return rc;
    }
    rc |= run_case("stdio+rpath (cli promises)", child_rpath_promises);
    rc |= run_case("stdio only (embedded weights)", child_stdio_only);
    rc |= run_case("stdio+anet+rpath (server promises)", child_anet_promises);
    rc |= run_case("unveil confinement", child_unveil);
    return rc;
}
