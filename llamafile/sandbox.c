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
// llamafile sandboxing (issue #930)
//
// This module wraps Cosmopolitan Libc's pledge(), which implements
// OpenBSD-style syscall sandboxing. On OpenBSD it calls the native
// pledge(2). On Linux it installs a SECCOMP BPF filter that emulates
// the requested promise set; no kernel configuration or privileges
// are required. On every other platform (macOS, Windows, *BSD) it is
// a no-op: llamafile_sandbox() reports that and callers keep running
// unsandboxed, matching the pre-0.10 behavior.
//
// Promise cheat sheet (see Cosmopolitan libc/calls/pledge.c for the
// full syscall lists):
//
//   stdio    read/write on already-open fds, mmap w/o PROT_EXEC,
//            futexes, clocks, poll/epoll, and thread creation --
//            enough for ggml inference on memory-mapped weights
//   rpath    open files read-only (model weights on disk, /etc/hosts)
//   wpath    open existing files for writing
//   cpath    create/rename/unlink files
//   tty      terminal ioctls (raw mode, window size)
//   inet     IPv4/IPv6 sockets incl. connect()
//   anet     like inet but connect() is forbidden: the process can
//            accept() connections yet never initiate them (Linux only;
//            on OpenBSD the closest promise is inet)
//
// Two rules of engagement, learned from the 0.9.3 implementation:
//
//   1. SECCOMP filters attach to the calling thread and are inherited
//      only by threads created afterward. Callers must pledge before
//      spawning worker threads (e.g. before the HTTP listener starts),
//      not merely "after the socket is bound".
//
//   2. GPU support is loaded with cosmo_dlopen() and drivers keep
//      talking to /dev nodes at inference time, which no reasonable
//      promise set covers. When a GPU is in use the sandbox is skipped
//      and we say so, exactly like 0.9.3 did.
//
// Violations return EPERM instead of killing the process
// (PLEDGE_PENALTY_RETURN_EPERM), so a blocked syscall surfaces as an
// ordinary I/O error, again matching 0.9.3.
//

#define _COSMO_SOURCE  // exposes pledge() and __pledge_mode

#include "llamafile.h"

#include <errno.h>
#include <stdio.h>

#ifdef COSMOCC
#include <cosmo.h>
#include <libc/calls/calls.h>
#include <libc/calls/pledge.h>
#endif

bool FLAG_unsecure = false;

// Reports whether the host OS can enforce pledge() promises
// (Linux 2.6.23+ with SECCOMP, or OpenBSD). Does not install anything.
bool llamafile_sandbox_supported(void) {
#ifdef COSMOCC
    return !pledge(0, 0);
#else
    return false;
#endif
}

// Installs the pledge() sandbox on the calling thread, unconditionally.
// Threads created after this call inherit the restrictions. Returns
// LLAMAFILE_SANDBOX_ACTIVE on success, LLAMAFILE_SANDBOX_UNSUPPORTED if
// the OS can't enforce it, or LLAMAFILE_SANDBOX_FAILED with errno set.
int llamafile_sandbox_apply(const char *promises) {
#ifdef COSMOCC
    if (pledge(0, 0))
        return LLAMAFILE_SANDBOX_UNSUPPORTED;
    __pledge_mode = PLEDGE_PENALTY_RETURN_EPERM;
    if (pledge(promises, 0))
        return LLAMAFILE_SANDBOX_FAILED;
    return LLAMAFILE_SANDBOX_ACTIVE;
#else
    (void)promises;
    return LLAMAFILE_SANDBOX_UNSUPPORTED;
#endif
}

// Policy entry point used by llamafile's execution modes: honors
// --unsecure and skips sandboxing when a GPU backend is loaded (GPU
// drivers need dlopen/ioctl/device access that pledge would break).
int llamafile_sandbox(const char *promises) {
    if (FLAG_unsecure)
        return LLAMAFILE_SANDBOX_UNSECURE;
    if (llamafile_has_gpu())
        return LLAMAFILE_SANDBOX_GPU;
    return llamafile_sandbox_apply(promises);
}

const char *llamafile_sandbox_describe(int status) {
    switch (status) {
    case LLAMAFILE_SANDBOX_ACTIVE:
        return "active";
    case LLAMAFILE_SANDBOX_UNSECURE:
        return "disabled by --unsecure";
    case LLAMAFILE_SANDBOX_GPU:
        return "disabled in GPU mode";
    case LLAMAFILE_SANDBOX_UNSUPPORTED:
        return "not supported on this OS";
    case LLAMAFILE_SANDBOX_FAILED:
        return "failed";
    default:
        return "unknown";
    }
}
