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
// This module wraps two Cosmopolitan Libc primitives:
//
//   pledge()  OpenBSD-style syscall sandboxing. On OpenBSD it calls the
//             native pledge(2); on Linux it installs a SECCOMP BPF filter
//             emulating the requested promise set. No kernel config or
//             privileges are needed.
//   unveil()  restricts which filesystem paths are reachable. On OpenBSD
//             it calls the native unveil(2); on Linux it uses the Landlock
//             LSM (kernel 5.13+). Together with pledge's "rpath" this turns
//             "read-only, anywhere" into "read-only, within these dirs".
//
// On every other platform (macOS, Windows, other BSDs) both are no-ops:
// the sandbox reports that and callers keep running unsandboxed, matching
// the pre-0.10 behavior.
//
// Promise cheat sheet (see Cosmopolitan libc/calls/pledge.c for the
// full syscall lists):
//
//   stdio    read/write on already-open fds, mmap w/o PROT_EXEC,
//            futexes, clocks, poll/epoll, and thread creation --
//            enough for ggml inference on memory-mapped weights
//   rpath    open files read-only (model weights, executable zip store)
//   wpath    open existing files for writing
//   cpath    create/rename/unlink files
//   tty      terminal ioctls (raw mode, window size)
//   inet     IPv4/IPv6 sockets incl. connect()
//   anet     like inet but connect() is forbidden: the process can
//            accept() connections yet never initiate them (Linux only;
//            on OpenBSD the closest promise is inet)
//
// Three rules of engagement:
//
//   1. SECCOMP/Landlock filters attach to the calling thread and are
//      inherited only by threads created *afterward* (cosmo does not use
//      SECCOMP_FILTER_FLAG_TSYNC). Callers must sandbox before spawning
//      worker threads, and must ensure no pre-existing background thread
//      (e.g. the llama.cpp log worker) survives across the call --- see
//      the common_log_pause()/resume() dance at the server call site.
//
//   2. GPU support is loaded with cosmo_dlopen() and drivers keep talking
//      to /dev nodes at inference time, which no reasonable promise set
//      covers. When a GPU is in use the sandbox is skipped and we say so.
//
//   3. unveil() must run *before* pledge(): it needs Landlock syscalls
//      that the pledge filter would otherwise deny.
//
// On Linux, violations return EPERM instead of killing the process
// (PLEDGE_PENALTY_RETURN_EPERM), so a blocked syscall surfaces as an
// ordinary I/O error. OpenBSD's native pledge(2) always kills with
// SIGABRT instead; that penalty mode is Linux-only.
//

#define _COSMO_SOURCE  // exposes pledge(), unveil(), __pledge_mode

#include "llamafile.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef COSMOCC
#include <cosmo.h>
#include <fcntl.h>
#include <libc/calls/calls.h>
#include <libc/calls/pledge.h>
#include <libc/dce.h>  // IsOpenbsd
#endif

bool FLAG_unsecure = false;

// Reports whether the host OS can enforce pledge() promises
// (Linux with SECCOMP_MODE_FILTER, or OpenBSD). Does not install anything.
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

// Policy entry point used by the CLI and chat modes: honors --unsecure
// and skips sandboxing when a GPU backend is loaded (GPU drivers need
// dlopen/ioctl/device access that pledge would break).
int llamafile_sandbox(const char *promises) {
    if (FLAG_unsecure)
        return LLAMAFILE_SANDBOX_UNSECURE;
    if (llamafile_has_gpu())
        return LLAMAFILE_SANDBOX_GPU;
    return llamafile_sandbox_apply(promises);
}

// Applies the CLI/chat sandbox and reports the outcome, collapsing the
// apply-perror-log boilerplate the two modes used to duplicate. On
// failure the caller decides whether to exit or return; every other
// status is informational and printed only when verbose.
int llamafile_sandbox_enter(const char *promises, bool verbose) {
    int status = llamafile_sandbox(promises);
    if (status == LLAMAFILE_SANDBOX_FAILED) {
        perror("pledge");
    } else if (verbose) {
        fprintf(stderr, "sandbox: %s\n", llamafile_sandbox_describe(status));
    }
    return status;
}

// Derives the server pledge() promise string. Pure and side-effect-free
// so it can be unit-tested. rpath is unconditional: the model loader
// always performs a real open() -- of the on-disk weights, or of the
// executable itself to read its embedded /zip/ store (llamafile_open_zip)
// -- so a server can never load a model without it. unveil() is what
// actually confines that rpath to the weights directories.
void llamafile_sandbox_server_promises(char *out, size_t len,
                                       bool is_openbsd, bool has_slot_save) {
    // anet = accept()-only networking (no connect); OpenBSD's native
    // pledge has no anet, so inet is the closest promise there.
    const char *base = is_openbsd ? "stdio inet rpath" : "stdio anet rpath";
    if (has_slot_save) {
        // slot save/restore reads and writes the state file on disk
        snprintf(out, len, "%s wpath cpath", base);
    } else {
        snprintf(out, len, "%s", base);
    }
}

#ifdef COSMOCC

// A path is "on disk" (needs its own unveil rule) when it exists on the
// real filesystem and isn't served from the executable's /zip/ store.
// Embedded weights -- a /zip/ path, or a bare basename that resolves to
// the executable via llamafile_open_zip -- are covered by unveiling the
// executable itself.
static bool path_on_disk(const char *path) {
    return path && *path && strncmp(path, "/zip/", 5) && !access(path, F_OK);
}

// unveil()s a path's directory read-only, so the whole weights directory
// (including multi-part GGUF shards) is reachable while the rest of the
// filesystem is not.
static void unveil_read_dir(const char *path) {
    if (!path_on_disk(path))
        return;
    char dir[PATH_MAX];
    strlcpy(dir, path, sizeof(dir));
    char *slash = strrchr(dir, '/');
    if (slash == dir) {
        slash[1] = '\0';  // path is "/file" -> unveil "/"
    } else if (slash) {
        *slash = '\0';    // "dir/file" -> unveil "dir"
    } else {
        strlcpy(dir, ".", sizeof(dir));  // bare "file" -> unveil cwd
    }
    unveil(dir, "r");
}

// Adds the read rules for the executable and every weights directory,
// plus read-write-create for the slot-save directory. Does NOT lock the
// ruleset (caller calls unveil(0,0)).
static void unveil_weights(const char *model_path, const char *mmproj_path,
                           const char *public_path,
                           const char *const *lora_paths, int n_loras,
                           const char *slot_save_path) {
    unveil(GetProgramExecutableName(), "r");
    unveil_read_dir(model_path);
    unveil_read_dir(mmproj_path);
    for (int i = 0; i < n_loras; ++i)
        unveil_read_dir(lora_paths[i]);
    if (path_on_disk(public_path))
        unveil(public_path, "r");
    if (slot_save_path && *slot_save_path)
        unveil(slot_save_path, "rwc");
}

// Landlock accepts unveil() rules on any filesystem but then denies all
// access on some (virtiofs, 9p, NFS, FUSE), which would break model
// loading. Probe in a throwaway child: apply the real rules, lock, and
// confirm the load-critical paths still open. A "no" here means we keep
// pledge but skip path confinement, rather than locking down a process
// that can no longer read its own weights.
static bool unveil_is_governable(const char *model_path,
                                 const char *mmproj_path,
                                 const char *public_path,
                                 const char *const *lora_paths, int n_loras,
                                 const char *slot_save_path) {
    pid_t pid = fork();
    if (pid < 0)
        return false;
    if (!pid) {
        unveil_weights(model_path, mmproj_path, public_path, lora_paths,
                       n_loras, slot_save_path);
        unveil(0, 0);
        int ok = 1;
        int fd = open(GetProgramExecutableName(), O_RDONLY);
        if (fd < 0)
            ok = 0;
        else
            close(fd);
        if (ok && path_on_disk(model_path)) {
            fd = open(model_path, O_RDONLY);
            if (fd < 0)
                ok = 0;
            else
                close(fd);
        }
        _exit(ok ? 0 : 1);
    }
    int ws;
    while (waitpid(pid, &ws, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(ws) && WEXITSTATUS(ws) == 0;
}

#endif  // COSMOCC

// Installs the server sandbox: confines filesystem reads to the executable
// and the weights directories via unveil(), then pledges accept()-only
// networking. Honors --unsecure and GPU mode. Fills promises_out with the
// pledge string for logging, and *confined_out with whether path
// confinement was applied. The caller is responsible for quiescing
// background threads (see rule 1 above) around this call.
int llamafile_sandbox_server(const char *model_path, const char *mmproj_path,
                             const char *public_path,
                             const char *const *lora_paths, int n_loras,
                             const char *slot_save_path, char *promises_out,
                             size_t promises_len, bool *confined_out) {
    if (promises_out && promises_len)
        promises_out[0] = '\0';
    if (confined_out)
        *confined_out = false;
    if (FLAG_unsecure)
        return LLAMAFILE_SANDBOX_UNSECURE;
    if (llamafile_has_gpu())
        return LLAMAFILE_SANDBOX_GPU;
#ifdef COSMOCC
    if (pledge(0, 0))
        return LLAMAFILE_SANDBOX_UNSUPPORTED;

    bool has_slot_save = slot_save_path && *slot_save_path;
    char promises[64];
    llamafile_sandbox_server_promises(promises, sizeof(promises), IsOpenbsd(),
                                      has_slot_save);
    if (promises_out)
        strlcpy(promises_out, promises, promises_len);

    // unveil() must run before pledge(): it needs Landlock syscalls the
    // pledge filter denies. Only apply it when the filesystem can actually
    // enforce it (probe first) -- otherwise skip confinement and keep
    // pledge, rather than making the server unable to read its weights.
    if (unveil_is_governable(model_path, mmproj_path, public_path, lora_paths,
                             n_loras, slot_save_path)) {
        unveil_weights(model_path, mmproj_path, public_path, lora_paths,
                       n_loras, slot_save_path);
        unveil(0, 0);  // lock the ruleset
        if (confined_out)
            *confined_out = true;
    }

    __pledge_mode = PLEDGE_PENALTY_RETURN_EPERM;
    if (pledge(promises, 0))
        return LLAMAFILE_SANDBOX_FAILED;
    return LLAMAFILE_SANDBOX_ACTIVE;
#else
    (void)model_path;
    (void)mmproj_path;
    (void)public_path;
    (void)lora_paths;
    (void)n_loras;
    return LLAMAFILE_SANDBOX_UNSUPPORTED;
#endif
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
