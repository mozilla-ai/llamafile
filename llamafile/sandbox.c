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
bool FLAG_confine_reads = false;

// Reports whether the host OS can enforce pledge() promises
// (Linux with SECCOMP_MODE_FILTER, or OpenBSD). Does not install anything.
bool llamafile_sandbox_supported(void) {
#ifdef COSMOCC
    return !pledge(0, 0);
#else
    return false;
#endif
}

// Shared policy gate: LLAMAFILE_SANDBOX_ACTIVE means "install it", otherwise a
// skip code. Both entry points consult this so the gate lives in one place.
static int sandbox_skip_status(void) {
    if (FLAG_unsecure)
        return LLAMAFILE_SANDBOX_UNSECURE;
    if (llamafile_has_gpu())
        return LLAMAFILE_SANDBOX_GPU;
    if (!llamafile_sandbox_supported())
        return LLAMAFILE_SANDBOX_UNSUPPORTED;
    return LLAMAFILE_SANDBOX_ACTIVE;
}

// Shared pledge() tail: EPERM penalty mode + install the filter.
static int sandbox_install_pledge(const char *promises) {
#ifdef COSMOCC
    __pledge_mode = PLEDGE_PENALTY_RETURN_EPERM;
    return pledge(promises, 0) ? LLAMAFILE_SANDBOX_FAILED
                               : LLAMAFILE_SANDBOX_ACTIVE;
#else
    (void)promises;
    return LLAMAFILE_SANDBOX_UNSUPPORTED;
#endif
}

// Installs the pledge() sandbox unconditionally, skipping the --unsecure/GPU
// policy gate (the gated entry points share sandbox_install_pledge() directly
// instead). Exposed for the unit test, which drives the promise sets in a
// forked child without the gate.
int llamafile_sandbox_apply(const char *promises) {
    if (!llamafile_sandbox_supported())
        return LLAMAFILE_SANDBOX_UNSUPPORTED;
    return sandbox_install_pledge(promises);
}

// Policy entry point for the CLI and chat modes: honors --unsecure and GPU
// mode, then pledges. These modes never confine paths -- they have no network
// at all and legitimately read user-chosen files (e.g. --image).
int llamafile_sandbox(const char *promises) {
    int skip = sandbox_skip_status();
    if (skip != LLAMAFILE_SANDBOX_ACTIVE)
        return skip;
    return sandbox_install_pledge(promises);
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

// Canary files the governability probe reads back after locking: if one is
// still readable, unveil() installed nothing (no Landlock, or an ungoverned
// filesystem) and we must not claim confinement. Requirements:
//   - world-readable, so the ONLY thing that can deny it is Landlock -- a
//     root-only file would be denied by ordinary permissions too, masking a
//     no-op unveil() and yielding a false "confined";
//   - outside every unveiled weights dir;
//   - benign and commonly read by ordinary software (distro/hostname
//     detection), so the probe isn't mistaken for credential recon the way
//     reading /etc/passwd would be.
// The probe uses the first one that exists; if none do, it conservatively
// reports "not confined" rather than guessing.
static const char *const kSandboxCanaries[] = {
    "/etc/os-release",
    "/etc/hostname",
};

// Derives the server pledge() promise string. Pure and side-effect-free so
// it can be unit-tested. rpath is unconditional: the model loader always
// performs a real open() -- of the on-disk weights, or of the executable
// itself to read its embedded /zip/ store (llamafile_open_zip) -- so a server
// can never load a model without it.
void llamafile_sandbox_server_promises(char *out, size_t len, bool is_openbsd,
                                       bool has_rw, bool needs_outbound) {
    // "anet" is accept()-only networking (no connect) -- the default, so a
    // compromised server can't dial out. When the server itself must make
    // outbound connections (--rpc, server-side tools, the MCP proxy) we relax
    // to "inet", which still blocks writes and exec. OpenBSD has no "anet", so
    // it always uses "inet".
    const char *net = (is_openbsd || needs_outbound) ? "inet" : "anet";
    if (has_rw)  // slot save/restore and the prompt cache read+write on disk
        snprintf(out, len, "stdio %s rpath wpath cpath", net);
    else
        snprintf(out, len, "stdio %s rpath", net);
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

// Writes into `dir` the path that must be reachable to open `path`: the
// path itself if it's a directory (a web root, a slot-save dir), else its
// parent directory, so multi-part GGUF shards beside a weights file come
// along -- except a file at the filesystem root, whose parent "/" would
// expose everything, so the file itself is used. Returns false for paths
// not on the real filesystem (empty, /zip/, or absent). Single source of
// truth for both unveil_container() (apply) and the governability probe.
static bool container_of(const char *path, char *dir, size_t len) {
    if (!path_on_disk(path))
        return false;
    if (isdirectory(path)) {
        strlcpy(dir, path, len);
        return true;
    }
    strlcpy(dir, path, len);
    char *slash = strrchr(dir, '/');
    if (!slash)
        strlcpy(dir, ".", len);        // bare "file" -> cwd
    else if (slash == dir)
        strlcpy(dir, path, len);       // "/file" -> the file, never "/"
    else
        *slash = '\0';                 // "dir/file" -> "dir"
    return true;
}

// unveil()s the container of `path` with the given perms, if it's on disk.
static void unveil_container(const char *path, const char *perms) {
    char dir[PATH_MAX];
    if (container_of(path, dir, sizeof(dir)))
        unveil(dir, perms);
}

// Installs the unveil() read/read-write rules (does NOT lock; caller locks
// with unveil(0,0)). The executable is always readable -- llamafile_open_zip
// reopens it for embedded weights and the bundled web UI -- as are the name
// resolution files a non-numeric --host needs. rw paths (which may not exist
// yet, e.g. a prompt cache created on first save) get their container dir.
static void unveil_apply(const char *const *read_paths, int n_read,
                         const char *const *rw_paths, int n_rw) {
    unveil(GetProgramExecutableName(), "r");
    if (path_on_disk("/etc/hosts"))
        unveil("/etc/hosts", "r");
    if (path_on_disk("/etc/resolv.conf"))
        unveil("/etc/resolv.conf", "r");
    for (int i = 0; i < n_read; ++i)
        unveil_container(read_paths[i], "r");
    for (int i = 0; i < n_rw; ++i)
        unveil_container(rw_paths[i], "rwc");
}

// True if `path` opens read-only under the (now locked) ruleset.
static bool can_read(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return false;
    close(fd);
    return true;
}

// True only if opening `path` is *actively denied* (EACCES/EPERM), not
// merely failing for some other reason such as ENOENT.
static bool is_denied(const char *path) {
    errno = 0;
    int fd = open(path, O_RDONLY);
    if (fd >= 0) {
        close(fd);
        return false;
    }
    return errno == EACCES || errno == EPERM;
}

// Probes whether unveil() confinement can be both enforced and survived on
// this filesystem, in a throwaway child so a "no" doesn't lock down the
// real process. Two failure modes to rule out:
//   - it installed nothing (no Landlock kernel, or cosmo's silent no-op):
//     the canary outside every rule stays readable -> not confined;
//   - it over-denies (virtiofs/9p/NFS accept rules then block everything):
//     a governed weights path can't be opened -> would break model load.
// Every path_on_disk()/container check runs before the lock, since access()
// is itself governed once the ruleset is locked.
#define SANDBOX_PROBE_MAX 64
static bool unveil_is_governable(const char *const *read_paths, int n_read,
                                 const char *const *rw_paths, int n_rw) {
    pid_t pid = fork();
    if (pid < 0)
        return false;
    if (!pid) {
        // Snapshot the paths the locked ruleset must still let us read,
        // computed while access() is unrestricted.
        char verify[SANDBOX_PROBE_MAX][PATH_MAX];
        int nv = 0;
        strlcpy(verify[nv++], GetProgramExecutableName(), PATH_MAX);
        for (int i = 0; i < n_read && nv < SANDBOX_PROBE_MAX; ++i)
            if (path_on_disk(read_paths[i]))
                strlcpy(verify[nv++], read_paths[i], PATH_MAX);
        for (int i = 0; i < n_rw && nv < SANDBOX_PROBE_MAX; ++i)
            if (container_of(rw_paths[i], verify[nv], PATH_MAX))
                ++nv;

        // Pick a canary that exists now, before the ruleset can hide it.
        const char *canary = 0;
        for (size_t i = 0; i < sizeof(kSandboxCanaries) / sizeof(*kSandboxCanaries); ++i)
            if (!access(kSandboxCanaries[i], F_OK)) {
                canary = kSandboxCanaries[i];
                break;
            }

        unveil_apply(read_paths, n_read, rw_paths, n_rw);
        unveil(0, 0);  // lock

        int ok = 1;
        for (int i = 0; ok && i < nv; ++i)
            ok = can_read(verify[i]);
        // Confinement must demonstrably bite: the canary, outside every
        // rule, must now be *denied* (EACCES/EPERM). No canary, or one that
        // fails for another reason, is inconclusive -- treat as "not
        // confined" rather than claiming a guarantee unveil() didn't install.
        if (ok && (!canary || !is_denied(canary)))
            ok = 0;
        _exit(ok ? 0 : 1);
    }
    int ws;
    while (waitpid(pid, &ws, 0) < 0 && errno == EINTR) {
    }
    return WIFEXITED(ws) && WEXITSTATUS(ws) == 0;
}

#endif  // COSMOCC

// Applies the server sandbox: pledge() always (accept()-only networking, no
// writes/exec unless configured), plus unveil() path confinement when
// spec->confine and the filesystem can enforce it. Fills promises_out with
// the pledge string for logging; returns an ACTIVE* status or a skip/FAILED
// code. The caller must quiesce background threads around this call (the
// filters are per-thread; see rule 1 above).
int llamafile_sandbox_server(const struct llamafile_sandbox_spec *spec,
                             char *promises_out, size_t promises_len) {
    if (promises_out && promises_len)
        promises_out[0] = '\0';
    int skip = sandbox_skip_status();
    if (skip != LLAMAFILE_SANDBOX_ACTIVE)
        return skip;
#ifdef COSMOCC
    // wpath/cpath only when a write target is actually configured; callers
    // pass fixed-position entries that are often empty strings.
    bool has_rw = false;
    for (int i = 0; i < spec->n_rw; ++i)
        if (spec->rw_paths[i] && *spec->rw_paths[i])
            has_rw = true;
    char promises[64];
    llamafile_sandbox_server_promises(promises, sizeof(promises), IsOpenbsd(),
                                      has_rw, spec->needs_outbound);
    if (promises_out)
        strlcpy(promises_out, promises, promises_len);

    // Opt-in unveil() path confinement, run before pledge() (it needs
    // Landlock syscalls the pledge filter denies). Skip it when the
    // filesystem can't enforce it, rather than locking down a server that can
    // no longer read its weights. 0 = not requested, 1 = confined, 2 = asked
    // for but unavailable.
    int confine_state = 0;
    if (spec->confine) {
        if (unveil_is_governable(spec->read_paths, spec->n_read,
                                 spec->rw_paths, spec->n_rw)) {
            unveil_apply(spec->read_paths, spec->n_read,
                         spec->rw_paths, spec->n_rw);
            unveil(0, 0);  // lock the ruleset
            confine_state = 1;
        } else {
            confine_state = 2;
        }
    }

    if (sandbox_install_pledge(promises) == LLAMAFILE_SANDBOX_FAILED)
        return LLAMAFILE_SANDBOX_FAILED;
    return confine_state == 1 ? LLAMAFILE_SANDBOX_ACTIVE_CONFINED
         : confine_state == 2 ? LLAMAFILE_SANDBOX_ACTIVE_UNCONFINED
                              : LLAMAFILE_SANDBOX_ACTIVE;
#else
    (void)spec;
    return LLAMAFILE_SANDBOX_UNSUPPORTED;
#endif
}

// True for the statuses where the pledge() filter is installed (whether or
// not path confinement also applied), as opposed to a skip or failure.
bool llamafile_sandbox_is_active(int status) {
    return status == LLAMAFILE_SANDBOX_ACTIVE ||
           status == LLAMAFILE_SANDBOX_ACTIVE_CONFINED ||
           status == LLAMAFILE_SANDBOX_ACTIVE_UNCONFINED;
}

const char *llamafile_sandbox_describe(int status) {
    switch (status) {
    case LLAMAFILE_SANDBOX_ACTIVE:
        return "active";
    case LLAMAFILE_SANDBOX_ACTIVE_CONFINED:
        return "active, reads confined to weights dirs";
    case LLAMAFILE_SANDBOX_ACTIVE_UNCONFINED:
        return "active (path confinement requested but unavailable on this filesystem)";
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
