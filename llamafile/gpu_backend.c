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

#include "gpu_backend.h"
#include "llamafile.h"
#include <cosmo.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

// Register a backend with ggml (from ggml-backend.h).
extern void ggml_backend_register(ggml_backend_reg_t reg);

// =============================================================================
// ABI-correct call helpers
//
// On Windows the DSO exports functions with the ms_abi calling convention,
// while the cosmocc host uses System V. We keep the IsWindows() branch in one
// place so each call site (and each backend) does the right thing identically.
// =============================================================================

ggml_backend_reg_t gpu_call_reg(void *fp) {
    if (!fp)
        return NULL;
    if (IsWindows())
        return ((ggml_backend_reg_t(__attribute__((__ms_abi__)) *)(void))fp)();
    return ((ggml_backend_reg_t(*)(void))fp)();
}

int gpu_call_device_count(void *fp) {
    if (!fp)
        return 0;
    if (IsWindows())
        return ((int(__attribute__((__ms_abi__)) *)(void))fp)();
    return ((int(*)(void))fp)();
}

void gpu_call_get_description(void *fp, int device, char *buf, size_t n) {
    if (!fp)
        return;
    if (IsWindows())
        ((void(__attribute__((__ms_abi__)) *)(int, char *, size_t))fp)(device, buf, n);
    else
        ((void(*)(int, char *, size_t))fp)(device, buf, n);
}

void gpu_call_log_set(void *fp, llamafile_log_callback cb, void *user_data) {
    if (!fp)
        return;
    if (IsWindows())
        ((void(__attribute__((__ms_abi__)) *)(llamafile_log_callback, void *))fp)(cb, user_data);
    else
        ((void(*)(llamafile_log_callback, void *))fp)(cb, user_data);
}

// =============================================================================
// Load / unload
// =============================================================================

void gpu_backend_unlink(GpuBackend *b) {
    if (b->lib_handle) {
        cosmo_dlclose(b->lib_handle);
        b->lib_handle = NULL;
    }
    b->backend_init = NULL;
    b->backend_reg = NULL;
    b->get_device_count = NULL;
    b->get_device_description = NULL;
    b->log_set = NULL;
    b->lib_path[0] = '\0';
}

bool gpu_backend_link(GpuBackend *b, const char *dso, const GpuBackendDesc *desc) {
    b->desc = desc;

    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        char *err = cosmo_dlerror();
        llamafile_info(desc->tag, "failed to load library %s: %s", dso,
                       err ? err : "unknown error");
        return false;
    }

    // Required symbols: the backend is unusable without them. get_device_count
    // is required because gpu_backend_probe() relies on it to reject 0-device
    // DSOs (which is what lets AUTO mode fall through to the next backend).
    bool ok = true;
    b->backend_init = cosmo_dlsym(lib, desc->init);
    ok &= (b->backend_init != NULL);
    b->backend_reg = cosmo_dlsym(lib, desc->reg);
    ok &= (b->backend_reg != NULL);
    b->get_device_count = cosmo_dlsym(lib, desc->get_device_count);
    ok &= (b->get_device_count != NULL);

    // Optional symbols: degrade gracefully if absent.
    b->get_device_description =
        desc->get_device_description ? cosmo_dlsym(lib, desc->get_device_description) : NULL;
    b->log_set = cosmo_dlsym(lib, "ggml_log_set");

    if (!ok) {
        char *err = cosmo_dlerror();
        llamafile_info(desc->tag, "could not import all symbols from %s: %s", dso,
                       err ? err : "unknown error");
        cosmo_dlclose(lib);
        b->lib_handle = NULL;
        b->backend_init = NULL;
        b->backend_reg = NULL;
        b->get_device_count = NULL;
        b->get_device_description = NULL;
        b->log_set = NULL;
        return false;
    }

    b->lib_handle = lib;
    snprintf(b->lib_path, sizeof(b->lib_path), "%s", dso);
    return true;
}

// =============================================================================
// Crash guard
//
// Some backends' get_device_count triggers full driver/instance initialisation
// inside the DSO (e.g. ggml's ggml_vk_instance_init), which can throw a C++
// exception on a broken / unsupported driver. That exception does NOT unwind
// across the cosmo_dlopen/ms_abi boundary: even the DSO's own try/catch is
// bypassed and cosmo surfaces it as an uncaught SIGSEGV (issue #988). C++
// `catch` on our side can't help for the same reason. So we install a
// temporary signal guard around the foreign call and siglongjmp back on a
// fault, converting the crash into a clean "backend unavailable" result.
//
// This runs during one-time GPU init (under cosmo_once), before any worker
// threads exist, so a single static jmp_buf is safe.
// =============================================================================

static sigjmp_buf g_gpu_guard_jmp;
static volatile sig_atomic_t g_gpu_guard_active;

static void gpu_guard_handler(int sig) {
    if (g_gpu_guard_active) {
        g_gpu_guard_active = 0;
        siglongjmp(g_gpu_guard_jmp, sig);
    }
    // Fault outside a guarded call: restore default disposition and re-raise so
    // a genuine crash is not silently swallowed.
    signal(sig, SIG_DFL);
    raise(sig);
}

// Run fn(arg) with SIGSEGV/SIGABRT/SIGILL/SIGBUS/SIGFPE trapped. Returns true if
// it completed normally, false if it faulted (in which case the DSO is left for
// the caller to unlink and abandon).
static bool gpu_run_guarded(void (*fn)(void *), void *arg) {
    static const int kSignals[] = {SIGSEGV, SIGABRT, SIGILL, SIGBUS, SIGFPE};
    struct sigaction sa, old[sizeof(kSignals) / sizeof(kSignals[0])];
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = gpu_guard_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    for (int i = 0; i < (int)(sizeof(kSignals) / sizeof(kSignals[0])); ++i)
        sigaction(kSignals[i], &sa, &old[i]);

    bool ok;
    // SA_NODEFER (set above) leaves the trapped signal unblocked while the
    // handler runs, so a re-fault in the tiny window before siglongjmp is not
    // dropped. savesigs=1 makes siglongjmp restore the signal mask captured
    // here, keeping the mask correct after we jump out of the fault.
    if (sigsetjmp(g_gpu_guard_jmp, 1) == 0) {
        g_gpu_guard_active = 1;
        fn(arg);
        g_gpu_guard_active = 0;
        ok = true;
    } else {
        g_gpu_guard_active = 0;
        ok = false;
    }

    for (int i = 0; i < (int)(sizeof(kSignals) / sizeof(kSignals[0])); ++i)
        sigaction(kSignals[i], &old[i], NULL);
    return ok;
}

// =============================================================================
// Probe (device-count gate) and register
// =============================================================================

struct gpu_device_count_call {
    void *fp;
    int count;
};

static void gpu_device_count_thunk(void *p) {
    struct gpu_device_count_call *c = (struct gpu_device_count_call *)p;
    c->count = gpu_call_device_count(c->fp);
}

bool gpu_backend_probe(GpuBackend *b) {
    // Suppress the DSO's ggml logging before touching any function that
    // triggers device enumeration inside the DSO. Without this, a failed
    // probe on the wrong backend prints confusing errors even without
    // --verbose.
    if (!FLAG_verbose)
        gpu_call_log_set(b->log_set, llamafile_log_callback_null, NULL);

    // The DSO loads fine even when no compatible hardware is present, so probe
    // the device count before committing. The call goes through the crash guard
    // because for some backends it triggers driver init that can fault across
    // the DSO boundary (see the Crash guard section / issue #988). A fault is
    // treated exactly like "no usable device": unlink and let AUTO mode fall
    // through to the next backend and ultimately to CPU.
    struct gpu_device_count_call call = {b->get_device_count, 0};
    if (!gpu_run_guarded(gpu_device_count_thunk, &call)) {
        llamafile_info(b->desc->tag, "%s crashed during device probe; trying next backend",
                       b->desc->name);
        gpu_backend_unlink(b);
        return false;
    }

    if (call.count <= 0) {
        llamafile_info(b->desc->tag,
                       "%s library loaded but no devices detected; trying next backend",
                       b->desc->name);
        gpu_backend_unlink(b);
        return false;
    }
    return true;
}

void gpu_backend_register(GpuBackend *b) {
    // No crash guard here: register() runs only after gpu_backend_probe()
    // succeeded, so the DSO's driver/instance init already completed without
    // faulting and reg() just returns the (cached) registry.
    ggml_backend_reg_t reg = gpu_call_reg(b->backend_reg);
    if (reg) {
        ggml_backend_register(reg);
        llamafile_info(b->desc->tag, "%s backend registered with GGML", b->desc->name);
    }
}

// =============================================================================
// Out-of-process probe (Windows)
//
// On Windows the get_device_count() call can trigger GPU driver init that
// throws a C++/SEH exception across the cosmo_dlopen/ms_abi boundary. Catching
// that in-process and siglongjmp-ing out (gpu_run_guarded) is unsafe: it leaves
// the process corrupted and it dies silently later, during model load (#988
// follow-up). Instead we re-exec this binary as a tiny child that just dlopens
// the DSO and calls get_device_count(); a crash dies in the child and the
// parent reads the device count from the child's exit status. The parent never
// runs the faulting driver-init code for a device-less backend, so CPU fallback
// stays clean no matter how deep the fault is.
//
// Transport is the child's exit code:
//   0..253 -> device count (clamped)
//   254    -> child could not load/resolve the DSO
//   255    -> child caught a crash during the probe call
// A child killed by an uncaught signal is treated like 255.
// =============================================================================

#define GPU_PROBE_ENV_DSO "LLAMAFILE_GPU_PROBE_DSO"
#define GPU_PROBE_ENV_SYM "LLAMAFILE_GPU_PROBE_SYM"
#define GPU_PROBE_MAX_COUNT 253
#define GPU_PROBE_EXIT_LOAD_FAILED 254
#define GPU_PROBE_EXIT_CRASHED 255

static void gpu_probe_child_crash(int sig) {
    (void)sig;
    _Exit(GPU_PROBE_EXIT_CRASHED);
}

// Runs in the re-exec'd child before main(). When the probe env vars are set it
// dlopens the DSO, calls its device-count symbol, and _Exit()s with the count
// (or an error/crash code). In every normal invocation the env is unset and it
// returns immediately, so the parent and all non-probe binaries are unaffected.
__attribute__((constructor)) static void gpu_probe_child(void) {
    const char *dso = getenv(GPU_PROBE_ENV_DSO);
    const char *sym = getenv(GPU_PROBE_ENV_SYM);
    if (!dso || !sym || !*dso || !*sym)
        return;

    // Convert any crash during driver init into a clean exit code. We never
    // resume after the fault (unlike gpu_run_guarded), so there is no unwind and
    // nothing to corrupt -- this just replaces a backtrace / error dialog with a
    // quiet _Exit().
    static const int kSignals[] = {SIGSEGV, SIGABRT, SIGILL, SIGBUS, SIGFPE};
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = gpu_probe_child_crash;
    sigemptyset(&sa.sa_mask);
    for (int i = 0; i < (int)(sizeof(kSignals) / sizeof(kSignals[0])); ++i)
        sigaction(kSignals[i], &sa, NULL);

    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib)
        _Exit(GPU_PROBE_EXIT_LOAD_FAILED);
    void *fp = cosmo_dlsym(lib, sym);
    if (!fp)
        _Exit(GPU_PROBE_EXIT_LOAD_FAILED);

    int n = gpu_call_device_count(fp);
    if (n < 0)
        n = 0;
    if (n > GPU_PROBE_MAX_COUNT)
        n = GPU_PROBE_MAX_COUNT;
    _Exit(n);
}

bool gpu_backend_probe_oop(GpuBackend *b) {
    // Keep the DSO's own ggml logging suppressed for the later in-parent reg()
    // call, matching gpu_backend_probe()'s behaviour.
    if (!FLAG_verbose)
        gpu_call_log_set(b->log_set, llamafile_log_callback_null, NULL);

    const char *exe = GetProgramExecutableName();

    // Build a private envp (environ + the two probe vars) rather than mutating
    // the global environment, so concurrent cuda/vulkan probes can't race.
    size_t nenv = 0;
    while (environ[nenv])
        ++nenv;
    char **envp = (char **)malloc((nenv + 3) * sizeof(char *));
    char *dso_var = (char *)malloc(sizeof(GPU_PROBE_ENV_DSO) + 1 + strlen(b->lib_path) + 1);
    char *sym_var =
        (char *)malloc(sizeof(GPU_PROBE_ENV_SYM) + 1 + strlen(b->desc->get_device_count) + 1);
    if (!envp || !dso_var || !sym_var) {
        free(envp);
        free(dso_var);
        free(sym_var);
        llamafile_info(b->desc->tag, "%s probe: out of memory; trying next backend",
                       b->desc->name);
        gpu_backend_unlink(b);
        return false;
    }
    sprintf(dso_var, "%s=%s", GPU_PROBE_ENV_DSO, b->lib_path);
    sprintf(sym_var, "%s=%s", GPU_PROBE_ENV_SYM, b->desc->get_device_count);
    for (size_t i = 0; i < nenv; ++i)
        envp[i] = environ[i];
    envp[nenv + 0] = dso_var;
    envp[nenv + 1] = sym_var;
    envp[nenv + 2] = NULL;

    // Hide the child's stdout/stderr unless --verbose so a probe's own driver
    // chatter stays quiet (a crash is already silent thanks to the child's
    // _Exit()).
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_t *fap = NULL;
    if (!FLAG_verbose && posix_spawn_file_actions_init(&fa) == 0) {
        posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        fap = &fa;
    }

    char *const argv[] = {(char *)exe, NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, exe, fap, NULL, argv, envp);
    if (fap)
        posix_spawn_file_actions_destroy(fap);

    bool available = false;
    if (rc != 0) {
        llamafile_info(b->desc->tag, "%s probe: failed to spawn helper (%s); trying next backend",
                       b->desc->name, strerror(rc));
        gpu_backend_unlink(b);
    } else {
        int ws;
        while (waitpid(pid, &ws, 0) == -1 && errno == EINTR) {
        }
        int code = WIFEXITED(ws) ? WEXITSTATUS(ws) : -1;
        if (code >= 1 && code <= GPU_PROBE_MAX_COUNT) {
            // At least one usable device. The parent keeps its linked handle and
            // the caller will register the backend; re-running driver init in the
            // parent is safe because the child already proved it initialises.
            available = true;
        } else if (code == 0) {
            llamafile_info(b->desc->tag,
                           "%s library loaded but no devices detected; trying next backend",
                           b->desc->name);
            gpu_backend_unlink(b);
        } else if (code == GPU_PROBE_EXIT_LOAD_FAILED) {
            // The child couldn't dlopen the DSO or resolve get_device_count.
            // Unexpected (the parent already linked it), but report it for what
            // it is rather than as a crash.
            llamafile_info(b->desc->tag,
                           "%s probe: failed to load/resolve library; trying next backend",
                           b->desc->name);
            gpu_backend_unlink(b);
        } else {
            // 255 or killed by an uncaught signal: the probe call faulted.
            llamafile_info(b->desc->tag, "%s crashed during device probe; trying next backend",
                           b->desc->name);
            gpu_backend_unlink(b);
        }
    }

    free(envp);
    free(dso_var);
    free(sym_var);
    return available;
}
