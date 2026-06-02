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
#include <setjmp.h>
#include <signal.h>
#include <string.h>

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
