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

#ifndef LLAMAFILE_GPU_BACKEND_H_
#define LLAMAFILE_GPU_BACKEND_H_
#include <limits.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared probing logic for dynamically-loaded GPU backends that export the
// ggml C ABI and, on Windows, use the ms_abi calling convention.
//
// Used by the prebuilt-DSO backends: CUDA, ROCm (cuda.c) and Vulkan
// (vulkan.c). Metal is intentionally NOT built on top of this: it is
// compiled at runtime, is macOS/AArch64-only (no ms_abi/sysv split) and has
// no notion of selecting among multiple devices, so it does not share this
// pattern.
//
// The point of centralising this is consistency: every backend here goes
// through the same load -> suppress-logs -> device-count gate -> register
// path, so a fix or a behavioural change applies to all of them at once.
// Vulkan was missing the device-count gate, which is what made a 0-device /
// failed Vulkan init block fallback to CPU instead of being skipped (#988).

typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;

// Must match ggml_log_callback from ggml.h.
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// Static identity of a backend: its log/display names plus the C symbols its
// DSO exports. The tag and name live here (not at each call site) so the
// shared probe/register routines read them straight off the backend. CUDA and
// ROCm share symbol strings but are two distinct descriptors because their
// display name differs. The log-control symbol is always "ggml_log_set" and is
// resolved unconditionally.
typedef struct GpuBackendDesc {
    const char *tag;                    // lowercase log channel: "cuda" / "vulkan"
    const char *name;                   // display word: "CUDA" / "ROCm" / "Vulkan"
    const char *init;                   // e.g. "ggml_backend_cuda_init" (required)
    const char *reg;                    // e.g. "ggml_backend_cuda_reg" (required)
    const char *get_device_count;       // required: the device-count gate needs it
    const char *get_device_description; // optional
} GpuBackendDesc;

// Runtime state for one loaded backend DSO.
//
// `desc` is the backend's identity, set by gpu_backend_link(); the shared
// routines read tag/name from it. Symbols are stored as raw pointers and
// invoked through the gpu_call_* helpers, which apply the correct calling
// convention (ms_abi on Windows, sysv elsewhere). backend_init has a
// backend-specific signature (CUDA takes int, Vulkan takes size_t), so it is
// stored here but cast and called by each backend's own typed wrapper rather
// than through a shared helper.
typedef struct GpuBackend {
    const GpuBackendDesc *desc;
    bool supported;
    atomic_uint once;
    void *lib_handle;
    void *backend_init;
    void *backend_reg;
    void *get_device_count;
    void *get_device_description;
    void *log_set;
    // Resolved on-disk path of the loaded DSO, recorded by gpu_backend_link().
    // The out-of-process probe needs it to tell the child what to dlopen.
    char lib_path[PATH_MAX];
} GpuBackend;

// dlopen `dso` and resolve `desc`'s symbols into `b`, recording `desc` as the
// backend's identity. On any failure (load error or a missing required symbol)
// it logs on desc->tag, frees what it took, and returns false. Designed to be
// wrapped by a tiny per-backend llamafile_link_dso_fn thunk so it can be passed
// to llamafile_try_load_prebuilt_dso().
bool gpu_backend_link(GpuBackend *b, const char *dso, const GpuBackendDesc *desc);

// dlclose the DSO (if any) and zero every pointer in `b` (keeping `desc`).
void gpu_backend_unlink(GpuBackend *b);

// Run immediately after a successful link, before registering: suppress the
// DSO's own ggml logging unless --verbose, then require get_device_count() > 0.
// If the backend reports no usable devices it logs, on desc->tag,
//   "<name> library loaded but no devices detected; trying next backend"
// calls gpu_backend_unlink() and returns false, so an AUTO-mode caller falls
// through to the next backend and ultimately to CPU. Returns true if the
// backend has at least one device and should be registered.
bool gpu_backend_probe(GpuBackend *b);

// Same contract as gpu_backend_probe(), but runs the device-count call in a
// short-lived child process (a re-exec of this binary) instead of under an
// in-process signal guard. Used on Windows, where the foreign driver-init code
// behind get_device_count() can throw a C++/SEH exception across the
// cosmo_dlopen/ms_abi boundary; catching it in-process and siglongjmp-ing out
// leaves the process corrupted (silent exit later, issue #988 follow-up). A
// crash here dies in the child, and the parent never executes the faulting
// driver-init code for a device-less backend, so CPU fallback stays clean
// regardless of how deep the fault is. Requires gpu_backend_link() to have run
// (it needs b->lib_path).
bool gpu_backend_probe_oop(GpuBackend *b);

// Call backend_reg() and, if it returns a registry, hand it to
// ggml_backend_register(), logging "<name> backend registered with GGML". Safe
// to call only after gpu_backend_probe() returned true.
void gpu_backend_register(GpuBackend *b);

// ABI-correct invocation helpers (ms_abi on Windows, sysv elsewhere). Each
// tolerates a NULL pointer the way the old per-backend wrappers did.
ggml_backend_reg_t gpu_call_reg(void *fp);
int gpu_call_device_count(void *fp);
void gpu_call_get_description(void *fp, int device, char *buf, size_t n);
void gpu_call_log_set(void *fp, llamafile_log_callback cb, void *user_data);

#ifdef __cplusplus
}
#endif
#endif // LLAMAFILE_GPU_BACKEND_H_
