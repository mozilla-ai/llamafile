// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

//
// Runtime Vulkan GPU support for llamafile
//
// This file implements dynamic loading of Vulkan GPU support.
// At runtime on Linux/Windows/macOS with Vulkan-compatible GPU:
//   1. Try to load pre-built DSO from /zip/ggml-vulkan.so (bundled)
//   2. Or try to load from ~/.llamafile/ (pre-compiled)
//   3. Load the DSO with cosmo_dlopen() and register the Vulkan backend
//
// The load/probe/register mechanics live in gpu_backend.c, shared with the
// CUDA/ROCm backends so all three behave identically.
//

#include "gpu_backend.h"
#include "llamafile.h"
#include <cosmo.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Vulkan backend state and identity (log/display names + exported symbols).
static GpuBackend g_vulkan;

static const GpuBackendDesc VULKAN_DESC = {
    .tag = "vulkan",
    .name = "Vulkan",
    .init = "ggml_backend_vk_init",
    .reg = "ggml_backend_vk_reg",
    .get_device_count = "ggml_backend_vk_get_device_count",
    .get_device_description = "ggml_backend_vk_get_device_description",
};

// Thin llamafile_link_dso_fn thunk over the shared linker.
static bool LinkVulkan(const char *dso) {
    return gpu_backend_link(&g_vulkan, dso, &VULKAN_DESC);
}

static bool ImportVulkanImpl(void) {
    // Note: Unlike CUDA, we don't skip Apple Silicon here because
    // Vulkan works on macOS via MoltenVK (Vulkan-to-Metal translation)

    // Check if we're allowed to even try
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_VULKAN:
        break;
    default:
        return false;
    }

    // Determine DSO name
    const char *ext = llamafile_get_dso_extension();
    char vulkan_dso[64];
    snprintf(vulkan_dso, sizeof(vulkan_dso), "ggml-vulkan.%s", ext);

    // Try to load pre-built DSO
    if (!llamafile_try_load_prebuilt_dso(vulkan_dso, "vulkan", LinkVulkan)) {
        // No pre-built DSO found
        llamafile_info("vulkan", "no pre-built GPU library found");
        llamafile_info("vulkan", "to enable Vulkan support, build with:");
        llamafile_info("vulkan", "  llamafile/vulkan.sh");
        return false;
    }

    // Gate on device count before registering (issue #988). The DSO loads even
    // when Vulkan has no usable device; without this check we'd register a
    // 0-device backend and block fallback to CPU. gpu_backend_probe() unlinks
    // and returns false in that case so AUTO mode moves on. On Windows a
    // faulting Vulkan driver init behind get_device_count() can corrupt the
    // process if caught in-process, so probe out-of-process there (#988).
    if (!(IsWindows() ? gpu_backend_probe_oop(&g_vulkan) : gpu_backend_probe(&g_vulkan)))
        return false;

    gpu_backend_register(&g_vulkan);
    return true;
}

static void ImportVulkan(void) {
    if (ImportVulkanImpl()) {
        g_vulkan.supported = true;
        llamafile_info("vulkan", "Vulkan GPU support successfully loaded");
        int count = gpu_call_device_count(g_vulkan.get_device_count);
        llamafile_info("vulkan", "found %d GPU device(s)", count);
    } else if (FLAG_gpu == LLAMAFILE_GPU_VULKAN) {
        // Vulkan was explicitly requested but the probe found no usable device
        // (or the driver faulted during probe and was skipped to keep the
        // process alive -- see gpu_backend.c). Fail loudly rather than
        // silently downgrade, but tell the user how to proceed.
        fprintf(stderr,
                "fatal error: --gpu vulkan was explicitly requested but Vulkan is not "
                "usable on this system\n"
                "  - no Vulkan-capable device was detected, or the driver "
                "failed/crashed during probe and was skipped for safety\n"
                "  - retry with --gpu auto to use the best available backend "
                "(falls back to CPU), or --gpu disabled to force CPU\n");
        exit(1);
    }
}

bool llamafile_has_vulkan(void) {
    cosmo_once(&g_vulkan.once, ImportVulkan);
    return g_vulkan.supported;
}

// Wrapper functions for dynamically loaded Vulkan backend

ggml_backend_t ggml_backend_vk_init(size_t device) {
    if (!llamafile_has_vulkan())
        return NULL;
    void *fp = g_vulkan.backend_init;
    if (!fp)
        return NULL;
    // backend_init has a Vulkan-specific signature (size_t device), so it is
    // called here rather than through a shared gpu_call_* helper.
    if (IsWindows())
        return ((ggml_backend_t(__attribute__((__ms_abi__)) *)(size_t))fp)(device);
    return ((ggml_backend_t(*)(size_t))fp)(device);
}

int ggml_backend_vk_get_device_count(void) {
    if (!llamafile_has_vulkan())
        return 0;
    return gpu_call_device_count(g_vulkan.get_device_count);
}

void ggml_backend_vk_get_device_description(int device, char *description, size_t description_size) {
    if (!llamafile_has_vulkan()) {
        if (description_size > 0)
            description[0] = '\0';
        return;
    }
    if (!g_vulkan.get_device_description) {
        if (description_size > 0)
            snprintf(description, description_size, "Vulkan GPU %d", device);
        return;
    }
    gpu_call_get_description(g_vulkan.get_device_description, device, description, description_size);
}

void llamafile_vulkan_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_vulkan())
        return;
    gpu_call_log_set(g_vulkan.log_set, log_callback, user_data);
}
