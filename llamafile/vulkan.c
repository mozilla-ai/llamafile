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

#include "llamafile.h"
#include <cosmo.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// Forward declarations for ggml backend types
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;

// Function to register a backend with ggml (from ggml-backend.h)
extern void ggml_backend_register(ggml_backend_reg_t reg);

// Log callback type (must match ggml_log_callback from ggml.h)
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// Vulkan backend state
//
// On Windows the DSO exports functions with ms_abi calling convention,
// but the cosmocc host uses System V ABI.  We store each dlsym'd pointer
// in a union so the correct ABI variant is called at each call site,
// following the same pattern used in cuda.c and localscore/nvml.cpp.
static struct VulkanBackend {
    bool supported;
    atomic_uint once;
    void *lib_handle;

    // Function pointers for Vulkan backend
    union {
        ggml_backend_t (*default_abi)(size_t device);
        ggml_backend_t (__attribute__((__ms_abi__)) *windows_abi)(size_t device);
    } backend_init;

    union {
        ggml_backend_reg_t (*default_abi)(void);
        ggml_backend_reg_t (__attribute__((__ms_abi__)) *windows_abi)(void);
    } backend_reg;

    union {
        int (*default_abi)(void);
        int (__attribute__((__ms_abi__)) *windows_abi)(void);
    } get_device_count;

    union {
        void (*default_abi)(int device, char *description, size_t description_size);
        void (__attribute__((__ms_abi__)) *windows_abi)(int device, char *description, size_t description_size);
    } get_device_description;

    // Logging control
    union {
        void (*default_abi)(llamafile_log_callback log_callback, void *user_data);
        void (__attribute__((__ms_abi__)) *windows_abi)(llamafile_log_callback log_callback, void *user_data);
    } log_set;
} g_vulkan;

static bool LinkVulkan(const char *dso) {
    // Load dynamic shared object using Cosmopolitan's dlopen
    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        char *err = cosmo_dlerror();
        llamafile_info("vulkan", "failed to load library %s: %s",
                       dso, err ? err : "unknown error");
        return false;
    }

    // Import functions into the correct ABI union member
    bool ok = true;
    void *sym;

    sym = cosmo_dlsym(lib, "ggml_backend_vk_init");
    ok &= (sym != NULL);
    if (IsWindows())
        *(void **)(&g_vulkan.backend_init.windows_abi) = sym;
    else
        *(void **)(&g_vulkan.backend_init.default_abi) = sym;

    sym = cosmo_dlsym(lib, "ggml_backend_vk_reg");
    ok &= (sym != NULL);
    if (IsWindows())
        *(void **)(&g_vulkan.backend_reg.windows_abi) = sym;
    else
        *(void **)(&g_vulkan.backend_reg.default_abi) = sym;

    // Optional - don't fail if not found
    sym = cosmo_dlsym(lib, "ggml_backend_vk_get_device_count");
    if (IsWindows())
        *(void **)(&g_vulkan.get_device_count.windows_abi) = sym;
    else
        *(void **)(&g_vulkan.get_device_count.default_abi) = sym;

    // Optional - don't fail if not found
    sym = cosmo_dlsym(lib, "ggml_backend_vk_get_device_description");
    if (IsWindows())
        *(void **)(&g_vulkan.get_device_description.windows_abi) = sym;
    else
        *(void **)(&g_vulkan.get_device_description.default_abi) = sym;

    // Import logging control (optional)
    sym = cosmo_dlsym(lib, "ggml_log_set");
    if (IsWindows())
        *(void **)(&g_vulkan.log_set.windows_abi) = sym;
    else
        *(void **)(&g_vulkan.log_set.default_abi) = sym;

    if (!ok) {
        char *err = cosmo_dlerror();
        llamafile_info("vulkan", "could not import all symbols from %s: %s",
                       dso, err ? err : "unknown error");
        memset(&g_vulkan.backend_init, 0, sizeof(g_vulkan.backend_init));
        memset(&g_vulkan.backend_reg, 0, sizeof(g_vulkan.backend_reg));
        memset(&g_vulkan.get_device_count, 0, sizeof(g_vulkan.get_device_count));
        memset(&g_vulkan.get_device_description, 0, sizeof(g_vulkan.get_device_description));
        memset(&g_vulkan.log_set, 0, sizeof(g_vulkan.log_set));
        cosmo_dlclose(lib);
        return false;
    }

    g_vulkan.lib_handle = lib;
    return true;
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

    // Suppress DSO's ggml logging before backend registration, which triggers
    // device enumeration inside the DSO. Without this, Vulkan device messages
    // appear even when --verbose is not set.
    if (!FLAG_verbose && (g_vulkan.log_set.default_abi || g_vulkan.log_set.windows_abi)) {
        if (IsWindows())
            g_vulkan.log_set.windows_abi(llamafile_log_callback_null, NULL);
        else
            g_vulkan.log_set.default_abi(llamafile_log_callback_null, NULL);
    }

    // Register the Vulkan backend with GGML
    if (g_vulkan.backend_reg.default_abi || g_vulkan.backend_reg.windows_abi) {
        ggml_backend_reg_t reg;
        if (IsWindows())
            reg = g_vulkan.backend_reg.windows_abi();
        else
            reg = g_vulkan.backend_reg.default_abi();
        if (reg) {
            ggml_backend_register(reg);
            llamafile_info("vulkan", "Vulkan backend registered with GGML");
        }
    }

    return true;
}

static void ImportVulkan(void) {
    if (ImportVulkanImpl()) {
        g_vulkan.supported = true;
        llamafile_info("vulkan", "Vulkan GPU support successfully loaded");
        if (g_vulkan.get_device_count.default_abi || g_vulkan.get_device_count.windows_abi) {
            int count;
            if (IsWindows())
                count = g_vulkan.get_device_count.windows_abi();
            else
                count = g_vulkan.get_device_count.default_abi();
            llamafile_info("vulkan", "found %d GPU device(s)", count);
        }
    } else if (FLAG_gpu == LLAMAFILE_GPU_VULKAN) {
        fprintf(stderr, "fatal error: support for --gpu vulkan was explicitly requested, "
                "but it wasn't available\n");
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
    if (!g_vulkan.backend_init.default_abi && !g_vulkan.backend_init.windows_abi)
        return NULL;
    if (IsWindows())
        return g_vulkan.backend_init.windows_abi(device);
    return g_vulkan.backend_init.default_abi(device);
}

int ggml_backend_vk_get_device_count(void) {
    if (!llamafile_has_vulkan())
        return 0;
    if (!g_vulkan.get_device_count.default_abi && !g_vulkan.get_device_count.windows_abi)
        return 0;
    if (IsWindows())
        return g_vulkan.get_device_count.windows_abi();
    return g_vulkan.get_device_count.default_abi();
}

void ggml_backend_vk_get_device_description(int device, char *description, size_t description_size) {
    if (!llamafile_has_vulkan()) {
        if (description_size > 0)
            description[0] = '\0';
        return;
    }
    if (!g_vulkan.get_device_description.default_abi && !g_vulkan.get_device_description.windows_abi) {
        if (description_size > 0)
            snprintf(description, description_size, "Vulkan GPU %d", device);
        return;
    }
    if (IsWindows())
        g_vulkan.get_device_description.windows_abi(device, description, description_size);
    else
        g_vulkan.get_device_description.default_abi(device, description, description_size);
}

void llamafile_vulkan_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_vulkan())
        return;
    if (g_vulkan.log_set.default_abi || g_vulkan.log_set.windows_abi) {
        if (IsWindows())
            g_vulkan.log_set.windows_abi(log_callback, user_data);
        else
            g_vulkan.log_set.default_abi(log_callback, user_data);
    }
}
