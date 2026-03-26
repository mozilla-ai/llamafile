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
static struct VulkanBackend {
    bool supported;
    atomic_uint once;
    void *lib_handle;

    // Function pointers for Vulkan backend
    ggml_backend_t (*backend_init)(size_t device);
    ggml_backend_reg_t (*backend_reg)(void);
    int (*get_device_count)(void);
    void (*get_device_description)(int device, char *description, size_t description_size);

    // Logging control
    void (*log_set)(llamafile_log_callback log_callback, void *user_data);
} g_vulkan;

static bool LinkVulkan(const char *dso) {
    // Load dynamic shared object using Cosmopolitan's dlopen
    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "vulkan: %s: failed to load library\n", err ? err : "unknown error");
        return false;
    }

    // Import functions
    bool ok = true;

    *(void **)(&g_vulkan.backend_init) = cosmo_dlsym(lib, "ggml_backend_vk_init");
    ok &= (g_vulkan.backend_init != NULL);

    *(void **)(&g_vulkan.backend_reg) = cosmo_dlsym(lib, "ggml_backend_vk_reg");
    ok &= (g_vulkan.backend_reg != NULL);

    *(void **)(&g_vulkan.get_device_count) = cosmo_dlsym(lib, "ggml_backend_vk_get_device_count");
    // Optional - don't fail if not found

    *(void **)(&g_vulkan.get_device_description) = cosmo_dlsym(lib, "ggml_backend_vk_get_device_description");
    // Optional - don't fail if not found

    // Import logging control (optional)
    *(void **)(&g_vulkan.log_set) = cosmo_dlsym(lib, "ggml_log_set");

    if (!ok) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "vulkan: %s: not all symbols could be imported\n", err ? err : "unknown error");
        g_vulkan.backend_init = NULL;
        g_vulkan.backend_reg = NULL;
        g_vulkan.get_device_count = NULL;
        g_vulkan.get_device_description = NULL;
        g_vulkan.log_set = NULL;
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
        if (FLAG_verbose) {
            fprintf(stderr, "vulkan: no pre-built GPU library found\n");
            fprintf(stderr, "vulkan: to enable Vulkan support, build with:\n");
            fprintf(stderr, "vulkan:   llamafile/vulkan.sh\n");
        }
        return false;
    }

    // Register the Vulkan backend with GGML
    if (g_vulkan.backend_reg) {
        ggml_backend_reg_t reg = g_vulkan.backend_reg();
        if (reg) {
            ggml_backend_register(reg);
            if (FLAG_verbose)
                fprintf(stderr, "vulkan: Vulkan backend registered with GGML\n");
        }
    }

    return true;
}

static void ImportVulkan(void) {
    if (ImportVulkanImpl()) {
        g_vulkan.supported = true;
        if (FLAG_verbose) {
            fprintf(stderr, "vulkan: Vulkan GPU support successfully loaded\n");
            if (g_vulkan.get_device_count) {
                int count = g_vulkan.get_device_count();
                fprintf(stderr, "vulkan: found %d GPU device(s)\n", count);
            }
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
    if (!g_vulkan.backend_init)
        return NULL;
    return g_vulkan.backend_init(device);
}

int ggml_backend_vk_get_device_count(void) {
    if (!llamafile_has_vulkan())
        return 0;
    if (!g_vulkan.get_device_count)
        return 0;
    return g_vulkan.get_device_count();
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
    g_vulkan.get_device_description(device, description, description_size);
}

void llamafile_vulkan_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_vulkan())
        return;
    if (g_vulkan.log_set)
        g_vulkan.log_set(log_callback, user_data);
}
