// -*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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
// Runtime CUDA/ROCm GPU support for llamafile
//
// This file implements dynamic loading of CUDA/ROCm GPU support.
// At runtime on Linux/Windows with NVIDIA or AMD GPU:
//   1. Try to load pre-built DSO from /zip/ggml-cuda.so (bundled)
//   2. Or try to load from ~/.llamafile/ (pre-compiled)
//   3. Or compile at runtime if nvcc/hipcc is available
//   4. Load the DSO with cosmo_dlopen() and register the CUDA backend
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
#include <sys/wait.h>
#include <unistd.h>

// Forward declarations for ggml backend types
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;

// Function to register a backend with ggml (from ggml-backend.h)
extern void ggml_backend_register(ggml_backend_reg_t reg);

// Log callback type (must match ggml_log_callback from ggml.h)
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// CUDA backend state
static struct CudaBackend {
    bool supported;
    bool is_amd;  // true if this is ROCm/AMD, false if NVIDIA
    atomic_uint once;
    void *lib_handle;

    // Function pointers for CUDA backend
    ggml_backend_t (*backend_init)(int device);
    ggml_backend_reg_t (*backend_reg)(void);
    int (*get_device_count)(void);
    void (*get_device_description)(int device, char *description, size_t description_size);

    // Logging control
    void (*log_set)(llamafile_log_callback log_callback, void *user_data);
} g_cuda;

static bool LinkCuda(const char *dso) {
    // Load dynamic shared object using Cosmopolitan's dlopen
    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "cuda: %s: failed to load library\n", err ? err : "unknown error");
        return false;
    }

    // Import functions
    bool ok = true;

    *(void **)(&g_cuda.backend_init) = cosmo_dlsym(lib, "ggml_backend_cuda_init");
    ok &= (g_cuda.backend_init != NULL);

    *(void **)(&g_cuda.backend_reg) = cosmo_dlsym(lib, "ggml_backend_cuda_reg");
    ok &= (g_cuda.backend_reg != NULL);

    *(void **)(&g_cuda.get_device_count) = cosmo_dlsym(lib, "ggml_backend_cuda_get_device_count");
    // Optional - don't fail if not found

    *(void **)(&g_cuda.get_device_description) = cosmo_dlsym(lib, "ggml_backend_cuda_get_device_description");
    // Optional - don't fail if not found

    // Import logging control (optional)
    *(void **)(&g_cuda.log_set) = cosmo_dlsym(lib, "ggml_log_set");

    if (!ok) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "cuda: %s: not all symbols could be imported\n", err ? err : "unknown error");
        g_cuda.backend_init = NULL;
        g_cuda.backend_reg = NULL;
        g_cuda.get_device_count = NULL;
        g_cuda.get_device_description = NULL;
        g_cuda.log_set = NULL;
        cosmo_dlclose(lib);
        return false;
    }

    g_cuda.lib_handle = lib;
    return true;
}

static bool ImportCudaImpl(void) {
    // Skip on Apple Silicon (use Metal instead)
    if (IsXnuSilicon()) {
        return false;
    }

    // Check if we're allowed to even try
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_NVIDIA:
        break;
    case LLAMAFILE_GPU_AMD:
        g_cuda.is_amd = true;
        break;
    default:
        return false;
    }

    // Determine DSO name based on GPU type
    const char *ext = llamafile_get_dso_extension();
    char cuda_dso[64];
    char rocm_dso[64];
    snprintf(cuda_dso, sizeof(cuda_dso), "ggml-cuda.%s", ext);
    snprintf(rocm_dso, sizeof(rocm_dso), "ggml-rocm.%s", ext);

    // Try to load pre-built DSO
    if (FLAG_gpu == LLAMAFILE_GPU_AMD || FLAG_gpu == LLAMAFILE_GPU_AUTO) {
        if (llamafile_try_load_prebuilt_dso(rocm_dso, "cuda", LinkCuda)) {
            g_cuda.is_amd = true;
            goto RegisterBackend;
        }
    }

    if (FLAG_gpu == LLAMAFILE_GPU_NVIDIA || FLAG_gpu == LLAMAFILE_GPU_AUTO) {
        if (llamafile_try_load_prebuilt_dso(cuda_dso, "cuda", LinkCuda)) {
            g_cuda.is_amd = false;
            goto RegisterBackend;
        }
    }

    // No pre-built DSO found
    if (FLAG_verbose) {
        fprintf(stderr, "cuda: no pre-built GPU library found\n");
        fprintf(stderr, "cuda: to enable GPU support, build with:\n");
        fprintf(stderr, "cuda:   llamafile/cuda.sh   (for NVIDIA)\n");
        fprintf(stderr, "cuda:   llamafile/rocm.sh   (for AMD)\n");
    }
    return false;

RegisterBackend:
    // Suppress DSO's ggml logging before backend registration, which triggers
    // ggml_cuda_init() inside the DSO. Without this, CUDA device enumeration
    // messages appear even when --verbose is not set.
    if (!FLAG_verbose && g_cuda.log_set)
        g_cuda.log_set(llamafile_log_callback_null, NULL);

    // Register the CUDA backend with GGML
    if (g_cuda.backend_reg) {
        ggml_backend_reg_t reg = g_cuda.backend_reg();
        if (reg) {
            ggml_backend_register(reg);
            if (FLAG_verbose)
                fprintf(stderr, "cuda: %s backend registered with GGML\n",
                        g_cuda.is_amd ? "ROCm" : "CUDA");
        }
    }

    return true;
}

static void ImportCuda(void) {
    if (ImportCudaImpl()) {
        g_cuda.supported = true;
        if (FLAG_verbose) {
            fprintf(stderr, "cuda: %s GPU support successfully loaded\n",
                    g_cuda.is_amd ? "AMD ROCm" : "NVIDIA CUDA");
            if (g_cuda.get_device_count) {
                int count = g_cuda.get_device_count();
                fprintf(stderr, "cuda: found %d GPU device(s)\n", count);
            }
        }
    } else if (FLAG_gpu == LLAMAFILE_GPU_NVIDIA || FLAG_gpu == LLAMAFILE_GPU_AMD) {
        fprintf(stderr, "fatal error: support for --gpu %s was explicitly requested, "
                "but it wasn't available\n", llamafile_describe_gpu());
        exit(1);
    }
}

bool llamafile_has_cuda(void) {
    cosmo_once(&g_cuda.once, ImportCuda);
    return g_cuda.supported && !g_cuda.is_amd;
}

bool llamafile_has_amd_gpu(void) {
    cosmo_once(&g_cuda.once, ImportCuda);
    return g_cuda.supported && g_cuda.is_amd;
}

// Wrapper functions for dynamically loaded CUDA backend

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return NULL;
    if (!g_cuda.backend_init)
        return NULL;
    return g_cuda.backend_init(device);
}

int ggml_backend_cuda_get_device_count(void) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return 0;
    if (!g_cuda.get_device_count)
        return 0;
    return g_cuda.get_device_count();
}

void ggml_backend_cuda_get_device_description(int device, char *description, size_t description_size) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu()) {
        if (description_size > 0)
            description[0] = '\0';
        return;
    }
    if (!g_cuda.get_device_description) {
        if (description_size > 0)
            snprintf(description, description_size, "GPU %d", device);
        return;
    }
    g_cuda.get_device_description(device, description, description_size);
}

void llamafile_cuda_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return;
    if (g_cuda.log_set)
        g_cuda.log_set(log_callback, user_data);
}
