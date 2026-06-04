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
//   3. Load the DSO with cosmo_dlopen() and register the CUDA backend
//
// The load/probe/register mechanics live in gpu_backend.c, shared with the
// Vulkan backend so all of them behave identically. CUDA and ROCm use the same
// exported symbols and the same DSO-loading machinery, so they share one
// GpuBackend; which descriptor loaded records whether it's NVIDIA or AMD.
//

#include "gpu_backend.h"
#include "llamafile.h"
#include <cosmo.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA and ROCm share symbol names but differ in display name, so they are two
// descriptors over one GpuBackend. The loaded descriptor tells NVIDIA vs AMD.
static GpuBackend g_cuda;

static const GpuBackendDesc CUDA_DESC = {
    .tag = "cuda",
    .name = "CUDA",
    .init = "ggml_backend_cuda_init",
    .reg = "ggml_backend_cuda_reg",
    .get_device_count = "ggml_backend_cuda_get_device_count",
    .get_device_description = "ggml_backend_cuda_get_device_description",
};

static const GpuBackendDesc ROCM_DESC = {
    .tag = "cuda",
    .name = "ROCm",
    .init = "ggml_backend_cuda_init",
    .reg = "ggml_backend_cuda_reg",
    .get_device_count = "ggml_backend_cuda_get_device_count",
    .get_device_description = "ggml_backend_cuda_get_device_description",
};

// Thin llamafile_link_dso_fn thunks over the shared linker, one per identity.
static bool LinkCudaNvidia(const char *dso) {
    return gpu_backend_link(&g_cuda, dso, &CUDA_DESC);
}

static bool LinkCudaAmd(const char *dso) {
    return gpu_backend_link(&g_cuda, dso, &ROCM_DESC);
}

// Load a prebuilt DSO and commit to it only if it exposes a usable device. The
// DSO loads fine even with no compatible hardware, so gpu_backend_probe()
// rejects 0-device backends and lets AUTO mode fall through to the next one.
static bool TryGpuBackend(const char *dso, llamafile_link_dso_fn link_fn) {
    if (!llamafile_try_load_prebuilt_dso(dso, "cuda", link_fn))
        return false;
    // On Windows a faulting driver-init behind get_device_count() can corrupt
    // the process if caught in-process; probe out-of-process there (#988).
    return IsWindows() ? gpu_backend_probe_oop(&g_cuda) : gpu_backend_probe(&g_cuda);
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
    case LLAMAFILE_GPU_AMD:
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

    // In AUTO mode, prefer CUDA over ROCm: it covers the common NVIDIA case
    // and lets ROCm be the fallback when CUDA is absent or has no devices.
    if (FLAG_gpu == LLAMAFILE_GPU_NVIDIA || FLAG_gpu == LLAMAFILE_GPU_AUTO) {
        if (TryGpuBackend(cuda_dso, LinkCudaNvidia))
            goto RegisterBackend;
    }

    if (FLAG_gpu == LLAMAFILE_GPU_AMD || FLAG_gpu == LLAMAFILE_GPU_AUTO) {
        if (TryGpuBackend(rocm_dso, LinkCudaAmd))
            goto RegisterBackend;
    }

    // No pre-built DSO found
    llamafile_info("cuda", "no pre-built GPU library found");
    llamafile_info("cuda", "to enable GPU support, build with:");
    llamafile_info("cuda", "  llamafile/cuda.sh   (for NVIDIA)");
    llamafile_info("cuda", "  llamafile/rocm.sh   (for AMD)");
    return false;

RegisterBackend:
    gpu_backend_register(&g_cuda);
    return true;
}

static void ImportCuda(void) {
    if (ImportCudaImpl()) {
        g_cuda.supported = true;
        llamafile_info("cuda", "%s GPU support successfully loaded",
                       g_cuda.desc == &ROCM_DESC ? "AMD ROCm" : "NVIDIA CUDA");
        int count = gpu_call_device_count(g_cuda.get_device_count);
        llamafile_info("cuda", "found %d GPU device(s)", count);
    } else if (FLAG_gpu == LLAMAFILE_GPU_NVIDIA || FLAG_gpu == LLAMAFILE_GPU_AMD) {
        fprintf(stderr, "fatal error: support for --gpu %s was explicitly requested, "
                "but it wasn't available\n", llamafile_describe_gpu());
        exit(1);
    }
}

bool llamafile_has_cuda(void) {
    cosmo_once(&g_cuda.once, ImportCuda);
    return g_cuda.supported && g_cuda.desc != &ROCM_DESC;
}

bool llamafile_has_amd_gpu(void) {
    cosmo_once(&g_cuda.once, ImportCuda);
    return g_cuda.supported && g_cuda.desc == &ROCM_DESC;
}

// Wrapper functions for dynamically loaded CUDA backend

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return NULL;
    void *fp = g_cuda.backend_init;
    if (!fp)
        return NULL;
    // backend_init has a CUDA-specific signature (int device), so it is called
    // here rather than through a shared gpu_call_* helper.
    if (IsWindows())
        return ((ggml_backend_t(__attribute__((__ms_abi__)) *)(int))fp)(device);
    return ((ggml_backend_t(*)(int))fp)(device);
}

int ggml_backend_cuda_get_device_count(void) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return 0;
    return gpu_call_device_count(g_cuda.get_device_count);
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
    gpu_call_get_description(g_cuda.get_device_description, device, description, description_size);
}

void llamafile_cuda_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_cuda() && !llamafile_has_amd_gpu())
        return;
    gpu_call_log_set(g_cuda.log_set, log_callback, user_data);
}
