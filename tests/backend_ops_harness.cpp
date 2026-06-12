// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
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

// Harness for upstream ggml's test-backend-ops under llamafile.
//
// Upstream test-backend-ops compares every ggml op (all MUL_MAT quant
// variants included) computed on each registered backend against the CPU
// reference, with NMSE thresholds. Upstream registers GPU backends at
// static-init time; llamafile instead loads them at runtime through the
// dlopen probe path (llamafile_has_* -> gpu_backend_register). This
// wrapper performs those loads, then delegates to the upstream test's
// main, which is compiled with -Dmain=backend_ops_main.
//
// Every backend DSO found at runtime (next to the executable or in
// ~/.llamafile/v/<version>/) is loaded and tested, so the same binary
// serves as the consistency gate for Vulkan, CUDA, ROCm, and Metal
// artifacts. Backends that fail to load are reported and skipped.
//
// Caveat: the CPU reference itself goes through llamafile's tinyBLAS/iqk
// fast path (GGML_USE_LLAMAFILE), so a mismatch can be a CPU-side bug as
// well as a GPU-side one. Run with LLAMAFILE_DISABLE_SGEMM=1 to fall back
// to vanilla ggml when attributing a failure to one side or the other
// (that toggle is how the iq4_xs/bf16 CPU bugs were isolated).
//
// Usage matches upstream, e.g.:
//   backend_ops_test test -o MUL_MAT
//   backend_ops_test test -b Vulkan0
//   backend_ops_test test                  # full op suite, all backends

#include <cstdio>

#include "llamafile.h"

int backend_ops_main(int argc, char **argv);

int main(int argc, char **argv) {
    FLAG_verbose = 1;  // show the probe path so DSO-resolution issues are visible

    struct {
        const char *name;
        bool (*probe)(void);
    } backends[] = {
        {"metal", llamafile_has_metal},
        {"cuda", llamafile_has_cuda},
        {"rocm", llamafile_has_amd_gpu},
        {"vulkan", llamafile_has_vulkan},
    };

    int loaded = 0;
    for (const auto &b : backends) {
        if (b.probe()) {
            fprintf(stderr, "backend_ops_harness: %s backend loaded\n", b.name);
            loaded++;
        } else {
            fprintf(stderr, "backend_ops_harness: %s backend not available\n", b.name);
        }
    }
    if (!loaded)
        fprintf(stderr,
                "backend_ops_harness: WARNING: no GPU backend loaded; "
                "only CPU will be tested\n");

    return backend_ops_main(argc, argv);
}
