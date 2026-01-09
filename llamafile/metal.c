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
// Runtime Metal GPU support for llamafile
//
// This file implements dynamic compilation and loading of Metal GPU support.
// At runtime on macOS ARM64:
//   1. Extract Metal source files from /zip/ to ~/.llamafile/
//   2. Preprocess ggml-metal.metal to inline header files
//   3. Compile everything into a self-contained dylib using system cc
//   4. Load the dylib with cosmo_dlopen() and register the Metal backend
//
// The dylib is self-contained (includes ggml core) because cosmo_dlopen()
// cannot resolve symbols from the parent process.
//

#include "llamafile.h"
#include <cosmo.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// Declare environ for posix_spawn
extern char **environ;

// Embed all Metal source files into the executable
// Core ggml (for self-contained dylib)
__static_yoink("llama.cpp/ggml/src/ggml.c");
__static_yoink("llama.cpp/ggml/src/ggml-alloc.c");
__static_yoink("llama.cpp/ggml/src/ggml-backend.cpp");
__static_yoink("llama.cpp/ggml/src/ggml-quants.c");
__static_yoink("llama.cpp/ggml/src/ggml-threading.cpp");

// Headers
__static_yoink("llama.cpp/ggml/include/ggml.h");
__static_yoink("llama.cpp/ggml/include/gguf.h");
__static_yoink("llama.cpp/ggml/include/ggml-cpu.h");
__static_yoink("llama.cpp/ggml/include/ggml-alloc.h");
__static_yoink("llama.cpp/ggml/include/ggml-backend.h");
__static_yoink("llama.cpp/ggml/include/ggml-metal.h");
__static_yoink("llama.cpp/ggml/src/ggml-impl.h");
__static_yoink("llama.cpp/ggml/src/ggml-common.h");
__static_yoink("llama.cpp/ggml/src/ggml-quants.h");
__static_yoink("llama.cpp/ggml/src/ggml-threading.h");
__static_yoink("llama.cpp/ggml/src/ggml-backend-impl.h");
__static_yoink("llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h");

// Metal backend
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal.metal");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-context.h");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-common.h");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.h");
__static_yoink("llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp");

// Sources to extract at runtime
static const struct MetalSource {
    const char *zip;
    const char *name;
} metal_srcs[] = {
    // Core ggml headers
    {"/zip/llama.cpp/ggml/include/ggml.h", "ggml.h"},
    {"/zip/llama.cpp/ggml/include/gguf.h", "gguf.h"},
    {"/zip/llama.cpp/ggml/include/ggml-cpu.h", "ggml-cpu.h"},
    {"/zip/llama.cpp/ggml/include/ggml-alloc.h", "ggml-alloc.h"},
    {"/zip/llama.cpp/ggml/include/ggml-backend.h", "ggml-backend.h"},
    {"/zip/llama.cpp/ggml/include/ggml-metal.h", "ggml-metal.h"},
    {"/zip/llama.cpp/ggml/src/ggml-impl.h", "ggml-impl.h"},
    {"/zip/llama.cpp/ggml/src/ggml-common.h", "ggml-common.h"},
    {"/zip/llama.cpp/ggml/src/ggml-quants.h", "ggml-quants.h"},
    {"/zip/llama.cpp/ggml/src/ggml-threading.h", "ggml-threading.h"},
    {"/zip/llama.cpp/ggml/src/ggml-backend-impl.h", "ggml-backend-impl.h"},
    {"/zip/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h", "ggml-cpu/ggml-cpu-impl.h"},

    // Core ggml implementation - needed for self-contained dylib
    {"/zip/llama.cpp/ggml/src/ggml.c", "ggml.c"},
    {"/zip/llama.cpp/ggml/src/ggml-alloc.c", "ggml-alloc.c"},
    {"/zip/llama.cpp/ggml/src/ggml-backend.cpp", "ggml-backend.cpp"},
    {"/zip/llama.cpp/ggml/src/ggml-quants.c", "ggml-quants.c"},
    {"/zip/llama.cpp/ggml/src/ggml-threading.cpp", "ggml-threading.cpp"},

    // Metal-specific files
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp", "ggml-metal.cpp"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal", "ggml-metal.metal"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h", "ggml-metal-impl.h"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h", "ggml-metal-device.h"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m", "ggml-metal-device.m"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp", "ggml-metal-device.cpp"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.h", "ggml-metal-context.h"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m", "ggml-metal-context.m"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.h", "ggml-metal-common.h"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-common.cpp", "ggml-metal-common.cpp"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.h", "ggml-metal-ops.h"},
    {"/zip/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp", "ggml-metal-ops.cpp"},
};

// Forward declarations for ggml backend types
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_backend_reg * ggml_backend_reg_t;

// Function to register a backend with ggml (from ggml-backend.h)
extern void ggml_backend_register(ggml_backend_reg_t reg);

// Log callback type (must match ggml_log_callback from ggml.h)
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// Function pointers for dynamically loaded Metal backend
static struct MetalBackend {
    bool supported;
    atomic_uint once;
    void *lib_handle;

    // Function pointers matching ggml-metal.h
    ggml_backend_t (*backend_init)(void);
    bool (*backend_is_metal)(ggml_backend_t backend);
    ggml_backend_reg_t (*backend_metal_reg)(void);

    // Logging control
    void (*log_set)(llamafile_log_callback log_callback, void *user_data);
} g_metal;

static int makedirs(const char *path, mode_t mode) {
    char tmp[PATH_MAX];
    char *p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);

    if (tmp[len - 1] == '/')
        tmp[len - 1] = '\0';

    if (mkdir(tmp, mode) == 0)
        return 0;

    if (errno == EEXIST) {
        struct stat st;
        if (stat(tmp, &st) == 0 && S_ISDIR(st.st_mode))
            return 0;
        return -1;
    }

    if (errno != ENOENT)
        return -1;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, mode) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }

    return mkdir(tmp, mode);
}

static char *read_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *content = malloc(size + 1);
    if (!content) {
        fclose(f);
        return NULL;
    }

    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);

    if (size_out) *size_out = size;
    return content;
}

static bool write_file(const char *path, const char *content, size_t size) {
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fwrite(content, 1, size, f);
    fclose(f);
    return true;
}

// Preprocess ggml-metal.metal to inline headers
// Metal runtime compiler doesn't support include paths
static bool PreprocessMetalShader(const char *app_dir) {
    char metal_path[PATH_MAX];
    char common_path[PATH_MAX];
    char impl_path[PATH_MAX];

    snprintf(metal_path, PATH_MAX, "%sggml-metal.metal", app_dir);
    snprintf(common_path, PATH_MAX, "%sggml-common.h", app_dir);
    snprintf(impl_path, PATH_MAX, "%sggml-metal-impl.h", app_dir);

    size_t common_size, impl_size, metal_size;
    char *common_content = read_file(common_path, &common_size);
    if (!common_content) {
        fprintf(stderr, "metal: failed to read %s\n", common_path);
        return false;
    }

    char *impl_content = read_file(impl_path, &impl_size);
    if (!impl_content) {
        free(common_content);
        fprintf(stderr, "metal: failed to read %s\n", impl_path);
        return false;
    }

    char *metal_content = read_file(metal_path, &metal_size);
    if (!metal_content) {
        free(common_content);
        free(impl_content);
        fprintf(stderr, "metal: failed to read %s\n", metal_path);
        return false;
    }

    // Find #include directives
    char *include_common = strstr(metal_content, "#include \"ggml-common.h\"");
    char *include_impl = strstr(metal_content, "#include \"ggml-metal-impl.h\"");

    if (!include_common && !include_impl) {
        // Nothing to preprocess
        free(common_content);
        free(impl_content);
        free(metal_content);
        return true;
    }

    // Create preprocessed content
    FILE *fout = fopen(metal_path, "w");
    if (!fout) {
        free(common_content);
        free(impl_content);
        free(metal_content);
        fprintf(stderr, "metal: failed to write to %s\n", metal_path);
        return false;
    }

    char *pos = metal_content;

    // Process includes in order they appear
    if (include_common && (!include_impl || include_common < include_impl)) {
        // Write everything before #include "ggml-common.h"
        fwrite(pos, 1, include_common - pos, fout);
        fputs("// ggml-common.h inlined below\n", fout);
        fwrite(common_content, 1, common_size, fout);
        fputs("\n// end of ggml-common.h\n", fout);
        pos = strchr(include_common, '\n');
        if (pos) pos++;
    }

    if (include_impl && pos && include_impl >= pos) {
        // Write everything between includes
        fwrite(pos, 1, include_impl - pos, fout);
        fputs("// ggml-metal-impl.h inlined below\n", fout);
        fwrite(impl_content, 1, impl_size, fout);
        fputs("\n// end of ggml-metal-impl.h\n", fout);
        pos = strchr(include_impl, '\n');
        if (pos) pos++;
    }

    // Write rest of file
    if (pos) {
        fputs(pos, fout);
    }

    fclose(fout);
    free(common_content);
    free(impl_content);
    free(metal_content);

    if (FLAG_verbose)
        fprintf(stderr, "metal: preprocessed %s\n", metal_path);

    return true;
}

static bool BuildMetal(const char *dso) {
    char app_dir[PATH_MAX];
    char src[PATH_MAX];
    bool needs_rebuild = false;

    llamafile_get_app_dir(app_dir, PATH_MAX);

    // Check if dylib already exists for this version
    // Since we use versioned paths, source updates come with new versions
    struct stat dso_stat;
    if (stat(dso, &dso_stat) == 0 && !FLAG_recompile) {
        if (FLAG_verbose)
            fprintf(stderr, "metal: using cached %s\n", dso);
        return true;
    }

    // Create app directory
    if (makedirs(app_dir, 0755) != 0) {
        perror(app_dir);
        return false;
    }

    // Extract all source files
    for (size_t i = 0; i < sizeof(metal_srcs) / sizeof(*metal_srcs); ++i) {
        snprintf(src, PATH_MAX, "%s%s", app_dir, metal_srcs[i].name);

        // Create parent directories if needed
        char *last_slash = strrchr(src, '/');
        if (last_slash && last_slash != src) {
            char parent_dir[PATH_MAX];
            size_t parent_len = last_slash - src;
            memcpy(parent_dir, src, parent_len);
            parent_dir[parent_len] = '\0';
            makedirs(parent_dir, 0755);
        }

        switch (llamafile_is_file_newer_than(metal_srcs[i].zip, src)) {
        case -1:
            return false;
        case 0:
            break;
        case 1:
            needs_rebuild = true;
            if (!llamafile_extract(metal_srcs[i].zip, src)) {
                return false;
            }
            break;
        default:
            __builtin_unreachable();
        }
    }

    // Preprocess Metal shader if rebuild needed
    if (needs_rebuild) {
        if (!PreprocessMetalShader(app_dir)) {
            return false;
        }
    }

    // Check if dylib needs rebuild
    snprintf(src, PATH_MAX, "%sggml-metal.cpp", app_dir);
    if (!needs_rebuild) {
        switch (llamafile_is_file_newer_than(src, dso)) {
        case -1:
            return false;
        case 0:
            break;
        case 1:
            needs_rebuild = true;
            break;
        default:
            __builtin_unreachable();
        }
    }

    // Compile dynamic shared object
    if (needs_rebuild || FLAG_recompile) {
        if (FLAG_verbose)
            fprintf(stderr, "metal: building ggml-metal.dylib with xcode...\n");

        char tmpdso[PATH_MAX];
        snprintf(tmpdso, PATH_MAX, "%s.XXXXXX", dso);
        int fd = mkstemp(tmpdso);
        if (fd == -1) {
            perror(tmpdso);
            return false;
        }
        close(fd);

        // Build paths to source files
        char src_ggml_c[PATH_MAX];
        char src_ggml_alloc_c[PATH_MAX];
        char src_ggml_backend_cpp[PATH_MAX];
        char src_threading_cpp[PATH_MAX];
        char src_quants_c[PATH_MAX];
        char src_metal_cpp[PATH_MAX];
        char src_metal_device_cpp[PATH_MAX];
        char src_metal_device_m[PATH_MAX];
        char src_metal_context_m[PATH_MAX];
        char src_metal_common_cpp[PATH_MAX];
        char src_metal_ops_cpp[PATH_MAX];

        snprintf(src_ggml_c, PATH_MAX, "%sggml.c", app_dir);
        snprintf(src_ggml_alloc_c, PATH_MAX, "%sggml-alloc.c", app_dir);
        snprintf(src_ggml_backend_cpp, PATH_MAX, "%sggml-backend.cpp", app_dir);
        snprintf(src_threading_cpp, PATH_MAX, "%sggml-threading.cpp", app_dir);
        snprintf(src_quants_c, PATH_MAX, "%sggml-quants.c", app_dir);
        snprintf(src_metal_cpp, PATH_MAX, "%sggml-metal.cpp", app_dir);
        snprintf(src_metal_device_cpp, PATH_MAX, "%sggml-metal-device.cpp", app_dir);
        snprintf(src_metal_device_m, PATH_MAX, "%sggml-metal-device.m", app_dir);
        snprintf(src_metal_context_m, PATH_MAX, "%sggml-metal-context.m", app_dir);
        snprintf(src_metal_common_cpp, PATH_MAX, "%sggml-metal-common.cpp", app_dir);
        snprintf(src_metal_ops_cpp, PATH_MAX, "%sggml-metal-ops.cpp", app_dir);

        // Build include path
        char include_arg[PATH_MAX + 2];
        snprintf(include_arg, sizeof(include_arg), "-I%s", app_dir);

        // Build args array dynamically (to support optional quiet compilation)
        char *args[64];
        int argc = 0;
        args[argc++] = "cc";
        args[argc++] = include_arg;
        args[argc++] = "-O3";
        args[argc++] = "-fPIC";
        args[argc++] = "-shared";
        args[argc++] = "-pthread";
        args[argc++] = "-DNDEBUG";
        args[argc++] = "-ffixed-x28";  // cosmo's TLS register
        args[argc++] = "-DTARGET_OS_OSX";
        args[argc++] = "-DGGML_MULTIPLATFORM";
        args[argc++] = "-DGGML_VERSION=\"0.9.4\"";
        args[argc++] = "-DGGML_COMMIT=\"unknown\"";
        args[argc++] = "-w";  // Suppress compilation warnings
        args[argc++] = src_ggml_c;
        args[argc++] = src_ggml_alloc_c;
        args[argc++] = src_ggml_backend_cpp;
        args[argc++] = src_threading_cpp;
        args[argc++] = src_quants_c;
        args[argc++] = src_metal_cpp;
        args[argc++] = src_metal_device_cpp;
        args[argc++] = src_metal_device_m;
        args[argc++] = src_metal_context_m;
        args[argc++] = src_metal_common_cpp;
        args[argc++] = src_metal_ops_cpp;
        args[argc++] = "-o";
        args[argc++] = tmpdso;
        args[argc++] = "-framework";
        args[argc++] = "Foundation";
        args[argc++] = "-framework";
        args[argc++] = "Metal";
        args[argc++] = "-framework";
        args[argc++] = "MetalKit";
        args[argc++] = "-lc++";
        args[argc] = NULL;

        if (FLAG_verbose) {
            fprintf(stderr, "metal: executing: cc");
            for (int i = 1; args[i]; i++)
                fprintf(stderr, " %s", args[i]);
            fprintf(stderr, "\n");
        }

        int pid, ws;
        errno_t err = posix_spawnp(&pid, "cc", NULL, NULL, args, environ);
        if (err) {
            perror("cc");
            if (err == ENOENT) {
                fprintf(stderr, "metal: PLEASE RUN: xcode-select --install\n");
            }
            unlink(tmpdso);
            return false;
        }

        while (waitpid(pid, &ws, 0) == -1) {
            if (errno != EINTR) {
                perror("waitpid");
                unlink(tmpdso);
                return false;
            }
        }

        if (ws) {
            fprintf(stderr, "metal: compiler returned nonzero exit status\n");
            unlink(tmpdso);
            return false;
        }

        if (rename(tmpdso, dso)) {
            perror(dso);
            unlink(tmpdso);
            return false;
        }

        if (FLAG_verbose)
            fprintf(stderr, "metal: successfully built %s\n", dso);
    }

    return true;
}

static bool LinkMetal(const char *dso) {
    // Load dynamic shared object using Cosmopolitan's dlopen
    void *lib = cosmo_dlopen(dso, RTLD_LAZY);
    if (!lib) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "metal: %s: failed to load library\n", err ? err : "unknown error");
        return false;
    }

    // Import functions
    bool ok = true;
    *(void **)(&g_metal.backend_init) = cosmo_dlsym(lib, "ggml_backend_metal_init");
    ok &= (g_metal.backend_init != NULL);

    *(void **)(&g_metal.backend_is_metal) = cosmo_dlsym(lib, "ggml_backend_is_metal");
    ok &= (g_metal.backend_is_metal != NULL);

    *(void **)(&g_metal.backend_metal_reg) = cosmo_dlsym(lib, "ggml_backend_metal_reg");
    ok &= (g_metal.backend_metal_reg != NULL);

    // Import logging control (optional - don't fail if not found)
    *(void **)(&g_metal.log_set) = cosmo_dlsym(lib, "ggml_log_set");

    if (!ok) {
        char *err = cosmo_dlerror();
        fprintf(stderr, "metal: %s: not all symbols could be imported\n", err ? err : "unknown error");
        cosmo_dlclose(lib);
        return false;
    }

    g_metal.lib_handle = lib;
    return true;
}

static bool ImportMetalImpl(void) {
    // Ensure this is macOS ARM64
    if (!IsXnuSilicon()) {
        return false;
    }

    // Check if we're allowed to even try
    switch (FLAG_gpu) {
    case LLAMAFILE_GPU_AUTO:
    case LLAMAFILE_GPU_APPLE:
        break;
    default:
        return false;
    }

    // Get path of DSO
    char dso[PATH_MAX];
    char app_dir[PATH_MAX];
    llamafile_get_app_dir(app_dir, PATH_MAX);
    snprintf(dso, PATH_MAX, "%sggml-metal.dylib", app_dir);

    if (FLAG_nocompile) {
        return LinkMetal(dso);
    }

    // Build and link Metal support DSO if possible
    if (BuildMetal(dso)) {
        if (LinkMetal(dso)) {
            // Register the Metal backend with GGML
            if (g_metal.backend_metal_reg) {
                ggml_backend_reg_t reg = g_metal.backend_metal_reg();
                if (reg) {
                    ggml_backend_register(reg);
                    if (FLAG_verbose)
                        fprintf(stderr, "metal: Metal backend registered with GGML\n");
                }
            }
            return true;
        }
    }
    return false;
}

static void ImportMetal(void) {
    if (ImportMetalImpl()) {
        g_metal.supported = true;
        if (FLAG_verbose)
            fprintf(stderr, "metal: Apple Metal GPU support successfully loaded\n");
    } else if (FLAG_gpu == LLAMAFILE_GPU_APPLE) {
        fprintf(stderr, "fatal error: support for --gpu %s was explicitly requested, "
                "but it wasn't available\n", llamafile_describe_gpu());
        exit(1);
    }
}

bool llamafile_has_metal(void) {
    cosmo_once(&g_metal.once, ImportMetal);
    return g_metal.supported;
}

// Wrapper functions that forward to dynamically loaded Metal backend

ggml_backend_t ggml_backend_metal_init(void) {
    if (!llamafile_has_metal())
        return NULL;
    return g_metal.backend_init();
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    if (!llamafile_has_metal())
        return false;
    return g_metal.backend_is_metal(backend);
}

void llamafile_metal_log_set(llamafile_log_callback log_callback, void *user_data) {
    if (!llamafile_has_metal())
        return;
    if (g_metal.log_set)
        g_metal.log_set(log_callback, user_data);
}
