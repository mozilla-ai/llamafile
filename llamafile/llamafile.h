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

#ifndef LLAMAFILE_H_
#define LLAMAFILE_H_
#include <stdbool.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FLAGS - Global configuration variables (defined in llamafile.c)
// =============================================================================

extern bool FLAG_log_disable;   // Disables logging (chatbot_comm.cpp)
extern bool FLAG_nocompile;     // Disables GPU library compilation (metal.c)
extern bool FLAG_ascii;         // Uses ASCII art for logo (chatbot_logo.cpp)
extern bool FLAG_nologo;        // Suppresses logo display (chatbot_main.cpp)
extern bool FLAG_nothink;       // Filters thinking/reasoning content (chatbot_cli.cpp)
extern bool FLAG_precise;       // Forces precise math in tinyblas (tinyblas_cpu.h)
extern bool FLAG_recompile;     // Forces GPU library recompilation (metal.c)
extern int FLAG_gpu;            // GPU backend selection (llamafile.c, metal.c, cuda.c)
extern int FLAG_verbose;        // Verbose output (chatbot_main.cpp, metal.c, cuda.c)

// =============================================================================
// File I/O - GGUF file handling with zip support
// Defined in llamafile.c, used internally for model loading
// UNUSED externally: These are defined but not called from outside llamafile.c
// =============================================================================

struct llamafile;
struct llamafile *llamafile_open_gguf(const char *, const char *);  // UNUSED externally
void llamafile_close(struct llamafile *);                           // UNUSED externally
long llamafile_read(struct llamafile *, void *, size_t);            // UNUSED externally
long llamafile_write(struct llamafile *, const void *, size_t);     // UNUSED externally
bool llamafile_seek(struct llamafile *, size_t, int);               // UNUSED externally
void *llamafile_content(struct llamafile *);                        // UNUSED externally
size_t llamafile_tell(struct llamafile *);                          // UNUSED externally
size_t llamafile_size(struct llamafile *);                          // UNUSED externally
size_t llamafile_position(struct llamafile *);                      // UNUSED externally
bool llamafile_eof(struct llamafile *file);                         // UNUSED externally
FILE *llamafile_fp(struct llamafile *);                             // UNUSED externally
void llamafile_ref(struct llamafile *);                             // UNUSED externally
void llamafile_unref(struct llamafile *);                           // UNUSED externally

// =============================================================================
// Utility functions
// =============================================================================

// NOT DEFINED: Declaration only, no implementation in llamafile_new/
void llamafile_govern(void);                              // NOT DEFINED
void llamafile_check_cpu(void);                           // NOT DEFINED
void llamafile_help(const char *);                        // NOT DEFINED
void llamafile_log_command(char *[]);                     // NOT DEFINED
const char *llamafile_get_tmp_dir(void);                  // NOT DEFINED
void llamafile_schlep(const void *, size_t);              // NOT DEFINED
void llamafile_launch_browser(const char *);              // NOT DEFINED
void llamafile_get_flags(int, char **);                   // NOT DEFINED
char *llamafile_get_prompt(void);                         // NOT DEFINED

// USED: Defined in llamafile.c
bool llamafile_has(char **, const char *);
void llamafile_get_app_dir(char *, size_t);
bool llamafile_extract(const char *, const char *);
int llamafile_is_file_newer_than(const char *, const char *);

// Common utilities for GPU backend loaders (defined in llamafile.c)
const char *llamafile_get_dso_extension(void);
bool llamafile_file_exists(const char *);
int llamafile_makedirs(const char *, int);

// Link function type for TryLoadPrebuiltDso
typedef bool (*llamafile_link_dso_fn)(const char *dso_path);

// Try to load a prebuilt DSO from /zip/, app dir, or home dir
// Returns true if successfully loaded via link_fn
bool llamafile_try_load_prebuilt_dso(const char *name, const char *backend_name,
                                     llamafile_link_dso_fn link_fn);

// =============================================================================
// GPU detection and configuration
// =============================================================================

#define LLAMAFILE_GPU_ERROR -2
#define LLAMAFILE_GPU_DISABLE -1
#define LLAMAFILE_GPU_AUTO 0
#define LLAMAFILE_GPU_AMD 1
#define LLAMAFILE_GPU_APPLE 2
#define LLAMAFILE_GPU_NVIDIA 4
#define LLAMAFILE_GPU_VULKAN 8

bool llamafile_has_gpu(void);             // Defined in llamafile.c
bool llamafile_has_metal(void);           // Defined in metal.c (dynamic loader)
bool llamafile_has_cuda(void);            // Defined in cuda.c (dynamic loader)
bool llamafile_has_amd_gpu(void);         // Defined in cuda.c (dynamic loader)
bool llamafile_has_vulkan(void);          // Defined in vulkan.c (dynamic loader)
int llamafile_gpu_parse(const char *);    // Defined in llamafile.c
const char *llamafile_describe_gpu(void); // Defined in llamafile.c
void llamafile_early_gpu_init(char **);   // Defined in llamafile.c

// Log callback type for Metal backend (matches ggml_log_callback)
typedef void (*llamafile_log_callback)(int level, const char *text, void *user_data);

// No-op log callback to disable logging (defined in llamafile.c)
void llamafile_log_callback_null(int level, const char *text, void *user_data);

// Set logging callback for Metal dylib (defined in metal.c)
// Pass a no-op callback to disable logging
void llamafile_metal_log_set(llamafile_log_callback log_callback, void *user_data);

// Set logging callback for CUDA/ROCm dylib (defined in cuda.c)
// Pass a no-op callback to disable logging
void llamafile_cuda_log_set(llamafile_log_callback log_callback, void *user_data);

// Set logging callback for Vulkan dylib (defined in vulkan.c)
// Pass a no-op callback to disable logging
void llamafile_vulkan_log_set(llamafile_log_callback log_callback, void *user_data);

#ifdef __cplusplus
}
#endif
#endif /* LLAMAFILE_H_ */
