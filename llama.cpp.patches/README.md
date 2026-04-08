# llama.cpp Patches for Llamafile

This directory contains patches that adapt llama.cpp for use with Llamafile and Cosmopolitan libc. These patches enable llama.cpp to run as a portable, single-file executable across Windows, macOS, Linux, and BSD without installation.

## Directory Structure

```
llama.cpp.patches/
├── README.md              # This file
├── apply-patches.sh       # Script to apply all patches to llama.cpp submodule
├── renames.sh             # Script for file renames/moves (if any)
├── llamafile-files/       # Additional files to copy into llama.cpp
│   ├── BUILD.mk           # Makefile for building llama.cpp with cosmocc
│   ├── README.llamafile   # License and modification notes
│   └── common/
│       └── license.cpp    # Llama.cpp's license file (cmake creates this at build time)
└── patches/               # Patch files for upstream sources
```

## Applying Patches

To apply all patches to the llama.cpp submodule:

```sh
./llama.cpp.patches/apply-patches.sh
```

To reset the submodule to its clean state:

```sh
cd llama.cpp && git reset --hard && git clean -fdx
```

## Patch Index

### Windows/macOS ABI Compatibility (`GGML_CALL`)

GPU backends (CUDA, Vulkan, Metal) are compiled as shared libraries (`.dll`/`.so`/`.dylib`) using native compilers, but the llamafile host binary is built with Cosmopolitan libc which uses System V AMD64 ABI everywhere — including on Windows. When the host calls function pointers inside backend interface structs, the calling convention must match.

The `GGML_CALL` macro (defined as `__attribute__((__ms_abi__))` when `GGML_MULTIPLATFORM` is set) annotates all function pointers in the backend interface structs and their implementations, so the correct calling convention is used on every platform.

| Patch | Description |
|-------|-------------|
| `ggml_include_ggml-backend.h.patch` | Defines the `GGML_CALL` macro; adds it to the five `get_proc_address` return typedefs (`ggml_backend_split_buffer_type_t`, `ggml_backend_set_n_threads_t`, `ggml_backend_dev_get_extra_bufts_t`, `ggml_backend_set_abort_callback_t`, `ggml_backend_get_features_t`) |
| `ggml_include_ggml-cpu.h.patch` | Adds `GGML_CALL` to declarations of `ggml_backend_cpu_set_n_threads` and `ggml_backend_cpu_set_abort_callback` (returned via `get_proc_address`) |
| `ggml_include_ggml-cuda.h.patch` | Adds `GGML_CALL` to declarations of `ggml_backend_cuda_split_buffer_type`, `ggml_backend_cuda_register_host_buffer`, and `ggml_backend_cuda_unregister_host_buffer` |
| `ggml_src_ggml-backend-impl.h.patch` | Adds `GGML_CALL` to all 49+ function pointers across the five interface structs (`ggml_backend_buffer_type_i`, `ggml_backend_buffer_i`, `ggml_backend_i`, `ggml_backend_device_i`, `ggml_backend_reg_i`); also adds `free_struct` callback (see Cross-Module Memory below) |
| `ggml_src_ggml-backend.cpp.patch` | Adds `GGML_CALL` to CPU buffer, buffer type, and multi-buffer callback implementations; also adds `free_struct` support (see Cross-Module Memory below) |
| `ggml_src_ggml-cpu_ggml-cpu.cpp.patch` | Adds `GGML_CALL` to all CPU backend, device, and registry callback implementations, plus `get_proc_address`-returned functions (`set_n_threads`, `set_abort_callback`, `get_extra_buffers_type`, `get_features`) |
| `ggml_src_ggml-cpu_amx_amx.cpp.patch` | Adds `GGML_CALL` to all AMX buffer and buffer type callback implementations (10 functions) |
| `ggml_src_ggml-cpu_repack.cpp.patch` | Adds `GGML_CALL` to CPU repack buffer and buffer type callback implementations (5 functions) |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Adds `GGML_CALL` to all CUDA backend callback implementations (60+ functions); also adds `free_struct` and TinyBLAS BF16 guard (see below) |
| `ggml_src_ggml-metal_ggml-metal.cpp.patch` | Adds `GGML_CALL` to all Metal backend callback implementations (62 functions); also adds `free_struct` (see below) |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Adds `GGML_CALL` to all Vulkan backend callback implementations; also adds `free_struct` and a heap memory underflow fix (see below) |

### Cross-Module Memory Management

When GPU backends (CUDA, Vulkan, Metal) are loaded as dynamic libraries, memory allocated by the DSO must be freed by the DSO's allocator, not the main executable's.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-backend-impl.h.patch` | Adds `free_struct` callback to `ggml_backend_buffer_i` interface for cross-module buffer cleanup |
| `ggml_src_ggml-backend.cpp.patch` | Implements `free_struct` callback support in `ggml_backend_buffer_free()` — calls DSO's `free_struct` instead of `delete` when set |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Adds `free_struct` implementation for CUDA buffers (regular, split, and host); sets it on fallback CPU buffers allocated within the DSO |
| `ggml_src_ggml-metal_ggml-metal.cpp.patch` | Adds `free_struct` implementation for Metal shared and private buffers |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Adds `free_struct` implementation for Vulkan buffers and host buffer fallback path |

### Cosmopolitan Libc Compatibility

These patches address compatibility issues when building with Cosmopolitan libc (cosmocc).

| Patch | Description |
|-------|-------------|
| `common_arg.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` (includes `linux/limits.h`) |
| `common_common.cpp.patch` | Adds platform-aware cache directory detection for Cosmopolitan (checks `LOCALAPPDATA`, `XDG_CACHE_HOME`, falls back to `~/.cache/`); also adds mmproj model size estimation to GPU fit params so the fit algorithm reserves enough VRAM for multimodal projectors |
| `common_download.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` |
| `common_ngram-mod.cpp.patch` | Adds missing `#include <algorithm>` (needed for `std::min`/`std::max`) |

### Threading and Signal Handling

Cosmopolitan libc has specific behaviors with condition variables and signals that require workarounds.

| Patch | Description |
|-------|-------------|
| `common_log.cpp.patch` | Adds `#include <csignal>`; blocks `SIGINT`/`SIGTERM` on logger thread via `pthread_sigmask` to prevent `EINTR` exceptions; replaces `cv.wait()` with `wait_for(30s)` loop to work around XNU futex timeout bug (~72 minute expiry) |
| `tools_server_server-queue.cpp.patch` | Adds missing includes (`<cerrno>`, `<system_error>`, `<csignal>`); blocks `SIGINT`/`SIGTERM` on queue thread; replaces `wait()` with `wait_for()` loops in three locations (`wait_until_no_sleep`, main loop, `recv`) |
| `vendor_cpp-httplib_httplib.cpp.patch` | Fixes httplib thread pool with `wait_for()` instead of `wait()` for XNU futex compatibility |

### TinyBLAS Integration

Llamafile uses TinyBLAS as a lightweight replacement for cuBLAS, enabling GPU support without CUDA SDK dependencies.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-cuda_vendors_cuda.h.patch` | Includes TinyBLAS headers (`tinyblas.h`, `tinyblas-compat.h`) instead of `cublas_v2.h` when `GGML_USE_TINYBLAS` is defined; guards backward-compat `CUBLAS_*` defines so they don't conflict with TinyBLAS's own definitions |
| `ggml_src_ggml-cuda_common.cuh.patch` | Disables BF16 MMA when using TinyBLAS (TinyBLAS would incorrectly interpret BF16 as FP16) |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Disables BF16 in `ggml_cuda_op_mul_mat_cublas` when using TinyBLAS |

### Llamafile File Handling

These patches integrate llamafile's file handling APIs for loading models from bundled zip archives and `.llamafile` containers.

| Patch | Description |
|-------|-------------|
| `src_llama-mmap.h.patch` | Adds `has_premapped_content()`, `premapped_content()`, and `get_llamafile()` methods to `llama_file` class |
| `src_llama-mmap.cpp.patch` | Under `COSMOCC`, redirects file open/read/seek/tell/close through llamafile API (`llamafile_open_gguf`, `llamafile_read`, etc.); adds premapped content support to `llama_mmap` using llamafile reference counting (`llamafile_ref`/`llamafile_unref`); skips `munmap` for premapped content |
| `ggml_src_gguf.cpp.patch` | Adds `tell()`/`seek()` to `gguf_reader`; under `COSMOCC`, adds `gguf_llamafile_reader` that reads via llamafile API; templatizes `gguf_init_from_reader_impl` so both readers work; redirects `gguf_init_from_file` through `llamafile_open_gguf` (supports `/zip/` paths, `.llamafile` containers) |

### Server Integration

| Patch | Description |
|-------|-------------|
| `tools_server_server.cpp.patch` | Renames `main()` to `server_main()` with `on_ready`/`on_shutdown_available` callbacks for combined TUI+server mode; adds Metal/GPU backend trigger before `common_init()`; adds Cosmopolitan-specific standalone `main()` with `cosmo_args`, verbose flag handling, and GPU pre-initialization; handles `LLAMAFILE_TUI` exit to avoid Metal cleanup crashes |

### Bug Fixes

| Patch | Description |
|-------|-------------|
| `common_chat.cpp.patch` | Fixes C++ type conversion: explicitly wraps `inputs.messages` in `std::optional<json>()` for Deepseek v3.1 template |
| `ggml_src_ggml-backend-reg.cpp.patch` | Suppresses debug log noise for non-existent backend search paths (irrelevant for llamafile's DSO loading approach) |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Fixes unsigned integer underflow in `ggml_backend_vk_get_device_memory` where Vulkan's `heapUsage` can exceed `heapBudget` (clamps to zero instead of wrapping) |

## Creating New Patches

Files in `llama.cpp` are usually modified in-place for development and testing.
Once they are ready to be committed, you can update all files in the `llama.cpp.patches` directory by running the following:

```sh
cd llama.cpp
../tools/generate-patches.sh --output-dir ../llama.cpp.patches
```

Patch filenames will automatically reflect the file path with underscores replacing slashes (e.g., `common_arg.cpp.patch` for `common/arg.cpp`).
