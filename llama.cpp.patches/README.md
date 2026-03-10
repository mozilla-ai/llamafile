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

### Cosmopolitan Libc Compatibility

These patches address compatibility issues when building with Cosmopolitan libc (cosmocc).

| Patch | Description |
|-------|-------------|
| `common_arg.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` (includes `linux/limits.h`) |
| `common_common.cpp.patch` | Adds platform-aware cache directory detection for Cosmopolitan (checks `LOCALAPPDATA`, `XDG_CACHE_HOME`, falls back to `~/.cache/`) |
| `common_download.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` |
| `common_ngram-mod.cpp.patch` | Adds missing `#include <algorithm>` for `std::fill` |

### Threading and Signal Handling

Cosmopolitan libc has specific behaviors with condition variables and signals that require workarounds.

| Patch | Description |
|-------|-------------|
| `common_log.cpp.patch` | Blocks `SIGINT`/`SIGTERM` on logger thread to prevent `EINTR` exceptions; uses `wait_for()` instead of `wait()` to work around XNU futex timeout bug (~72 minute expiry) |
| `tools_server_server-queue.cpp.patch` | Same threading fixes for server queue: signal masking and `wait_for()` timeouts |
| `vendor_cpp-httplib_httplib.cpp.patch` | Fixes httplib thread pool with `wait_for()` instead of `wait()` for XNU futex compatibility |

### Cross-Module Memory Management

When GPU backends (CUDA, Metal) are loaded as dynamic libraries, memory allocated by the DSO must be freed by the DSO's allocator, not the main executable's.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-backend-impl.h.patch` | Adds `free_struct` callback to `ggml_backend_buffer_i` interface for cross-module buffer cleanup |
| `ggml_src_ggml-backend.cpp.patch` | Implements `free_struct` callback support in `ggml_backend_buffer_free()` |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Adds `free_struct` implementation for CUDA buffers; disables BF16 with TinyBLAS |
| `ggml_src_ggml-metal_ggml-metal.cpp.patch` | Adds `free_struct` implementation for Metal buffers |

### TinyBLAS Integration

Llamafile uses TinyBLAS as a lightweight replacement for cuBLAS, enabling GPU support without CUDA SDK dependencies.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-cuda_vendors_cuda.h.patch` | Includes TinyBLAS headers instead of `cublas_v2.h` when `GGML_USE_TINYBLAS` is defined |
| `ggml_src_ggml-cuda_common.cuh.patch` | Disables BF16 MMA when using TinyBLAS (TinyBLAS would incorrectly interpret BF16 as FP16) |
| `ggml_src_ggml-cuda_solve_tri.cu.patch` | Disables cuBLAS TRSM path when using TinyBLAS (only affects Qwen3-Next models with large matrices) |

### Llamafile File Handling

These patches integrate llamafile's file handling APIs for loading models from bundled zip archives and `.llamafile` containers.

| Patch | Description |
|-------|-------------|
| `src_llama-mmap.h.patch` | Adds `has_premapped_content()`, `premapped_content()`, and `get_llamafile()` methods to `llama_file` class |
| `src_llama-mmap.cpp.patch` | Implements llamafile API integration for file I/O (`llamafile_open_gguf`, `llamafile_read`, etc.) and memory mapping with reference counting for bundled assets |
| `ggml_src_gguf.cpp.patch` | Adds `gguf_llamafile_reader` for reading GGUF files via llamafile API (supports `/zip/` paths, `foo.zip@weights.gguf` syntax, `.llamafile` containers) |

### Server Integration

| Patch | Description |
|-------|-------------|
| `tools_server_server.cpp.patch` | Refactors `main()` to `server_main()` for llamafile integration; adds Metal backend trigger, cosmo_args support, TUI mode handling, and proper exit for Metal async logging |

### Miscellaneous

| Patch | Description |
|-------|-------------|
| `common_chat.cpp.patch` | Fixes C++ type conversion: explicitly wraps `inputs.messages` in `std::optional<json>()` for Deepseek v3.1 template |
| `ggml_src_ggml-backend-reg.cpp.patch` | Suppresses debug log noise for non-existent backend search paths (irrelevant for llamafile's DSO loading approach) |

## Creating New Patches

Files in `llama.cpp` are usually modified in-place for development and testing.
Once they are ready to be committed, you can update all files in the `llama.cpp.patches` directory by running the following:

```sh
cd llama.cpp
../tools/generate-patches.sh --output-dir ../llama.cpp.patches
```

Patch filenames will automatically reflect the file path with underscores replacing slashes (e.g., `common_arg.cpp.patch` for `common/arg.cpp`).
