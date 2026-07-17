# llama.cpp Patches for Llamafile

This directory contains patches that adapt llama.cpp for use with Llamafile and Cosmopolitan libc. These patches enable llama.cpp to run as a portable, single-file executable across Windows, macOS, Linux, and BSD without installation.

## Directory Structure

```
llama.cpp.patches/
├── README.md              # This file
├── apply-patches.sh       # Script to apply all patches to llama.cpp submodule
├── fetch-ui-assets.sh     # Downloads + validates the prebuilt web UI (see Server Integration)
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
| `ggml_include_ggml-cuda.h.patch` | Adds `GGML_CALL` to the declaration of `ggml_backend_cuda_register_host_buffer` (upstream removed the row-split multi-GPU buffer in b10052, so the former `ggml_backend_cuda_split_buffer_type` annotation is gone) |
| `ggml_src_ggml-backend-impl.h.patch` | Adds `GGML_CALL` to all 49+ function pointers across the five interface structs (`ggml_backend_buffer_type_i`, `ggml_backend_buffer_i`, `ggml_backend_i`, `ggml_backend_device_i`, `ggml_backend_reg_i`); also adds `free_struct` callback (see Cross-Module Memory below) |
| `ggml_src_ggml-backend.cpp.patch` | Adds `GGML_CALL` to CPU buffer, buffer type, and multi-buffer callback implementations; also adds `free_struct` support (see Cross-Module Memory below) |
| `ggml_src_ggml-cpu_ggml-cpu.cpp.patch` | Adds `GGML_CALL` to all CPU backend, device, and registry callback implementations, plus `get_proc_address`-returned functions (`set_n_threads`, `set_abort_callback`, `get_extra_buffers_type`, `get_features`) |
| `ggml_src_ggml-cpu_amx_amx.cpp.patch` | Adds `GGML_CALL` to all AMX buffer and buffer type callback implementations (10 functions) |
| `ggml_src_ggml-cpu_repack.cpp.patch` | Adds `GGML_CALL` to CPU repack buffer and buffer type callback implementations (5 functions) |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Adds `GGML_CALL` to all CUDA backend callback implementations (60+ functions); also adds `free_struct` and TinyBLAS BF16 guard (see below) |
| `ggml_src_ggml-metal_ggml-metal.cpp.patch` | Adds `GGML_CALL` to all Metal backend callback implementations (62 functions); also adds `free_struct` (see below) |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Adds `GGML_CALL` to all Vulkan backend callback implementations; also adds `free_struct` and a heap memory underflow fix (see below) |
| `ggml_src_ggml-backend-meta.cpp.patch` | Adds `GGML_CALL` to all meta-device, meta-buffer-type, meta-buffer, and meta-backend callback implementations (the meta backend aggregates several simple backends behind one interface, so its callbacks are reached through the same function-pointer structs) |

### Cross-Module Memory Management

When GPU backends (CUDA, Vulkan, Metal) are loaded as dynamic libraries, memory allocated by the DSO must be freed by the DSO's allocator, not the main executable's.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-backend-impl.h.patch` | Adds `free_struct` callback to `ggml_backend_buffer_i` interface for cross-module buffer cleanup |
| `ggml_src_ggml-backend.cpp.patch` | Implements `free_struct` callback support in `ggml_backend_buffer_free()` — calls DSO's `free_struct` instead of `delete` when set |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Adds `free_struct` implementation for CUDA buffers (regular and host; upstream removed the split buffer in b10052); sets it on fallback CPU buffers allocated within the DSO |
| `ggml_src_ggml-metal_ggml-metal.cpp.patch` | Adds `free_struct` implementation for Metal shared and private buffers |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Adds `free_struct` implementation for Vulkan buffers and host buffer fallback path |

### Cosmopolitan Libc Compatibility

These patches address compatibility issues when building with Cosmopolitan libc (cosmocc).

| Patch | Description |
|-------|-------------|
| `common_arg.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` (includes `linux/limits.h`) |
| `common_common.cpp.patch` | Adds platform-aware cache directory detection for Cosmopolitan (checks `LOCALAPPDATA`, `XDG_CACHE_HOME`, falls back to `~/.cache/`); also adds mmproj model size estimation to GPU fit params so the fit algorithm reserves enough VRAM for multimodal projectors |
| `common_download.cpp.patch` | Adds `COSMOCC` platform detection for `PATH_MAX` |

### Threading and Signal Handling

Cosmopolitan libc has specific behaviors with condition variables and signals that require workarounds.

| Patch | Description |
|-------|-------------|
| `common_log.cpp.patch` | Adds `#include <csignal>`; blocks `SIGINT`/`SIGTERM` on logger thread via `pthread_sigmask` to prevent `EINTR` exceptions; replaces `cv.wait()` with `wait_for(30s)` loop to work around XNU futex timeout bug (~72 minute expiry) |
| `tools_server_server-models.cpp.patch` | Adds `#include <csignal>`; blocks signals on the stopping thread via `pthread_sigmask`; replaces untimed `cv.wait()` with `wait_for(30s)` loops on every model-lifecycle wait (`unload_lru`, the reload-drain wait, `stopping_thread`, the `is_reloading` guard in `load`, and the generic `wait()` predicate helper) to work around the XNU futex timeout bug |
| `tools_server_server-queue.cpp.patch` | Adds missing includes (`<cerrno>`, `<system_error>`, `<csignal>`); blocks `SIGINT`/`SIGTERM` on queue thread; replaces `wait()` with `wait_for()` loops in three locations (`wait_until_no_sleep`, main loop, `recv`) |
| `vendor_cpp-httplib_httplib.cpp.patch` | Fixes httplib thread pool with `wait_for()` instead of `wait()` for XNU futex compatibility; also see HTTPS / TLS Support below |

### HTTPS / TLS Support

Upstream llama.cpp gets TLS from cpp-httplib's OpenSSL backend
(`CPPHTTPLIB_OPENSSL_SUPPORT`, satisfied by system OpenSSL or vendored
BoringSSL/LibreSSL at cmake time). None of those is available in the
cosmocc make build, so llamafile instead enables cpp-httplib's **Mbed TLS
backend** (`CPPHTTPLIB_MBEDTLS_SUPPORT`) against the mbedtls fork already
vendored in `third_party/mbedtls` — the same TLS stack llamafile <= 0.9.3
used. `third_party/mbedtls/include/` maps the canonical `<mbedtls/*.h>`
include paths onto the fork's headers, and `BUILD.mk` sets the macro on
every object that can reach `httplib.h` (the macro changes httplib class
layouts, so all TUs must agree) and links `mbedtls.a` into `llama-server`.
This enables HTTPS model downloads (`-hf`, `--model-url`), https clients
in server-models, and TLS serving via `--ssl-cert-file`/`--ssl-key-file`.

| Patch | Description |
|-------|-------------|
| `common_http.h.patch` | `#ifndef CPPHTTPLIB_OPENSSL_SUPPORT` -> `#ifndef CPPHTTPLIB_SSL_ENABLED` for the "HTTPS is not supported" guard, so any cpp-httplib TLS backend counts (candidate for upstreaming) |
| `tools_server_server-http.cpp.patch` | Same macro fix for the `httplib::SSLServer` (`--ssl-cert-file`/`--ssl-key-file`) guard (candidate for upstreaming) |
| `tools_server_server-models.cpp.patch` | Same macro fix for the direct `httplib::SSLClient` construction in `server_http_proxy` (candidate for upstreaming) |
| `vendor_cpp-httplib_httplib.cpp.patch` | Under `__COSMOPOLITAN__`: appends `/zip/third_party/mbedtls/sslroot` to `system_ca_dirs()` as the trust-store fallback (essential on Windows hosts, where the `_WIN32` cert-store branches are not compiled into an APE), and `__static_yoink("ssl_root_support")` so the Mozilla root PEMs bundled by `third_party/mbedtls/BUILD.mk` are pulled into the executable's zip |

### TinyBLAS Integration

Llamafile uses TinyBLAS as a lightweight replacement for cuBLAS, enabling GPU support without CUDA SDK dependencies.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-cuda_vendors_cuda.h.patch` | Includes TinyBLAS headers (`tinyblas.h`, `tinyblas-compat.h`) instead of `cublas_v2.h` when `GGML_USE_TINYBLAS` is defined; guards backward-compat `CUBLAS_*` defines so they don't conflict with TinyBLAS's own definitions |
| `ggml_src_ggml-cuda_common.cuh.patch` | Disables BF16 MMA when using TinyBLAS (TinyBLAS would incorrectly interpret BF16 as FP16) |
| `ggml_src_ggml-cuda_ggml-cuda.cu.patch` | Disables BF16 in `ggml_cuda_op_mul_mat_cublas` when using TinyBLAS |

### Optional IQ-Quant Exclusion (CUDA)

The IQ ("importance") quantization formats (`IQ1_S`, `IQ2_XXS`/`XS`/`S`, `IQ3_S`/`XXS`, `IQ4_NL`/`XS`) pull in a large amount of CUDA template instantiation that inflates compile time and binary size. These patches gate every IQ code path behind `#ifndef GGML_CUDA_NO_IQ_QUANTS`, so a build can compile them out by defining `GGML_CUDA_NO_IQ_QUANTS`. When the macro is undefined (the default), behavior is unchanged.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-cuda_convert.cu.patch` | Guards IQ dequantization cases in `ggml_get_to_fp16_cuda` and `ggml_get_to_fp32_cuda` |
| `ggml_src_ggml-cuda_cpy.cu.patch` | Guards the `f32 → IQ4_NL` copy helper and its dispatch case |
| `ggml_src_ggml-cuda_mmq.cu.patch` | Guards IQ cases in `ggml_cuda_mul_mat_q_switch_type` and in the `ggml_cuda_should_use_mmq` support/heuristic switches |
| `ggml_src_ggml-cuda_mmq.cuh.patch` | Guards the `extern DECL_MMQ_CASE(...)` declarations for IQ types |
| `ggml_src_ggml-cuda_mmvq.cu.patch` | Guards IQ cases in `get_vec_dot_q_cuda` and `get_vdr_mmvq` |

### CPU Performance Optimizations (llamafile #975)

These patches restore llamafile's optimized CPU kernels (TinyBLAS matmul, AVX-512 flash-attention helpers) on top of upstream's CPU backend, and tune CPU-only defaults. The hooks call into symbols exported from `llamafile/sgemm.cpp` and are compiled only when `GGML_USE_LLAMAFILE` is defined.

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-cpu_ggml-cpu.c.patch` | Routes MoE matmul (`ggml_compute_forward_mul_mat_id`) through `llamafile_mixmul` / `llamafile_mixmul_iqk`, mirroring the dense-matmul `llamafile_sgemm` hook; reserves work-buffer space for the MoE kernel in `ggml_graph_plan` via `llamafile_mixmul_needs` |
| `ggml_src_ggml-cpu_ops.cpp.patch` | Routes flash-attention inner loops through llamafile's AVX-512 helpers (`llamafile_fa_vec_dot_f16`, `llamafile_fa_fp16_to_fp32_row`, `llamafile_fa_simd_gemm`) in both the one-chunk and tiled FA paths; also accumulates VKQ in f32 on CPUs lacking native f16 FMA (avoiding costly f16↔f32 round-trips per KV step) |
| `src_llama-context.cpp.patch` | Defaults `-fa auto` to **off** on CPU-only setups (no GPU devices), since the CPU flash-attention path is slower than the non-FA path on x86; users can still force `-fa on` for memory savings on long contexts |

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
| `tools_server_server.cpp.patch` | Renames upstream's `llama_server()` to `server_main()` and adds `on_ready`/`on_shutdown_available` callbacks for combined TUI+server mode; adds Metal/GPU backend trigger before `common_init()`; installs the sandbox (`llamafile_sandbox_server()`, issue #930) — the mechanism lives in `llamafile/sandbox.c`; the patch fills a `llamafile_sandbox_spec` with the on-disk paths the loader opens after the lock (model, mmproj, media dir, LoRA, draft model, control vectors, public path as read; slot-save and prompt cache as read-write), detects outbound-needing features (`--rpc` via argv/env, MCP proxy, server tools) to relax `anet`→`inet`, passes `FLAG_confine_reads` for opt-in unveil(), quiesces the log worker (`common_log_pause`/`resume`, so no thread escapes the per-thread filter) around the call, before the HTTP listener spawns and before model load, and skips it in combined/GPU modes; adds Cosmopolitan-specific standalone `main()` with `cosmo_args`, verbose flag handling, `--unsecure` consumption (`llamafile_consume_flag`), and GPU pre-initialization; handles `LLAMAFILE_TUI` exit to avoid Metal cleanup crashes |

The web UI moved upstream from prebuilt `tools/server/public/*` assets to
a Svelte/PWA project under `tools/ui/`, embedded at CMake time via
`tools/ui/embed.cpp`. cosmocc has no JS toolchain, so `fetch-ui-assets.sh`
(run by `apply-patches.sh` / `make setup`) downloads the prebuilt site
**tarball** `dist.tar.gz` (plus its `.sha256`) from the `ggml-org/llama-ui`
Hugging Face bucket — picking the newest `bNNNN` tag `<=` our pinned build —
verifies it, and extracts the whole static site into
`llama.cpp/tools/ui/dist/`. The modern site is no longer four flat files
(`bundle.css`/`bundle.js`/`index.html`/`loading.html`) but a hashed tree
(`_app/immutable/bundle.HASH.js`, a service worker, manifest, icons, splash
screens). To keep the embedded payload small, the script also builds a
`dist/_gzip/` mirror with every file gzip-compressed under its original name.

At build time, `llama.cpp/BUILD.mk` compiles `tools/ui/embed.cpp` with
`cosmoc++` (its APE output runs on the host) and runs it against the `dist/`
**directory** (the new `embed <out_cpp> <out_h> [<asset_dir>]` interface):
`embed.cpp` recursively bakes every file — auto-detecting `dist/_gzip/` and
emitting gzip-encoded assets keyed by relative path — into
`o/$(MODE)/llama.cpp/tools/ui/ui.{cpp,h}`, which is compiled like any other
C++ source and linked into `llama-server` and `llamafile`. Upstream's
`server-http.cpp` (unpatched) registers a route per embedded asset
(`index.html` at `/`) and serves gzip assets with `Content-Encoding: gzip`.
If the download fails (offline, version not yet on HF) the script leaves
`dist/` empty — `embed.cpp` then emits a no-op `llama_ui_find_asset` and the
`LLAMA_UI_HAS_ASSETS` guard keeps the UI routes unregistered, so the REST
API still works.

Note that `embed.cpp` only takes that graceful no-asset path when `dist/` is
*empty*: once the directory is non-empty it enforces a required-asset set
(its `required_check[]` table — `index.html`, `loading.html`,
`manifest.webmanifest`, `sw.js`, `build.json`, `version.json`, and the hashed
`bundle*.js`/`bundle*.css`/`workbox*.js`) and returns a hard error if any is
missing. So `fetch-ui-assets.sh` validates the *same* set after extracting
(`ui_missing_assets`); if a downloaded tarball is partial or has drifted, it
clears `dist/` and falls back to a UI-less build rather than letting the embed
step abort the whole build. Because `embed.cpp` is an upstream file, that list
can change on a llama.cpp bump — when it does, the asset list in
`ui_missing_assets` must be updated to match.

### Bug Fixes

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-backend-reg.cpp.patch` | Suppresses debug log noise for non-existent backend search paths (irrelevant for llamafile's DSO loading approach) |
| `ggml_src_ggml-vulkan_ggml-vulkan.cpp.patch` | Fixes unsigned integer underflow in `ggml_backend_vk_get_device_memory` where Vulkan's `heapUsage` can exceed `heapBudget` (clamps to zero instead of wrapping) |
| `src_models_t5.cpp.patch` | Forward-declares the `graph<false>`/`graph<true>` explicit specializations before `build_arch_graph` so clang's `-std=gnu++23` doesn't reject them as specializations after implicit instantiation |
| `src_models_eagle3.cpp.patch` | Moves `build_arch_graph` to the end of the file, after the `graph<true>`/`graph<false>` constructor specializations, so clang's `-std=gnu++23` doesn't reject them as explicit specializations appearing after the `make_unique<graph<...>>` implicit instantiation point |
| `src_models_dflash.cpp.patch` | Same fix as `eagle3` for the DFlash model (new in b10052): moves `build_arch_graph` to the end of the file, after the `graph<true>`/`graph<false>` specializations, so clang's `-std=gnu++23` doesn't reject them as explicit specializations after the `make_unique<graph<...>>` implicit instantiation point |

## Creating New Patches

Files in `llama.cpp` are usually modified in-place for development and testing.
Once they are ready to be committed, you can update all files in the `llama.cpp.patches` directory by running the following:

```sh
cd llama.cpp
../tools/generate-patches.sh --output-dir ../llama.cpp.patches
```

Patch filenames will automatically reflect the file path with underscores replacing slashes (e.g., `common_arg.cpp.patch` for `common/arg.cpp`).
