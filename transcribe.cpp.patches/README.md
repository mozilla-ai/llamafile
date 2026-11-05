# transcribe.cpp Patches for Llamafile

This directory contains the patches and extra files that adapt
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) for
transcribefile: a cosmocc build, the cross-DSO ABI annotations that let
the host load GPU backends built from llama.cpp's ggml, and (for now) a
temporary alignment between the two ggml versions. Only transcribefile
links transcribe.cpp; none of these patches touch llama.cpp or the
llamafile binary.

## ⚠️ Temporary patches: ggml 0.28 alignment

**These are a stopgap. Remove them as soon as transcribe.cpp vendors
ggml >= 0.28.0:**

- `ggml_include_ggml.h.patch`, `ggml_src_ggml.c.patch` and
  `ggml_src_ggml-cpu_ggml-cpu.c.patch` (whole files): op numbering.
- The `GGML_OP_ARGMAX` hunk of `ggml_src_ggml-backend.cpp.patch`
  (marked `[llamafile] TEMPORARY`; the rest of that patch is the
  `GGML_CALL`/`free_struct` work and the upstream #25584 allocator fix,
  see the patch index): allocation-size check.

| tree | ggml |
|---|---|
| transcribe.cpp v0.4.1 (`ggml/UPSTREAM`: ggml-org/ggml `d7cb574`) | 0.26.0 |
| llama.cpp v0.8.0 | 0.28.0 |

transcribefile builds its graphs with transcribe.cpp's own ggml but
runs them on Metal through a dylib built from **llama.cpp's** ggml
sources (`llamafile/metal.c`, `LLAMAFILE_METAL_SOURCES`). The two trees
must agree on everything that crosses that boundary: the backend
interface structs, `struct ggml_tensor`, `struct ggml_cgraph`, the
op-params layout of each op, and the numbering of `enum ggml_op`,
`enum ggml_type`, and `enum ggml_unary_op`/`enum ggml_glu_op` (whose
values travel in `op_params`).

The host side (transcribe.cpp's scheduler and allocator) must also
expect what 0.28 backends do. Between 0.26 and 0.28 all of that is
identical except two things, one patch each:

1. **Op numbering.** ggml 0.28 inserted `GGML_OP_TOP_K_MASK` right
   before `GGML_OP_UNARY`. Without the patch every op from
   `GGML_OP_UNARY` on (unary activations, GLU, custom ops, ...) is off by
   one on the GPU side, and transcribefile segfaults inside the Metal
   backend's graph optimizer on the first graph. The patch inserts the
   enum value (plus its name/symbol table entries and the
   `GGML_OP_COUNT` asserts in `ggml.c`) so the numbering matches.
   transcribe.cpp never builds a `TOP_K_MASK` node, so its CPU backend
   never sees the new value; there is no implementation of the op on
   this side, and the CPU backend aborts if one ever reaches it.
2. **Allocation size of `ARGMAX`.** 0.28 backends may allocate more than
   `ggml_nbytes()` for `GGML_OP_ARGMAX`, and 0.28 added it to
   `ggml_op_alloc_size_may_expand()`. Without it, the
   `[TAG_ALLOC_SIZE_EXPAND]` assert in `ggml_backend_buft_get_alloc_size`
   aborts every model that runs greedy argmax on the GPU (moonshine,
   canary and qwen3-asr, for instance).

What these patches do **not** do is make transcribe.cpp's ggml *be*
0.28. They only cover the differences found so far; GPU support for
transcribe.cpp models is "best effort" until transcribe.cpp itself
moves to the ggml llama.cpp uses. Compare the Metal and CPU transcripts
when trying a new model family.

### When to remove it

When bumping transcribe.cpp, check `ggml/CMakeLists.txt`
(`GGML_VERSION_MINOR`) or `ggml/UPSTREAM` in the new release:

- **transcribe.cpp now vendors ggml >= 0.28.0** (so it defines
  `GGML_OP_TOP_K_MASK` itself): `git rm` the three whole-file patches,
  drop the `ARGMAX` hunk from `ggml_src_ggml-backend.cpp.patch`, drop
  this section, and delete the "temporary patch" paragraph in
  `transcribefile/README.md` and the matching comment in
  `transcribefile/main.cpp`.
- **Still older than llama.cpp's ggml**: keep the patches, and re-check
  that they still cover every difference. After `make setup` (both trees
  patched), from the repository root:

  ```sh
  ext() { awk -v b="$1" 'index($0, b) { f = 1 } f { print } f && /^ *};/ { exit }' "$2"; }
  for b in 'enum ggml_type {' 'enum ggml_op {' 'enum ggml_unary_op {' \
           'enum ggml_glu_op {' 'struct ggml_tensor {'; do
      [ -n "$(ext "$b" llama.cpp/ggml/include/ggml.h)" ] || echo "!!! '$b' not found"
      diff <(ext "$b" transcribe.cpp/ggml/include/ggml.h) \
           <(ext "$b" llama.cpp/ggml/include/ggml.h) || echo "^^^ $b"
  done
  for f in ggml-backend-impl.h ggml-impl.h; do
      diff transcribe.cpp/ggml/src/$f llama.cpp/ggml/src/$f || echo "^^^ $f"
  done
  m='/^bool ggml_op_alloc_size_may_expand(/,/^}/p'
  [ -n "$(sed -n "$m" llama.cpp/ggml/src/ggml-backend.cpp)" ] || echo "!!! may_expand not found"
  diff <(sed -n "$m" transcribe.cpp/ggml/src/ggml-backend.cpp) \
       <(sed -n "$m" llama.cpp/ggml/src/ggml-backend.cpp) || echo "^^^ may_expand"
  ```

  The only expected differences are the two `[llamafile] TEMPORARY`
  comment blocks (in `enum ggml_op` and in
  `ggml_op_alloc_size_may_expand()`); a `!!!` line means a pattern no
  longer matches and the check proved nothing. Then skim a full
  `diff` of the two `ggml/include/ggml.h`: today it shows only that
  comment, doc-comment changes and llama.cpp-only API
  (`ggml_prec_get_acc`, `ggml_prec_get_src`, `ggml_top_k_mask`).
  A changed op-params layout (the `ggml_set_op_params*()` calls in
  `ggml/src/ggml.c`) of an op transcribe.cpp uses has no mechanical
  check; comparing Metal and CPU transcripts is the backstop.

The same check applies when **llama.cpp** is bumped: a new op in
llama.cpp's ggml shifts the numbering again and needs to be mirrored
here (or Metal disabled in `transcribefile/main.cpp`) until
transcribe.cpp catches up. A value inserted into `ggml_unary_op` or
`ggml_glu_op` is worse: nothing crashes, Metal just dispatches the wrong
kernels and the transcripts come out wrong.

When testing Metal after any of these changes, make sure the dylib is
rebuilt: `metal.c` reuses `~/.transcribefile/v/<llamafile version>/ggml-metal.dylib`
whenever it exists, without checking it against the bundled sources.
Remove that directory, or run with `HOME` pointed at an empty directory.

## Directory Structure

```
transcribe.cpp.patches/
├── README.md          # This file
├── apply-patches.sh   # Applies everything below to the transcribe.cpp submodule
├── renames.sh         # File renames/moves (none yet)
├── llamafile-files/   # Files copied into transcribe.cpp
│   └── BUILD.mk       # cosmocc build of transcribe.cpp and its vendored ggml
└── patches/           # Patches for upstream sources
```

`apply-patches.sh` also generates `ggml/src/ggml-version.h` from
upstream's `ggml-version.h.in` (version from `ggml/CMakeLists.txt`,
commit from `ggml/UPSTREAM`), as CMake's `configure_file()` would. It is
a build artifact, not a patch: if `generate_patches.sh` copies it into
`llamafile-files/`, delete that copy.

## Applying Patches

`make setup` applies them. To do it by hand:

```sh
./transcribe.cpp.patches/apply-patches.sh
```

To reset the submodule to its clean state:

```sh
cd transcribe.cpp && git reset --hard && git clean -fdx
```

Regenerate the patches from in-place edits with `tools/generate_patches.sh`
(see `docs/commands/generate-patches.md`).

## Patch Index

### Windows/macOS ABI compatibility (`GGML_CALL`)

Same scheme as `llama.cpp.patches` (see its README for the background):
GPU backends are native shared libraries, while the host is built with
cosmocc, so every function pointer in the backend interface structs and
every implementation stored in one carries `GGML_CALL`
(`__ms_abi__` on x86-64 when `GGML_MULTIPLATFORM` is set, a no-op on
aarch64). As of transcribe.cpp v0.4.1 the upstream
`ggml-backend-impl.h`, `ggml-cpu.h` and `ggml-cpu.cpp` are identical to
llama.cpp v0.8.0's, and so are their patched versions.

| Patch | Description |
|-------|-------------|
| `ggml_include_ggml-backend.h.patch` | Defines `GGML_CALL`; adds it to the `get_proc_address` return typedefs |
| `ggml_include_ggml-cpu.h.patch` | Adds `GGML_CALL` to `ggml_backend_cpu_set_n_threads` and `ggml_backend_cpu_set_abort_callback` (returned via `get_proc_address`) |
| `ggml_src_ggml-backend-impl.h.patch` | Adds `GGML_CALL` to every function pointer of the five interface structs and to `ggml_backend_graph_optimize_params::add_alloc_dep`; adds the `free_struct` buffer callback (cross-module memory: a buffer allocated by a dylib is freed by it). Identical to the llama.cpp patch for the same file |
| `ggml_src_ggml-backend.cpp.patch` | `GGML_CALL` on the CPU buffer, buffer-type and multi-buffer callbacks; `free_struct` support in `ggml_backend_buffer_free`; turns the scheduler's `add_alloc_dep` lambda into a named `GGML_CALL` function (a lambda cannot carry the attribute; the Metal backend calls it from the dylib). Also carries the upstream #25584 fix (see "Upstream fixes" below) and the **temporary** `GGML_OP_ARGMAX` hunk described at the top |
| `ggml_src_ggml-backend-meta.cpp.patch` | `GGML_CALL` on all meta device/buffer/backend callbacks |
| `ggml_src_ggml-cpu_ggml-cpu.cpp.patch` | `GGML_CALL` on the CPU backend, device and registry callbacks and on the `get_proc_address`-returned functions |
| `ggml_src_ggml-cpu_repack.cpp.patch` | `GGML_CALL` on the repack buffer and buffer-type callbacks |

### Upstream fixes

An upstream fix we need before it lands in transcribe.cpp's ggml. Drop
it at the bump that first includes it (`check_patches.sh` flags it as
conflicting once upstream has the same change).

| Patch | Description |
|-------|-------------|
| `ggml_src_ggml-backend.cpp.patch` (part) | [PR #25584](https://github.com/ggml-org/llama.cpp/pull/25584), the same two hunks `llama.cpp.patches` carries in `ggml_backend_buft_alloc_buffer_n_default()` (see its README for the full story). When a buffer type has a `max_size`, a range made only of views that follows a tensor larger than `max_size` is never planned, and those views keep `data == NULL`; the fix allocates only parent tensors per range and initializes every view in one final pass. transcribe.cpp's ggml 0.26 has the identical code, and since the Metal backend leaves `alloc_buffer_n` unset, this host-side copy allocates for every backend. Not reachable today (CPU's `max_size` is `SIZE_MAX`, Metal's is the device's `maxBufferLength`); it becomes reachable once a backend with a small `max_size`, such as Vulkan (1 GiB), is wired up |

### Temporary ggml 0.28 alignment

See the section at the top. Remove all three (and the `ARGMAX` hunk of
`ggml_src_ggml-backend.cpp.patch`) when transcribe.cpp vendors
ggml >= 0.28.0.

| Patch | Description |
|-------|-------------|
| `ggml_include_ggml.h.patch` | Inserts `GGML_OP_TOP_K_MASK` before `GGML_OP_UNARY` so `enum ggml_op` numbers ops like llama.cpp's ggml 0.28 |
| `ggml_src_ggml.c.patch` | Adds the matching `GGML_OP_NAME`/`GGML_OP_SYMBOL` entries and bumps the two `GGML_OP_COUNT` asserts from 101 to 102 |
| `ggml_src_ggml-cpu_ggml-cpu.c.patch` | Sends `GGML_OP_TOP_K_MASK` to the existing `GGML_OP_COUNT` abort in `ggml_compute_forward` (its switch has no `default`, so the new value would otherwise be a silent no-op and a `-Wswitch` warning) |

### transcribefile entry point

| Patch | Description |
|-------|-------------|
| `examples_cli_main.cpp.patch` | Renames the CLI's `main()` to `transcribe_cli_main()` and drops the standalone `main()` under `-DTRANSCRIBEFILE`, so `transcribefile/main.cpp` owns the entry point (bundled `/zip/.args`, GPU loading) |

## Build Notes (`llamafile-files/BUILD.mk`)

- transcribe.cpp ships its own ggml (see `ggml/UPSTREAM`); it is built
  here CPU-only with `-DGGML_CPU_GENERIC` (one fat binary for x86-64 and
  aarch64), like llama.cpp's.
- The transcribe library is compiled as C++17, as upstream's CMake does
  (`CMAKE_CXX_STANDARD 17`), overriding the global `-std=gnu++23`: it
  relies on C++17 semantics such as `std::filesystem::path::u8string()`
  returning `std::string`.
- Source lists mirror `src/CMakeLists.txt`,
  `ggml/src/CMakeLists.txt` and `ggml/src/ggml-cpu/CMakeLists.txt`;
  the CLI sources (`examples/cli/*.cpp`) are listed in
  `transcribefile/BUILD.mk`.
