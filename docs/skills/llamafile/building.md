# Building Llamafile

Complete guide to the llamafile build system and toolchain.

## Prerequisites

### Cosmopolitan Toolchain

Llamafile uses Cosmopolitan C/C++ compiler (cosmocc) to create Actually Portable Executables (APE). The toolchain 
is downloaded automatically when `make setup` is called but can be fetched manually too with:

```sh
build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
```

Arguments:
1. Destination directory (`.cosmocc/4.0.2`)
2. Version (`4.0.2`)
3. SHA256 checksum for verification

### Git Submodules

Three main dependencies are git submodules:
- llama.cpp - LLM inference engine
- whisper.cpp - Speech-to-text engine
- stable-diffusion.cpp - Image generation engine

## Initial Setup

Before first build, initialize and configure dependencies:

```sh
make setup
```

This command:
1. Initializes git submodules (clones if needed)
2. Applies llamafile-specific patches from `<submodule>.patches/` directories
3. Modifies submodules in-place for llamafile integration

**Important:** Run `make setup` after:
- Fresh clone
- Updating submodules
- Pulling changes that modify patch files

## Build Commands

### Full Build

```sh
.cosmocc/4.0.2/bin/make -j $(nproc)  # or: llamafile:build
```

The `-j $(nproc)` flag enables parallel compilation (adjust based on CPU cores).
Adapt `nproc` to the OS where you are building, (e.g. `sysctl -n hw.physicalcpu` on mac)

**Critical:** Always use `.cosmocc/4.0.2/bin/make`, not system make. The cosmocc toolchain includes its own make with Cosmopolitan-specific behavior. The only exceptions are `make setup` and `make reset-repo`, which use bare `make` — `setup` bootstraps cosmocc on a fresh clone, and both are exempt from the version check.

### Clean Build

Remove build outputs:

```sh
.cosmocc/4.0.2/bin/make clean  # or: llamafile:clean
```

This removes the `o/` directory containing all compiled objects and binaries.

### Install compiled binaries

```sh
sudo .cosmocc/4.0.2/bin/make install PREFIX=/usr/local
```

Installs binaries and man pages.

## Build System Architecture

### Directory Structure

```
build/
├── config.mk          # Compiler, flags, toolchain version
├── rules.mk           # Generic build patterns
├── download-cosmocc.sh    # Toolchain download script
├── llamafile-convert      # Model conversion script
└── llamafile-upgrade-engine   # Engine update script
```

### Configuration (build/config.mk)

Defines:
- Compiler paths (CC, CXX pointing to cosmocc)
- Compiler flags (optimization, warnings)
- Toolchain version
- Platform-specific settings

### Build Rules (build/rules.mk)

Generic patterns for:
- `.c` → `.o` compilation
- `.a` archive creation
- `.zip.o` asset bundling (embed files into executables)

### BUILD.mk Files

Each major component has a BUILD.mk file defining:
- Source files to compile
- Dependencies
- Build targets
- Test targets

The top-level Makefile includes all BUILD.mk files to orchestrate the build.

## Build Outputs

All outputs go to `o/$(MODE)/`:

```
o/
└── $(MODE)/
    ├── llamafile/
    │   ├── llamafile          # Main executable
    │   ├── *.o                # Object files
    │   └── *.a                # Static libraries
    ├── llama.cpp/
    ├── whisper.cpp/
    ├── stable-diffusion.cpp/
    └── third_party/
        └── zipalign/
            └── zipalign       # Asset bundling tool
```

## Multi-Architecture Support

The build system creates universal binaries supporting:
- x86_64 (Intel/AMD)
- aarch64 (ARM64)

Both architectures are compiled simultaneously and combined into single APE binaries.

### Runtime Dispatch

Binaries detect CPU features at runtime and select optimal code paths:
- AVX, AVX2, AVX-512 (x86_64)
- ARM NEON (aarch64)

## Asset Bundling

Files can be embedded into executables using the `.zip.o` pattern:

```makefile
o/$(MODE)/path/to/asset.zip.o: path/to/asset
```

The `zipalign` tool handles bundling. Embedded assets are accessible at runtime through the Cosmopolitan virtual filesystem.

## GPU Support

GPU acceleration (CUDA/ROCm/Vulkan) uses dynamic loading:
- Shared libraries (.so/.dll) are not linked at compile time
- Libraries are loaded at runtime if available
- Can be bundled into executables using zipalign

### Building the GPU dylibs

The host `make` does **not** build the GPU backends. They are separate,
prebuilt shared libraries produced by per-backend scripts in `llamafile/`:

- `llamafile/cuda.sh`   → `ggml-cuda.so`   (needs `nvcc`; `CUDA_PATH`, default `/usr/local/cuda`)
- `llamafile/rocm.sh`   → `ggml-rocm.so`   (needs `hipcc`; `ROCM_PATH`, default `/opt/rocm`)
- `llamafile/vulkan.sh` → `ggml-vulkan.so` (needs `glslc` + SPIR-V headers + libvulkan)
- `.bat` / `*_parallel.bat` variants build the Windows DLLs.

`make_dylibs.sh` (if present) just wraps these for a release (writes to a
versioned `gpulibs/<VERSION>/`, passes `--clean`, and `--minimize-size` for
CUDA). **Run the scripts one at a time**, always with `--clean` — a stale
`~/.cache/llamafile-{cuda,rocm,vulkan}-build` is the usual "first build fails"
cause (leftover objects that reference symbols upstream removed, e.g. the b10052
CUDA split-buffer removal). `cuda.sh`/`rocm.sh` share `build-functions.sh`;
`vulkan.sh` is standalone.

### Verifying a GPU build actually succeeded (do not trust "Successfully built")

The parallel compile launches each `nvcc`/`hipcc`/`glslc` in the background.
`compile_gpu_sources_parallel` (cuda/rocm) now reaps each job and fails loudly,
but `vulkan.sh`'s shader loops do **not** yet check per-job status — a single
failed source can still be dropped while the link proceeds and prints
"Successfully built" with a silently incomplete `.so`. So always confirm, don't
assume:

```bash
llamafile/cuda.sh --clean --output /tmp/ggml-cuda.so --minimize-size 2>&1 | tee /tmp/b.log
grep -c 'error:' /tmp/b.log          # must be 0
grep 'Linking'   /tmp/b.log          # object count should match the source count
```

If a source fails to compile, fix the root cause (a new upstream cuBLAS call
missing from TinyBLAS is a recurring one — see `update_llamacpp.md`), never let
the "success" line stand in for a green build.

### Vulkan speed paths depend on the glslc/SDK version

`vulkan.sh` probes the build host's `glslc` for shader extensions
(coopmat, coopmat2, integer-dot, bf16, fp4/fp8) — mirroring upstream's
`ggml-vulkan/CMakeLists.txt` `test_shader_extension_support`. Each probe that
fails drops that acceleration pipeline from the shipped dylib (correct output,
just slower on hardware that supports it). Distro `shaderc` (Ubuntu, SteamOS)
is typically **too old** — it doesn't know `GL_EXT_integer_dot_product`,
`GL_EXT_bfloat16`, `GL_NV_cooperative_matrix2`, and can't target
`--target-env=vulkan1.4`. For a full-featured release Vulkan dylib, build on a
host with a recent **LunarG Vulkan SDK** (1.4.x); after installing, `source`
its `setup-env.sh` so `vulkan.sh` picks up the SDK's `glslc`, then re-run the
build and confirm the probes now print "supported". Keep `vulkan.sh`'s probe
list in sync with the CMakeLists (it must probe all seven feature tests).

### CUDA `--minimize-size` drops IQ-quant kernels (CPU fallback)

The release CUDA dylib is built with `--minimize-size`, which (among the size
options) passes `-DGGML_CUDA_NO_IQ_QUANTS`. That compiles out every IQ-quant
CUDA path — the MMQ/MMVQ matmul kernels and the fp16/fp32 dequant converters
in `convert.cu`, `mmq.cu`, `mmvq.cu`, and the `f32→iq4_nl` copy in `cpy.cu`.
The payoff is a much smaller `.so`; the cost is that it cannot run IQ-quantized
tensors on the GPU.

`ggml_backend_cuda_device_supports_op` is guarded by the same macro: the IQ
cases in the `MUL_MAT`/`MUL_MAT_ID` and `CPY` switches sit inside
`#ifndef GGML_CUDA_NO_IQ_QUANTS`. So a minimized build honestly reports those
ops unsupported and the scheduler runs them on the CPU backend — correct
output, just slower for the IQ layers. **Without that guard the op is
dispatched to a kernel that was never compiled and aborts at
`GGML_ASSERT(convert_func != nullptr)` (`ggml-cuda.cu`)** — this is exactly
what `backend_ops_test` hit on `iq2_xxs` MUL_MAT. If you add or bump IQ
handling, keep the `supports_op` cases and their kernels behind the same macro
or the mismatch reappears. (ROCm shares `ggml-cuda.cu`, but its build does not
define the macro, so IQ stays enabled there.)

For a CUDA dylib that runs IQ quants on the GPU, build without the flag — drop
`--minimize-size` (or use its other pieces like `--minimal-archs`/`--strip`
without `--no-iq-quants`) — at the cost of a larger library. The ROCm, Vulkan,
and Metal builds are not size-minimized, so none of them strips IQ the way the
CUDA build does: ROCm shares this file with the macro undefined, so it keeps the
full IQ mul_mat set; Metal and Vulkan keep whatever IQ ops their own backends
implement (Vulkan's `supports_op` still declines some IQ mul_mat/cpy shapes, but
it declines cleanly and falls back to CPU rather than crashing — which, after
this fix, is exactly how CUDA behaves too).

## Troubleshooting

### "make: command not found" or Wrong Make

Ensure using the cosmocc make:

```sh
# Wrong
make -j $(nproc)

# Correct
.cosmocc/4.0.2/bin/make -j $(nproc)

# Or use the command directly:
# llamafile:build
```

### Submodule Not Initialized

If build fails with missing files in llama.cpp/whisper.cpp/stable-diffusion.cpp:

```sh
make setup
```

### Stale Object Files

After significant changes, clean and rebuild:

```sh
.cosmocc/4.0.2/bin/make clean          # or: llamafile:clean
.cosmocc/4.0.2/bin/make -j $(nproc)   # or: llamafile:build
```

### Toolchain Checksum Mismatch

If `download-cosmocc.sh` fails verification, check:
1. Correct version specified
2. Correct checksum for that version
3. Network connectivity
