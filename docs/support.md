
## Supported OSes

llamafile supports the following operating systems, which require a minimum
stock install:

- Linux 2.6.18+ (i.e. every distro since RHEL5 c. 2007)
- Darwin (macOS) 23.1.0+ [1] (GPU is only supported on ARM64)
- Windows 10+ (AMD64 only)
- FreeBSD 13+
- NetBSD 9.2+ (AMD64 only)
- OpenBSD 7.0 to 7.4 (AMD64 only)

On Windows, llamafile runs as a native portable executable. On UNIX
systems, llamafile extracts a small loader program named `ape` to
`$TMPDIR/.ape-1.10` which is used to map your model into memory.

[1] Darwin kernel versions 15.6+ *should* be supported, but we currently
    have no way of testing that.

## Supported CPUs

llamafile supports the following CPUs:

- **AMD64** microprocessors must have AVX. Otherwise llamafile will
  print an error and refuse to run. This means that if you have an Intel
  CPU, it needs to be Intel Core or newer (circa 2006+), and if you have
  an AMD CPU, then it needs to be K8 or newer (circa 2003+). Support for
  AVX512, AVX2, FMA, F16C, and VNNI are conditionally enabled at runtime
  if you have a newer CPU. For example, Zen4 has very good AVX512 that
  can speed up BF16 llamafiles.

- **ARM64** microprocessors must have ARMv8a+. This means everything
  from Apple Silicon to 64-bit Raspberry Pis will work, provided your
  weights fit into memory.

## GPU support

llamafile ships GPU acceleration for Apple Metal, NVIDIA, AMD, and Vulkan.
There is no Intel oneAPI/SYCL backend, but Vulkan covers a lot of the
hardware that CUDA and ROCm do not. On hardware outside the table below,
llamafile runs on the CPU.

CUDA, ROCm, and Vulkan dynamic libraries are loaded as follows. The executable looks
for prebuilt `ggml-(cuda|rocm|vulkan).(so|dll)` files in this order:

- in the same directory (deliberately first, so a hand-built DSO overrides everything else)
- in the llamafile bundle
- in `~/.llamafile/v/LLAMAFILE_VERSION`
- in `$HOME`

Check out [Building the GPU libraries](building_dlls.md) for more info on how these libraries are built.
On macOS, Vulkan runs through MoltenVK; Apple Silicon users normally want the built-in Metal
backend instead. If any library loads but reports no usable device, llamafile skips it and continues
with the next backend rather than failing.

| Vendor | Backend | Platforms | Status | Notes |
|--------|---------|-----------|--------|-------|
| Apple | Metal (built-in) | macOS ARM64 | Supported | Offload is enabled by default; disable with `-ngl 0` or `--gpu disable` |
| NVIDIA | CUDA / cuBLAS | Linux, Windows, WSL | Supported | Pass `-ngl 999` to offload; the bundled library needs compute capability 7.5+ (see below) |
| AMD | HIP / rocBLAS | Linux, Windows | Supported | Pass `-ngl 999` to offload; the library is downloaded separately and needs the ROCm runtime (see below); multi-GPU may be broken on Radeon |
| Any (incl. Intel) | Vulkan | Linux, Windows, macOS | Supported | Pass `-ngl 999` to offload; select with `--gpu vulkan`. Used in `--gpu auto` when no vendor backend is available |

The 0.10.* series has not been tested on every GPU and platform yet, so
treat the AMD and Windows paths in particular as best-effort.

### IQ-quantized models on NVIDIA GPUs

The CUDA library bundled in our releases is size-optimized and leaves out
the IQ-quant kernels (the `IQ1_*`, `IQ2_*`, `IQ3_*`, `IQ4_*` quantizations).
When you offload an IQ-quantized model to an NVIDIA GPU, llamafile keeps just
those layers on the CPU automatically — the output is correct, those specific
layers simply don't get GPU acceleration. Every other quantization (`Q*_*`,
`MXFP4`, `NVFP4`, `F16`, `BF16`, …) runs fully on the GPU. The Apple Metal and
AMD (ROCm) builds are not size-optimized and include full IQ-quant GPU support.
For full IQ acceleration on NVIDIA, build or supply a full (non-minimized) CUDA
library — see [Building the GPU libraries](building_dlls.md).

GPU on MacOS ARM64 is supported by compiling a small module using the
Xcode Command Line Tools, which need to be installed. This is a one time
cost that happens the first time you run your llamafile. The DSO built
by llamafile is stored in `$TMPDIR/.llamafile` or `$HOME/.llamafile`.
Offloading to GPU is enabled by default when a Metal GPU is present.
This can be disabled by passing `-ngl 0` or `--gpu disable` to force
llamafile to perform CPU inference.

Owners of NVIDIA and AMD graphics cards need to pass the `-ngl 999` flag
to enable maximum offloading. If multiple GPUs are present then the work
will be divided evenly among them by default, so you can load larger
models. Multiple GPU support may be broken on AMD Radeon systems. If
that happens to you, then use `export HIP_VISIBLE_DEVICES=0` which
forces llamafile to only use the first GPU.

Windows users are encouraged to use our release binaries, because they
bundle prebuilt CUDA and Vulkan DLLs, which only depend on the graphics
driver being installed. llamafile does not compile GPU libraries at
runtime, so installing NVIDIA's CUDA SDK or AMD's HIP SDK does not make
it build one; to use a different library (for example one linked against
cuBLAS), build it with the scripts described in
[Building the GPU libraries](building_dlls.md) and place it next to the
executable. You can use CUDA via WSL by enabling
[Nvidia CUDA on
WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
and running your llamafiles inside of WSL. Using WSL has the added
benefit of letting you run llamafiles greater than 4GB on Windows.

On Linux, the bundled CUDA library only needs the NVIDIA driver, so the
CUDA SDK is not required. AMD users need the ROCm runtime and the
`ggml-rocm.so` library described below. Running with `--gpu amd` or
`--gpu nvidia` is a useful way to turn an otherwise quiet CPU fallback
into an explicit startup error while you diagnose the setup.

If you have both an AMD GPU *and* an NVIDIA GPU in your machine, then
you may need to qualify which one you want used, by passing either
`--gpu amd` or `--gpu nvidia`.

In the event that GPU support couldn't be set up for any reason,
llamafile will fall back to CPU inference.

### Supported NVIDIA GPU generations

The CUDA library bundled in our releases contains GPU code for compute
capability 7.5 (Turing: RTX 20 and GTX 16 series) and newer. Older GPUs,
such as Maxwell (GTX 900 series), Pascal (GTX 10 series) and Volta, have no
kernels they can run in it, so CUDA offload is not available on them; the
Vulkan backend may still work. You can check your GPU's compute capability
with `nvidia-smi --query-gpu=name,compute_cap --format=csv`. To use CUDA on
an older GPU, build the library yourself with a `-gencode` target for it
(see [Building the GPU libraries](building_dlls.md)). CUDA 13 no longer
supports Maxwell, Pascal or Volta, so use CUDA 12.x for those.

### AMD GPUs

The ROCm library is not bundled in the release binaries. Download
`ggml-rocm.dll` (Windows) or `ggml-rocm.so` (Linux) from
[Hugging Face](https://huggingface.co/mozilla-ai/llamafile_0.10/tree/main)
and put it in one of the locations listed above. It is linked against the
ROCm runtime, which must be installed separately: on Windows,
`ggml-rocm.dll` needs `amdhip64_7.dll` and `libhipblas.dll`, which come with
the [AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html),
not with the graphics driver. If they can't be found, `--verbose` prints
`probing library ...ggml-rocm.dll` followed by `failed to load library`.

The prebuilt library targets gfx906, gfx942, gfx1030, gfx1031, gfx1032,
gfx1100, gfx1101, gfx1102 and gfx1103. For other GPUs, build it yourself
and set `OFFLOAD_ARCH` (for example `--offload-arch=gfx1035`), see
[Building the GPU libraries](building_dlls.md).

### Verifying GPU acceleration

Because llamafile silently falls back to the CPU when a GPU backend can't
be set up, it isn't always obvious whether offloading actually happened.
To check:

- Pass `-ngl 999` on NVIDIA and AMD to request maximum offloading (Metal
  offloads by default).
- Force a specific backend with `--gpu nvidia`, `--gpu amd`, or
  `--gpu vulkan`. This turns an otherwise quiet CPU fallback into an
  explicit startup error, which makes a missing or misconfigured
  CUDA/ROCm/Vulkan installation easy to spot.
- Watch the startup logs for the messages about building and loading the
  GPU module. If you don't see them, llamafile is running on the CPU.

**NOTE** that the 0.10.* build of llamafile has not been tested on all
GPUs/platforms yet, so we welcome your feedback both whether there are
any issues or if everything runs smoothly on your specific setup!
