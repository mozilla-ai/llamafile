# GPU Support

llamafile supports GPU acceleration on multiple platforms. This page covers how to enable and configure GPU support.

## Supported GPUs

llamafile supports the following kinds of GPUs:

- Apple Metal
- NVIDIA
- AMD

## Apple Metal (macOS)

GPU on macOS ARM64 is supported by compiling a small module using the Xcode Command Line Tools, which need to be installed. This is a one time cost that happens the first time you run your llamafile. The DSO built by llamafile is stored in `$TMPDIR/.llamafile` or `$HOME/.llamafile`.

Offloading to GPU is **enabled by default** when a Metal GPU is present. This can be disabled by passing `-ngl 0` or `--gpu disable` to force llamafile to perform CPU inference.

## NVIDIA and AMD GPUs

Owners of NVIDIA and AMD graphics cards need to pass the `-ngl 999` flag to enable maximum offloading. If multiple GPUs are present then the work will be divided evenly among them by default, so you can load larger models.

!!! note
    Multiple GPU support may be broken on AMD Radeon systems. If that happens to you, then use `export HIP_VISIBLE_DEVICES=0` which forces llamafile to only use the first GPU.

## Platform-specific setup

### Windows

Windows users are encouraged to use our release binaries, because they contain prebuilt DLLs for both NVIDIA and AMD graphics cards, which only depend on the graphics driver being installed.

If llamafile detects that NVIDIA's CUDA SDK or AMD's ROCm HIP SDK are installed, then llamafile will try to build a faster DLL that uses cuBLAS or rocBLAS. In order for llamafile to successfully build a cuBLAS module, it needs to be run on the x64 MSVC command prompt.

You can use CUDA via WSL by enabling [Nvidia CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) and running your llamafiles inside of WSL. Using WSL has the added benefit of letting you run llamafiles greater than 4GB on Windows.

### Linux

On Linux, NVIDIA users will need to install the CUDA SDK (ideally using the shell script installer) and ROCm users need to install the HIP SDK. They're detected by looking to see if `nvcc` or `hipcc` are on the PATH.

## Multiple GPU types

If you have both an AMD GPU *and* an NVIDIA GPU in your machine, then you may need to qualify which one you want used, by passing either `--gpu amd` or `--gpu nvidia`.

## Fallback behavior

In the event that GPU support couldn't be compiled and dynamically linked on the fly for any reason, llamafile will fall back to CPU inference.

## How GPU support works

Cosmopolitan Libc uses static linking, since that's the only way to get the same executable to run on six OSes. This presents a challenge for llama.cpp, because it's not possible to statically link GPU support.

The way we solve that is by checking if a compiler is installed on the host system. For Apple, that would be Xcode, and for other platforms, that would be `nvcc`. llama.cpp has a single file implementation of each GPU module, named `ggml-metal.m` (Objective C) and `ggml-cuda.cu` (Nvidia C). llamafile embeds those source files within the zip archive and asks the platform compiler to build them at runtime, targeting the native GPU microarchitecture. If it works, then it's linked with platform C library dlopen() implementation.

See [llamafile/cuda.c](https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/cuda.c) and [llamafile/metal.c](https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/metal.c) for implementation details.
