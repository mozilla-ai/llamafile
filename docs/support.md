
## Supported OSes

llamafile supports the following operating systems, which require a minimum
stock install:

- Linux 2.6.18+ (i.e. every distro since RHEL5 c. 2007)
- Darwin (macOS) 23.1.0+ [1] (GPU is only supported on ARM64)
- Windows 10+ (AMD64 only)
- FreeBSD 13+
- NetBSD 9.2+ (AMD64 only)
- OpenBSD 7+ (AMD64 only)

On Windows, llamafile runs as a native portable executable. On UNIX
systems, llamafile extracts a small loader program named `ape` to
`$TMPDIR/.llamafile` or `~/.ape-1.9` which is used to map your model
into memory.

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

llamafile supports the following kinds of GPUs:

- Apple Metal
- NVIDIA
- AMD

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
contain prebuilt DLLs for both NVIDIA and AMD graphics cards, which only
depend on the graphics driver being installed. If llamafile detects that
NVIDIA's CUDA SDK or AMD's ROCm HIP SDK are installed, then llamafile
will try to build a faster DLL that uses cuBLAS or rocBLAS. In order for
llamafile to successfully build a cuBLAS module, it needs to be run on
the x64 MSVC command prompt. You can use CUDA via WSL by enabling
[Nvidia CUDA on
WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
and running your llamafiles inside of WSL. Using WSL has the added
benefit of letting you run llamafiles greater than 4GB on Windows.

On Linux, NVIDIA users will need to install the CUDA SDK (ideally using
the shell script installer) and ROCm users need to install the HIP SDK.
They're detected by looking to see if `nvcc` or `hipcc` are on the PATH.

If you have both an AMD GPU *and* an NVIDIA GPU in your machine, then
you may need to qualify which one you want used, by passing either
`--gpu amd` or `--gpu nvidia`.

In the event that GPU support couldn't be compiled and dynamically
linked on the fly for any reason, llamafile will fall back to CPU
inference.
