Here is a succinct overview of the tricks we used to create the fattest
executable format ever. The long story short is llamafile is a shell
script that launches itself and runs inference on embedded weights in
milliseconds without needing to be copied or installed. What makes that
possible is mmap(). Both the llama.cpp executable and the weights are
concatenated onto the shell script. A tiny loader program is then
extracted by the shell script, which maps the executable into memory.
The llama.cpp executable then opens the shell script again as a file,
and calls mmap() again to pull the weights into memory and make them
directly accessible to both the CPU and GPU.

### ZIP weights embedding

The trick to embedding weights inside llama.cpp executables is to ensure
the local file is aligned on a page size boundary. That way, assuming
the zip file is uncompressed, once it's mmap()'d into memory we can pass
pointers directly to GPUs like Apple Metal, which require that data be
page size aligned. Since no existing ZIP archiving tool has an alignment
flag, we had to write about [500 lines of code](https://github.com/Mozilla-Ocho/llamafile/blob/HEAD/llamafile/zipalign.c) to
insert the ZIP files ourselves. However, once there, every existing ZIP
program should be able to read them, provided they support ZIP64. This
makes the weights much more easily accessible than they otherwise would
have been, had we invented our own file format for concatenated files.

### Microarchitectural portability

On Intel and AMD microprocessors, llama.cpp spends most of its time in
the matmul quants, which are usually written thrice for SSSE3, AVX, and
AVX2. llamafile pulls each of these functions out into a separate file
that can be `#include`ed multiple times, with varying
`__attribute__((__target__("arch")))` function attributes. Then, a
wrapper function is added which uses Cosmopolitan's `X86_HAVE(FOO)`
feature to runtime dispatch to the appropriate implementation.

### Architecture portability

llamafile solves architecture portability by building llama.cpp twice:
once for AMD64 and again for ARM64. It then wraps them with a shell
script which has an MZ prefix. On Windows, it'll run as a native binary.
On Linux, it'll extract a small 8kb executable called [APE
Loader](https://github.com/jart/cosmopolitan/blob/master/ape/loader.c)
to `${TMPDIR:-${HOME:-.}}/.ape` that'll map the binary portions of the
shell script into memory. It's possible to avoid this process by running
the
[`assimilate`](https://github.com/jart/cosmopolitan/blob/master/tool/build/assimilate.c)
program that comes included with the `cosmocc` compiler. What the
`assimilate` program does is turn the shell script executable into
the host platform's native executable format. This guarantees a fallback
path exists for traditional release processes when it's needed.

### GPU support

Cosmopolitan Libc uses static linking, since that's the only way to get
the same executable to run on six OSes. This presents a challenge for
llama.cpp, because it's not possible to statically link GPU support. The
way we solve that is by checking if a compiler is installed on the host
system. For Apple, that would be Xcode, and for other platforms, that
would be `nvcc`. llama.cpp has a single file implementation of each GPU
module, named `ggml-metal.m` (Objective C) and `ggml-cuda.cu` (Nvidia
C). llamafile embeds those source files within the zip archive and asks
the platform compiler to build them at runtime, targeting the native GPU
microarchitecture. If it works, then it's linked with platform C library
dlopen() implementation. See [llamafile/cuda.c](https://github.com/mozilla-ai/llamafile/blob/HEAD/llamafile/cuda.c) and
[llamafile/metal.c](https://github.com/mozilla-ai/llamafile/blob/HEAD/llamafile/metal.c).

In order to use the platform-specific dlopen() function, we need to ask
the platform-specific compiler to build a small executable that exposes
these interfaces. On ELF platforms, Cosmopolitan Libc maps this helper
executable into memory along with the platform's ELF interpreter. The
platform C library then takes care of linking all the GPU libraries, and
then runs the helper program which longjmp()'s back into Cosmopolitan.
The executable program is now in a weird hybrid state where two separate
C libraries exist which have different ABIs. For example, thread local
storage works differently on each operating system, and programs will
crash if the TLS register doesn't point to the appropriate memory. The
way Cosmopolitan Libc solves that on AMD is by using SSE to recompile
the executable at runtime to change `%fs` register accesses into `%gs`
which takes a millisecond. On ARM, Cosmo uses the `x28` register for TLS
which can be made safe by passing the `-ffixed-x28` flag when compiling
GPU modules. Lastly, llamafile uses the `__ms_abi__` attribute so that
function pointers passed between the application and GPU modules conform
to the Windows calling convention. Amazingly enough, every compiler we
tested, including nvcc on Linux and even Objective-C on MacOS, all
support compiling WIN32 style functions, thus ensuring your llamafile
will be able to talk to Windows drivers, when it's run on Windows,
without needing to be recompiled as a separate file for Windows. See
[cosmopolitan/dlopen.c](https://github.com/jart/cosmopolitan/blob/master/libc/dlopen/dlopen.c)
for further details.
