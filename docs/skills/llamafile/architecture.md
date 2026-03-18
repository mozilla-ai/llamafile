# Llamafile Architecture

Repository structure and component overview.

## Project Overview

Llamafile creates single-file executables that run LLMs locally across Windows, macOS, Linux, and BSD without installation. It achieves this by:

1. Combining multiple inference engines (llama.cpp, whisper.cpp, stable-diffusion.cpp)
2. Using Cosmopolitan Libc for cross-platform portability
3. Bundling models and assets into Actually Portable Executables (APE)

## Repository Structure

```
llamafile/
├── llamafile/              # Core library
│   ├── server/             # HTTP server implementation
│   └── highlight/          # Syntax highlighting
├── llama.cpp/              # LLM inference (submodule)
│   ├── ggml/               # Low-level tensor ops
│   ├── src/                # Model implementations
│   ├── common/             # Utilities
│   └── tools/              # CLI applications
├── whisper.cpp/            # Speech-to-text (submodule)
├── stable-diffusion.cpp/   # Image generation (submodule)
├── localscore/             # Benchmarking tool
├── third_party/            # External dependencies
├── build/                  # Build system
├── docs/                   # User documentation
├── *.patches/              # Patch directories
└── o/                      # Build outputs
```

## Core Components

### llamafile/ - Core Library

The heart of llamafile, containing:

- **tinyblas**: BLAS kernels for CUDA support without cublas and optimized CPU inference
- **GPU support**: Metal, CUDA and ROCm integration (dynamic loading)
- **Multiplatform optimizations**: CPU feature detection, runtime dispatch
- **TUI**: Chat interface running in the terminal

#### llamafile/highlight/

Syntax highlighting for code output in chat responses.

### llama.cpp/ - LLM Inference Engine

Git submodule providing:

- **ggml/**: Low-level tensor library
  - Matrix operations
  - Quantization support
  - Backend abstraction (CPU, CUDA, Metal, etc.)

- **src/**: LLM implementations
  - 100+ model architectures
  - GGUF format handling
  - KV cache management

- **common/**: Shared utilities
  - Argument parsing
  - Sampling algorithms
  - Chat templates

- **tools/**: CLI applications
  - main (inference)
  - quantize (model quantization)
  - imatrix (importance matrix)
  - perplexity (model evaluation)
  - llama-bench (benchmarking)

### whisper.cpp/ - Speech-to-Text

Git submodule for audio transcription:
- Whisper model implementation
- Audio processing utilities
- Multiple model sizes (tiny to large)

### stable-diffusion.cpp/ - Image Generation

Git submodule for image synthesis:
- Stable Diffusion implementation
- Image encoding/decoding
- Various SD model support

### third_party/ - Dependencies

External libraries:
- **double-conversion**: Float-to-string conversion
- **mbedtls**: TLS/SSL support
- **sqlite**: Database support
- **stb**: Image loading/saving
- **zipalign**: Tool to bundle llamafile executables with model weights and configurations

## Patch System

Each submodule has a corresponding patches directory:

```
llama.cpp.patches/
├── patches/           # .patch files modifying upstream
└── llamafile-files/   # New files for integration

whisper.cpp.patches/
├── patches/
└── llamafile-files/

stable-diffusion.cpp.patches/
├── patches/
└── llamafile-files/
```

### Patch Types

1. **Modifications** (`.patch` files):
   - Changes to existing upstream code
   - Applied with `git apply`
   - Track upstream file changes

2. **Additions** (`llamafile-files/`):
   - New files for llamafile integration
   - Example: BUILD.mk for each submodule
   - Utility scripts
   - Additional documentation

3. **Deletions**:
   - Removal of upstream build systems (CMakeLists.txt, Makefiles)
   - Replaced by llamafile's unified build
   - NOTE: deletions were common in the original llamafile but are no longer used,
as submodule code is pulled rather than redistributed

### Patch Application

`make setup` applies patches:
1. Initialize/update git submodules
2. Apply each .patch file in order
3. Copy llamafile-files/ contents into submodule
4. Remove conflicting build files

Finally, if cosmocc is not present, it is automatically downloaded at the end of `make setup`.

## Build Infrastructure

### build/ Directory

```
build/
├── config.mk              # Toolchain configuration
├── rules.mk               # Generic build patterns
├── download-cosmocc.sh    # Toolchain download
├── llamafile-convert      # Model conversion
└── llamafile-upgrade-engine   # Engine updates
```

### BUILD.mk Pattern

Each component has a BUILD.mk defining:

```makefile
# Source files
COMPONENT_SRCS = \
    component/file1.c \
    component/file2.c

# Object files
COMPONENT_OBJS = $(COMPONENT_SRCS:%.c=o/$(MODE)/%.o)

# Library target
o/$(MODE)/component/libcomponent.a: $(COMPONENT_OBJS)

# Executable target
o/$(MODE)/component/binary: o/$(MODE)/component/libcomponent.a

# Test targets
o/$(MODE)/component/test.runs: o/$(MODE)/component/test
```

### Output Organization

```
o/$(MODE)/
├── package/
│   ├── file.o         # Object files
│   ├── libpackage.a   # Static libraries
│   └── binary         # Executables
└── ...
```

## Key Technologies

### Actually Portable Executable (APE)

Cosmopolitan's executable format:
- Single file runs on Windows, macOS, Linux, BSD
- Contains x86_64 and aarch64 code
- Self-extracting when needed
- No installation required

### Asset Bundling

Files embedded into executables:
- Models (.gguf)
- Web assets (HTML, CSS, JS)
- Shared libraries (.so, .dll)

The `zipalign` tool handles bundling, and files are accessible via Cosmopolitan's VFS.

### Runtime CPU Dispatch

Binaries detect CPU features and select optimal code:
- x86_64: SSE, AVX, AVX2, AVX-512, FMA
- aarch64: NEON, SVE

This happens transparently at runtime, no user configuration needed.

### Dynamic GPU Loading

GPU support loads at runtime:
- CUDA: Loads from system or bundled .so/.dll
- ROCm: Similar dynamic loading
- Fallback to CPU if GPU unavailable

## Licensing

- **Llamafile project**: Apache 2.0
- **Llamafile changes to llama.cpp**: MIT (upstream compatibility)
- **Dependencies**: Retain original licenses
