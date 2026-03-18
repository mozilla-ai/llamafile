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

**Critical:** Always use `.cosmocc/4.0.2/bin/make`, not system make. The cosmocc toolchain includes its own make with Cosmopolitan-specific behavior.

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

GPU acceleration (CUDA/ROCm) uses dynamic loading:
- Shared libraries (.so/.dll) are not linked at compile time
- Libraries are loaded at runtime if available
- Can be bundled into executables using zipalign

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
