---
name: llamafile
description: This skill should be used when the user asks to "build llamafile", "rebuild llamafile", "run llamafile tests", "set up llamafile", "update patches", "modify llama.cpp", "how does llamafile work", "llamafile architecture", or needs guidance on the llamafile build system, patch workflow, submodule integration, or development practices.
version: 0.1.0
---

# Llamafile Development Guide

Llamafile combines llama.cpp, whisper.cpp, and stable-diffusion.cpp with Cosmopolitan Libc to create single-file executables that run LLMs locally across Windows, macOS, Linux, and BSD without installation.

## Version Disambiguation

- **New llamafile** (or simply "llamafile"): The work-in-progress rebuild in the `new_build_wip` branch, based on a more recent llama.cpp
- **Old/Classic llamafile**: The original version currently in the `main` branch

This guide covers the **new llamafile** build system.

## Quick Reference

### Initial Setup

```sh
make setup
```

Immediately after cloning the repo (or after a reset done with `make reset-repo`, this command initializes git submodules and applies llamafile-specific patches.

### Building

```sh
# Download toolchain if needed
build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44

# Build everything
.cosmocc/4.0.2/bin/make -j8
```

### Testing

```sh
.cosmocc/4.0.2/bin/make check
```

### Cleaning

```sh
.cosmocc/4.0.2/bin/make clean
```

### Reset Submodules

After `make setup`, the submodules are patched so they are not in a clean state anymore.
To reset them you can run:

```sh
make reset-repo  # Warning: removes all local changes
```

WARNING: this command removes all local changes: you don't want to run this if care about
the changes you have done and have not created patches from them yet.


## Core Workflows

### Building from Scratch

To build llamafile from a fresh clone:

1. Clone the repository and checkout `new_build_wip` branch
2. Run `make setup` to initialize submodules and apply patches
3. Download cosmocc if not present: `build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44`
4. Build with `.cosmocc/4.0.2/bin/make -j8`

Build outputs appear in `o/$(MODE)/` directory.

### Modifying Core Code

For changes to llamafile's own code (not submodules):

1. Edit files in `llamafile/` directory
2. Rebuild with `.cosmocc/4.0.2/bin/make -j8`
3. Run unit tests with `.cosmocc/4.0.2/bin/make check`

### Modifying Submodule Code

Submodules (llama.cpp, whisper.cpp, stable-diffusion.cpp) require a patch-based workflow:

1. Make changes directly in the submodule directory
2. Rebuild with `.cosmocc/4.0.2/bin/make -j8`
3. Run unit tests with `.cosmocc/4.0.2/bin/make check`

NOTE: here we directly build and test dirty repos, but at some point we want to bring
our changes into a commit. To do that, we need to generate patches from them: see
`references/development.md` for detailed patch workflow.

### Running Specific Tests

Tests use the `.runs` pattern in BUILD.mk files:

```makefile
o/$(MODE)/llamafile/json_test.runs
```

To run all tests: `.cosmocc/4.0.2/bin/make check`

## Key Concepts

### Cosmopolitan Toolchain

The project uses Cosmopolitan Libc (cosmocc) to create Actually Portable Executables (APE) - single files that run on multiple platforms without modification. Always use `.cosmocc/4.0.2/bin/make` for building, not system make.

### Patch System

Each submodule has a corresponding patches directory:
- `llama.cpp.patches/`
- `whisper.cpp.patches/`
- `stable-diffusion.cpp.patches/`

Patches include:
- **Modifications** (.patch files): Changes to upstream code
- **Additions** (llamafile-files/): New files for integration (BUILD.mk, utilities)
- **Deletions**: Removal of upstream build systems

### Build System

- **build/config.mk**: Compiler and toolchain configuration
- **build/rules.mk**: Generic build patterns (.c → .o, archives, asset bundling)
- **BUILD.mk files**: Per-package build logic

Outputs: `o/$(MODE)/package/file.o`

### Multi-Architecture Support

Binaries include both x86_64 and aarch64 code paths with runtime CPU feature detection (AVX, AVX2, AVX-512, ARM NEON).

## Main Executables

After building, find binaries in `o/$(MODE)/`:

| Binary | Purpose |
|--------|---------|
| `llamafile/llamafile` | Main LLM inference CLI |
| `third_party/zipalign/zipalign` | Bundle assets into executables |

## Troubleshooting

### Build Fails After Submodule Update

Run `make setup` to reapply patches after any submodule changes.

### Submodule Has Uncommitted Changes

To reset a single submodule:
```sh
cd <submodule> && git reset --hard && git clean -fdx
```

To reset all submodules:
```sh
make reset-repo
```

### Wrong Make Being Used

Ensure using cosmocc's make, not system make:
```sh
.cosmocc/4.0.2/bin/make -j8  # Correct
make -j8                      # Wrong - uses system make
```

## Additional Resources

### Reference Files

For detailed information, consult:
- **`references/building.md`** - Complete build system documentation, toolchain details
- **`references/architecture.md`** - Repository structure, component overview
- **`references/development.md`** - Development workflow, patch management, submodule integration
- **`references/testing.md`** - Test patterns, running and writing tests

### Project Documentation

- **README.md** in repo: Project introduction
- **docs/** directory: User documentation (quickstart, installation, troubleshooting)
- **RELEASE.md**: Release process
- Most executables support `--help`
