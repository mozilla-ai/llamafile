---
name: llamafile
description: This skill should be used when the user asks to "build llamafile", "rebuild llamafile", "run llamafile", "run llamafile tests", "debug llamafile", "set up llamafile", "update patches", "fix patch conflict", "update llama.cpp", "pull latest llama.cpp", "sync upstream llama.cpp", "reset submodules", "write a test for llamafile", "how does llamafile work", "llamafile architecture", or needs guidance on the llamafile build system, patch workflow, submodule integration, cosmocc toolchain, or development practices.
version: 0.1.2
---

# Llamafile Development Guide

Llamafile combines llama.cpp, whisper.cpp, and stable-diffusion.cpp with Cosmopolitan Libc to create single-file executables that run LLMs locally across Windows, macOS, Linux, and BSD without installation.

## Version Disambiguation

- **New llamafile** (or simply "llamafile"): The code in the `main` branch, used for releases >=0.10.0
- **Old/Classic llamafile**: The legacy code, used for releases until 0.9.3 (see commit 7e7d33c).

This guide covers the **new llamafile** project.

## Quick Reference

### Initial Setup

```sh
make setup
```

Immediately after cloning the repo (or after a reset done with `make reset-repo`), this command initializes git submodules and applies llamafile-specific patches.

### Building

Run `llamafile:build` to build all targets.

### Testing

Run `llamafile:check` to run the unit test suite.

### Cleaning

Run `llamafile:clean` to remove all build outputs.

### Reset Submodules

After `make setup`, submodules contain patches and are no longer in a clean state.
To reset them, run:

```sh
make reset-repo  # Warning: removes all local changes
```

WARNING: this command removes all local changes. Do not run it without first generating patches from any modifications.


## Core Workflows

### Building from Scratch

To build llamafile from a fresh clone:

1. Clone the repository
2. Run `make setup` to initialize submodules and apply patches
3. Build with `llamafile:build`

Build outputs appear in `o/$(MODE)/` directory.

### Modifying Core Code

For changes to llamafile's own code (not submodules):

1. Edit files in `llamafile/` directory
2. Rebuild with `llamafile:build`
3. Run unit tests with `llamafile:check`

### Modifying Submodule Code

Submodules (llama.cpp, whisper.cpp, stable-diffusion.cpp) require a patch-based workflow:

1. Make changes directly in the submodule directory
2. Rebuild with `llamafile:build`
3. Run unit tests with `llamafile:check`

NOTE: never try to edit patches or generate them manually. This step is 
done only after rebuild and tests (even manual ones) are successful. See
`development.md` for detailed patch workflow.

### Running Specific Tests

Tests use the `.runs` pattern in BUILD.mk files:

```makefile
o/$(MODE)/llamafile/json_test.runs
```

To run all tests: `llamafile:check`

## Key Concepts

### Cosmopolitan Toolchain

The project uses Cosmopolitan Libc (cosmocc) to create Actually Portable Executables (APE) - single files that run on multiple platforms without modification. Always use the `llamafile:build`, `llamafile:check`, and `llamafile:clean` commands (which use cosmocc's make), not system make.

### Patch System

Each submodule has a corresponding patches directory:
- `llama.cpp.patches/`
- `whisper.cpp.patches/`
- `stable-diffusion.cpp.patches/`

Patches include:
- **Modifications** (.patch files): Changes to upstream code
- **Additions** (llamafile-files/): New files for integration (BUILD.mk, utilities)

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
| `llamafile/llamafile` | Main llamafile executable |
| `third_party/zipalign/zipalign` | Bundle assets into executables |
| `whisperfile/whisperfile` | Main whisperfile executable |

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

Ensure using the `llamafile:build` command (which uses cosmocc's make), not system make.

## Additional Resources

### Reference Files

For detailed information, consult:
- **`building.md`** - Complete build system documentation, toolchain details
- **`architecture.md`** - Repository structure, component overview
- **`development.md`** - Development workflow, patch management, submodule integration
- **`testing.md`** - Test patterns, running and writing tests
- **`update_llamacpp.md`** - Keeping llamafile updated with upstream llama.cpp

### Project Documentation

- **README.md** in repo: Project introduction
- **docs/** directory: User documentation (quickstart, installation, troubleshooting)
- **RELEASE.md**: Release process
- Most executables support `--help`
