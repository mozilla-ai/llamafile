# Whisper.cpp Patches for Llamafile Integration

This directory contains patches and additional files needed to integrate whisper.cpp into llamafile.

## Base Version

Based on whisper.cpp commit: `6739eb83c3ca5cf40d24c6fe8442a761a1eb6248`

## Modifications

The patches make the following changes to whisper.cpp:

1. **Server Integration** (`001-server-llamafile-integration.patch`)
   - Integrates with llamafile's file loader and debug systems
   - Adds support for llamafile command-line flags (--gpu, --nocompile, --recompile, etc.)
   - Removes ffmpeg dependency in favor of llamafile's built-in audio conversion
   - Renames `main()` to `whisper_server_main()` for integration

2. **Common Library Updates** (`002-common-llamafile-integration.patch`)
   - Integrates with llamafile's logging system
   - Removes duplicate DR_WAV implementation
   - Removes GPT-related code not needed for whisper server

3. **Core Whisper Library** (`003-whisper-core-llamafile-integration.patch`)
   - Uses llamafile's GGML and GPU detection
   - Replaces static GPU flags with runtime detection via llamafile
   - Updates include paths to use llama.cpp from llamafile

4. **Common Header** (`004-common-header.patch`)
   - Header file updates corresponding to common.cpp changes

## New Files

- `BUILD.mk` - Makefile integration for llamafile build system
- `README.llamafile` - Documentation of the integration
- `color.cpp/color.h` - Terminal color output utilities
- `slurp.cpp/slurp.h` - File loading utilities for llamafile

## Applying Patches

The patches are applied at build/development time, not committed to the submodule.

To apply the patches:

```bash
cd whisper.cpp.patches
./apply-patches.sh
```

This will:
1. Apply all patches to the appropriate files in the submodule
2. Copy new llamafile-specific files
3. Copy modified files from their original locations (examples/, src/, include/) to the root

The submodule itself remains at the clean upstream commit `6739eb83`.

## Resetting the Submodule

To reset the submodule back to its clean upstream state:

```bash
cd whisper.cpp.submodule
git reset --hard 6739eb83c3ca5cf40d24c6fe8442a761a1eb6248
git clean -fd
```

Then you can reapply patches as needed.

## Regenerating Patches

If you need to update the patches after making changes:

1. Apply patches and make your changes in the submodule
2. Regenerate patches from the llamafile root directory:

```bash
# From /Users/nbrake/scm/llamafile
diff -u whisper.cpp.submodule/examples/server/server.cpp whisper.cpp.submodule/server.cpp | \
  sed 's|whisper.cpp.submodule/examples/server/|examples/server/|' | \
  sed 's|whisper.cpp.submodule/|examples/server/|' > whisper.cpp.patches/001-server-llamafile-integration.patch

diff -u whisper.cpp.submodule/examples/common.cpp whisper.cpp.submodule/common.cpp | \
  sed 's|whisper.cpp.submodule/examples/|examples/|' | \
  sed 's|whisper.cpp.submodule/|examples/|' > whisper.cpp.patches/002-common-llamafile-integration.patch

diff -u whisper.cpp.submodule/src/whisper.cpp whisper.cpp.submodule/whisper.cpp | \
  sed 's|whisper.cpp.submodule/src/|src/|' | \
  sed 's|whisper.cpp.submodule/|src/|' > whisper.cpp.patches/003-whisper-core-llamafile-integration.patch

diff -u whisper.cpp.submodule/examples/common.h whisper.cpp.submodule/common.h | \
  sed 's|whisper.cpp.submodule/examples/|examples/|' | \
  sed 's|whisper.cpp.submodule/|examples/|' > whisper.cpp.patches/004-common-header.patch
```

## Workflow

1. Clone the repository with submodules: `git clone --recursive <repo>`
2. Apply patches: `cd whisper.cpp.patches && ./apply-patches.sh`
3. Build as normal: `make`

The build system references `whisper.cpp.submodule/` and expects the patches to be applied.
