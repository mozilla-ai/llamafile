# Llamafile Development Workflow

Guide to modifying code, managing patches, and working with submodules.

## Development Overview

Llamafile development involves two distinct workflows:

1. **Core code changes**: Direct edits to root-level directories such as `llamafile/`, `whisperfile/`, etc.
2. **Submodule changes**: Patch-based modifications to `llama.cpp`, `whisper.cpp`, `stable-diffusion.cpp`

## Modifying Core Code

For changes which are not affecting submodules:

### Workflow

1. Edit files
2. Rebuild: `llamafile:build`
3. Test: `llamafile:check`
4. Commit changes normally with git

### Key Directories

```
llamafile/
├── server/      # HTTP server, API endpoints
├── highlight/   # Syntax highlighting
├── tinyblas/    # Optimized BLAS kernels
└── *.c, *.h     # Core utilities
```

## Modifying Submodule Code

Submodules require a patch-based workflow because:
- Submodules point to specific upstream commits
- Direct commits in submodules would be lost
- Patches preserve modifications across submodule updates

### Understanding the Patch System

Each submodule has a patches directory. For instance, for `llama.cpp`:

```
llama.cpp.patches/
├── README.md              # Patching info + list of all patches and their purpose
├── apply-patches.sh       # Script to apply all patches to llama.cpp submodule
├── renames.sh             # Script for file renames/moves (if any)
├── llamafile-files/       # Additional files to copy into llama.cpp
│   ├── BUILD.mk           # Makefile for building llama.cpp with cosmocc
│   └── README.llamafile   # License and modification notes
└── patches/               # Patch files for upstream sources
```

Patches are applied by `make setup`:
1. Submodule is reset to clean state
2. Each .patch file is applied in alphabetical order
3. Files from llamafile-files/ are copied into the submodule

### Making Changes to a Submodule

#### Step 1: Make Changes

Edit files directly in the submodule directory:

```sh
cd llama.cpp
# Make your changes
vim src/llama.cpp
```

#### Step 2: Generate Patches

Patches are usually generated after the code has been thoroughly tested and is
ready to commit. To avoid manual errors, use the script `tools/generate-patches.sh`
which automatically saves all new files and patches in the specified output directory.

```sh
cd llama.cpp
../tools/generate-patches.sh --output-dir ../llama.cpp.patches
```

After this operation, one can double check which files have been modified / added
via a `git diff`.

Naming convention:
- all patches have a `.patch` extension
- patch filenames reflect the file path with underscores replacing slashes (e.g., `common_arg.cpp.patch` for `common/arg.cpp`).


#### Step 3: Verify Patches

Once you are sure all patches have been saved, reset and reapply to verify:

```sh
# Reset everything
make reset-repo

# Reapply patches
make setup

# Rebuild and test
# llamafile:build
# llamafile:check
```

### Adding New Files to Submodules

For new files (not modifications), use llamafile-files/:

```sh
# Create directory structure matching submodule
mkdir -p llama.cpp.patches/llamafile-files/src/

# Add your new file
cp new-utility.cpp llama.cpp.patches/llamafile-files/src/
```

The file will be copied into the submodule during `make setup`.

### Updating BUILD.mk for Submodules

Each submodule needs a BUILD.mk in llamafile-files/:

```makefile
# llama.cpp.patches/llamafile-files/BUILD.mk

LLAMA_SRCS = \
    llama.cpp/src/llama.cpp \
    llama.cpp/src/new-file.cpp    # Add new files here

LLAMA_OBJS = $(LLAMA_SRCS:%.cpp=o/$(MODE)/%.o)

# ... rest of build rules
```

## Submodule Management

### Resetting a Single Submodule

To discard changes in one submodule:

```sh
cd llama.cpp
git reset --hard
git clean -fdx
```

Then reapply patches:

```sh
cd ..
make setup
```

### Resetting All Submodules

To reset everything (warning: loses all local changes):

```sh
make reset-repo
make setup
```


## Git Workflow

### Committing Changes

For core code changes:
```sh
git add llamafile/modified-file.c
git commit -m "Fix: description"
```

For submodule patches:
```sh
git add llama.cpp.patches/patches/new-patch.patch
git commit -m "llama.cpp: Add feature X"
```

### Pull Request Checklist

Before submitting changes:

1. [ ] Patches apply cleanly from fresh clone
2. [ ] Build succeeds: `llamafile:build`
3. [ ] Tests pass: `llamafile:check`
4. [ ] Patches are focused and documented
5. [ ] BUILD.mk updated if adding new files

## Debugging Tips

### Viewing Applied Patches

To see what patches are currently applied:

```sh
cd llama.cpp
git log --oneline HEAD...$(git rev-parse --short @{u} 2>/dev/null || echo "origin/master")
```

### Checking Submodule State

```sh
git submodule status
```

Output shows:
- `-` : Not initialized
- `+` : Different commit than recorded
- ` ` : Clean, matches recorded commit

### Finding Which Patch Changed a File

```sh
grep -l "filename" llama.cpp.patches/patches/*.patch
```
