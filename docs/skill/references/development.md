# Llamafile Development Workflow

Guide to modifying code, managing patches, and working with submodules.

## Development Overview

Llamafile development involves two distinct workflows:

1. **Core code changes**: Direct edits to `llamafile/` directory
2. **Submodule changes**: Patch-based modifications to llama.cpp, whisper.cpp, stable-diffusion.cpp

## Modifying Core Code

For changes to llamafile's own code (not submodules):

### Workflow

1. Edit files in `llamafile/` directory
2. Rebuild: `.cosmocc/4.0.2/bin/make -j8`
3. Test: `.cosmocc/4.0.2/bin/make check`
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

Each submodule has a patches directory:

```
llama.cpp.patches/
├── patches/              # .patch files
│   ├── 001-fix-build.patch
│   ├── 002-add-feature.patch
│   └── ...
└── llamafile-files/      # New files to add
    ├── BUILD.mk
    └── ...
```

Patches are applied by `make setup`:
1. Submodule is reset to clean state
2. Each .patch file is applied in alphabetical order
3. Files from llamafile-files/ are copied into the submodule

### Making Changes to a Submodule

#### Step 1: Make Changes

Edit files directly in the submodule:

```sh
cd llama.cpp
# Make your changes
vim src/llama.cpp
```

#### Step 2: Generate Patches

Patches are usually generated after the code has been thoroughly tested and we feel
ready to commit it.

1. From the submodule directory, run `../tools/generate_patches.sh`. The tool will generate a patch for every file that was modified and save them in a `candidate_patches` directory
2. Move the patches to the patches directory corresponding to the submodule, e.g. `mv candidate_patches/* ../llama.cpp.patches/patches/`
3. For newly created files in the submodule directory (e.g. `BUILD.mk`), just copy them
in the `llamafile-files` directory: e.g. `cp BUILD.mk ../llama.cpp.patches/llamafile-files`
3. Now you can check all the patches for diffs with `git diff`, see what changed from the previous commit, and what new patches have been added

Naming convention:
- all patches have a `.patch` extension
- filenames match the corresponding paths (the slash character is converted to an underscore)

#### Step 3: Verify Patch

Once you are sure all patches have been saved, reset and reapply to verify:

```sh
# Reset everything
make reset-repo

# Reapply patches
make setup

# Rebuild and test
.cosmocc/4.0.2/bin/make -j8
.cosmocc/4.0.2/bin/make check
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

### Updating Submodules to New Upstream

When updating a submodule to a newer upstream commit:

1. Update the submodule pointer:
   ```sh
   cd llama.cpp
   git fetch origin
   git checkout <new-commit>
   cd ..
   git add llama.cpp
   ```

2. Try applying existing patches:
   ```sh
   make setup
   ```

3. Fix any patches that fail to apply:
   - Update the patch to match new upstream code
   - Or regenerate from scratch if changes are extensive

4. Test thoroughly:
   ```sh
   .cosmocc/4.0.2/bin/make clean
   .cosmocc/4.0.2/bin/make -j8
   .cosmocc/4.0.2/bin/make check
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

### Branches

- `main`: Old/classic llamafile (stable)
- `new_build_wip`: New llamafile (work in progress)

Development typically happens on `new_build_wip` or feature branches.

### Pull Request Checklist

Before submitting changes:

1. [ ] Patches apply cleanly from fresh clone
2. [ ] Build succeeds: `.cosmocc/4.0.2/bin/make -j8`
3. [ ] Tests pass: `.cosmocc/4.0.2/bin/make check`
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
