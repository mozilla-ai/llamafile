# Contributing to llamafile

Thank you for your interest in contributing to llamafile.

We welcome fixes, docs improvements, tests, build work, and larger feature work. 

Submodule changes (`llama.cpp/`, `whisper.cpp/`, `stable-diffusion.cpp/`) are applied as patches rather than committed directly. If your change should also go upstream, open a PR to the upstream repository (e.g., [llama.cpp](https://github.com/ggml-org/llama.cpp)). Otherwise, follow the [submodule changes workflow](#submodule-changes) described below.

## Before You Start

### Check for duplicates

Before starting new work:

- Search [existing issues](https://github.com/mozilla-ai/llamafile/issues) for duplicates
- Check [open pull requests](https://github.com/mozilla-ai/llamafile/pulls) to see if someone is already working on it
- For bugs, verify the issue still exists on `main`

### Discuss major changes first

Please open an issue before starting larger changes such as:

- new user-facing features
- architectural changes
- changes to public behavior or defaults
- new dependencies
- significant build or packaging changes

This helps us stay aligned and avoids duplicate work.

## Development Setup

### Prerequisites

You will need:

- GNU `make` (called `gmake` on some systems)
- `sha256sum` or a working `cc`
- `wget` or `curl`
- `unzip`
- Git

Windows contributors can use [MSYS2](https://www.msys2.org/) or WSL. See [docs/building_dlls.md](docs/building_dlls.md) for detailed Windows setup instructions.

### Quick Start

```sh
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/llamafile.git
cd llamafile

# 3. Add upstream remote
git remote add upstream https://github.com/mozilla-ai/llamafile.git

# 4. Set up submodules, patches, and toolchain
make setup

# 5. Build with cosmocc's make
.cosmocc/4.0.2/bin/make -j8

# 6. Run the default test suite
.cosmocc/4.0.2/bin/make check
```

`make setup` initializes submodules, applies llamafile-specific patches, and downloads the `cosmocc` toolchain into `.cosmocc/`.

For builds and tests, use `.cosmocc/4.0.2/bin/make`, not your system `make`.

## Making Changes

### 1. Create a branch

Always work on a branch, not directly on `main`:

```sh
git checkout -b docs/your-change
```

Common branch prefixes:

- `docs/` for documentation
- `fix/` for bug fixes
- `feature/` for new features
- `build/` for build and tooling changes

### 2. Make your changes

There are two common workflows in this repo.

#### Core code changes

For changes in directories like:

- `llamafile/`
- `whisperfile/`
- `docs/`
- `tests/`

you can edit files normally, rebuild, test, and commit as usual.

#### Submodule changes

The following directories are submodules:

- `llama.cpp/`
- `whisper.cpp/`
- `stable-diffusion.cpp/`

If you change code inside one of those directories, you also need to save those changes as patches in the matching `*.patches/` directory.

When working inside a submodule, follow that submodule's local coding and contribution guidelines in addition to this repository's workflow.

Example for `llama.cpp`:

```sh
cd llama.cpp
../tools/generate-patches.sh --output-dir ../llama.cpp.patches
```

After generating patches, verify them from a clean state:

```sh
make reset-repo
make setup
.cosmocc/4.0.2/bin/make -j8
.cosmocc/4.0.2/bin/make check
```

For a more detailed walkthrough of the patch-based workflow, see [docs/skills/llamafile/development.md](docs/skills/llamafile/development.md#making-changes-to-a-submodule).

### 3. Write tests

Please add or update tests whenever your change affects behavior.

- New features should include tests
- Bug fixes should include a regression test when practical
- Docs-only changes usually do not need tests
- Avoid mixing unrelated changes in one pull request

There are also integration tests under [tests/integration/README.md](tests/integration/README.md) if you want to validate changes with a real model.

### 4. Update documentation

If your change affects how developers or users work with llamafile, update the relevant docs in `README.md` or `docs/`.

If you add a new page to `docs/`, also add it to [`docs/SUMMARY.md`](docs/SUMMARY.md) — that file controls the GitBook navigation and is maintained by hand. CI will catch any SUMMARY entries that point to missing files, but it will not catch a new file that was never added to SUMMARY.

### 5. Commit your changes

Use clear commit messages:

```sh
git commit -m "Fix server startup when model path is missing"
git commit -m "Update contributor guide for patch workflow"
```

## Submitting Changes

Before opening a pull request, please make sure:

- the project builds cleanly
- the default test suite passes
- submodule changes have been converted into patch files
- related documentation has been updated
- the change is focused and easy to review
- you are ready to explain and maintain the code you changed

## Useful Docs

- [README.md](README.md)
- [docs/source_installation.md](docs/source_installation.md)
- [docs/running_llamafile.md](docs/running_llamafile.md)
- [docs/creating_llamafiles.md](docs/creating_llamafiles.md)
- [tests/integration/README.md](tests/integration/README.md)
