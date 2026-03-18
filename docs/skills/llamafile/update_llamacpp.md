# Keeping llamafile updated with upstream llama.cpp

llamafile relies on llama.cpp for many of its functionalities. Keeping it up-to-date
with the latest version upstream is generally a good practice, as it brings both
bugfixes and support for recent models and features.

This document describes the steps to keep llamafile updated with upstream.

## Step 1: Update the submodule

The output of this step is a new branch with the submodule checked out
at its latest commit id.

```bash
# make sure the submodule is initialized
git submodule update --init llama.cpp

# check current commit
cd llama.cpp
OLD_ID=`git rev-parse HEAD`

# checkout latest commit
git fetch origin master
COMMIT_ID=`git rev-parse origin/master`
git checkout origin/master

# create new branch for merging
cd ..
git checkout -b llamacpp_$COMMIT_ID
git add llama.cpp
git commit -m "Update llama.cpp submodule to $COMMIT_ID"

# this branch becomes the starting point of a new PR
```

## Step 2: Verify and update patches

Review the patches in `llama.cpp.patches/patches/` as follows:

- As a first pass, run `tools/check_patches.sh` to check if applying any of the
patches causes an error. Directly apply all and only the patches you see working.

- Any patch that has conflicts due to upstream changes has to be inspected 
in detail and updated. Useful references are:
  - the file the patch refers to
  - the patch description in `llama.cpp.patches/README.md`

- To update patches that have conflicts, first edit the new llama.cpp code
in-place, then call the `generate_patches` script (more info in `development.md`).

At the end of this step, your patches should all work (i.e. it should be possible
to apply them without conflicts). Note that you might still not have a working build,
but you should at least be able to run `make setup` without any errors.

## Step 3: Update BUILD.mk dependencies

- Review `llama.cpp/BUILD.mk` for any new source files or dependencies added upstream
- Remove references to any deleted source files
- Ensure all new dependencies are properly included
- Check the upstream changes for new/removed files in `llama.cpp/src/`, `llama.cpp/common/`, `llama.cpp/ggml/`, `llama.cpp/tools`, (all the relevant subdirectories you'd find in `llama.cpp/BUILD.mk`)

Useful references:

- check changes in each dir
```bash
cd llama.cpp
git diff --stat --summary $OLD_ID -- src/
```

- the `llama.cpp/CMakeLists.txt` file, showing what files are included in the latest llama.cpp build

At the end of this step, the `llama.cpp/BUILD.mk` file should include all the
updated dependencies to build, at least, the `o//llama.cpp/server/llama-server`
target.

## Step 4: Update llamafile integration code

- Check if the llamafile code that calls llama.cpp server/main needs updates
- Review `llamafile/` for any API changes in llama.cpp that need to be reflected
- Pay attention to changes in `llama.cpp/include/` for API modifications

At the end of this step, you should be able to build all targets in this repo,
i.e. the following verification step should return a successful result

## Verification

After making changes, verify the build works:
```sh
# llamafile:clean
# llamafile:build
```



## Reference

- **Upstream changes:** https://github.com/ggerganov/llama.cpp/compare/$OLD_ID...$COMMIT_ID
- **Example PR with similar updates:** https://github.com/mozilla-ai/llamafile/pull/847
