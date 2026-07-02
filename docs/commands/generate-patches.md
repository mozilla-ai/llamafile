---
description: Regenerate llamafile patches from in-place submodule edits
---

# Generate Patches

Capture the current in-place edits of a submodule as patch files, using the
project's `generate_patches.sh` tool. This is the **only** sanctioned way to
produce patches — never hand-craft them with `git diff` (the tool rewrites the
`a/`,`b/` paths to be repo-rooted, strips the volatile `index` line, names files
by convention, and routes new/untracked files into `llamafile-files/`; a raw
`git diff` gets all of that wrong).

Run it for the `llama.cpp` submodule (the common case). The tool is general and
works for any submodule — swap the directory and output path for `whisper.cpp`
or `stable-diffusion.cpp`.

The subshell keeps the working directory restored even if the tool fails, and
`echo y` answers the script's confirmation prompt non-interactively:

```bash
( cd llama.cpp && echo y | ../tools/generate_patches.sh --output-dir ../llama.cpp.patches )
```

Output lands in `llama.cpp.patches/patches/` (modified files) and
`llama.cpp.patches/llamafile-files/` (new files, including `BUILD.mk`).

The tool **only writes/overwrites — it never deletes**. If you dropped a patch
during a bump (the file is no longer modified, e.g. upstream absorbed the
change), its old `.patch` will still be sitting in `patches/` and will keep
being applied by `setup`. `git rm` each dropped patch by hand, and confirm the
final count (`ls llama.cpp.patches/patches | wc -l`) matches your intent.

IMPORTANT: only run this **after** the in-place edits are proven to work
(a clean build succeeds and llamafile runs as expected). Generating patches
from unproven edits bakes in breakage. After generating, verify the patch set
round-trips with `llamafile:verify-clean`.
