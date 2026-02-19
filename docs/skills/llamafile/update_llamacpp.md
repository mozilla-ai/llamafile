## Step 1: Update the submodule

```bash
cd llama.cpp
git fetch origin master
git checkout origin/master
cd ..
git add llama.cpp
```

## Step 2: Verify and update patches

- Review the patches in `llama.cpp.patches/patches/`
- Check if each patch still applies cleanly to the new llama.cpp version
- Update any patches that have conflicts due to upstream changes
- Run the patch application script to verify:
    ```bash
    cd llama.cpp.patches && ./apply-patches.sh
    ```

## Step 3: Update BUILD.mk dependencies

- Review `llama.cpp/BUILD.mk` for any new source files or dependencies added upstream
- Remove references to any deleted source files
- Ensure all new dependencies are properly included
- Check the upstream changes for new/removed files in `llama.cpp/src/`, `llama.cpp/common/`, `llama.cpp/ggml/`

## Step 4: Update llamafile integration code

- Check if the llamafile code that calls llama.cpp server/main needs updates
- Review `llamafile/` for any API changes in llama.cpp that need to be reflected
- Pay attention to changes in `llama.cpp/include/` for API modifications

## Verification

After making changes, verify the build works:
```bash
make clean
make -j\$(nproc)
```

## Reference

- **Upstream changes:** https://github.com/ggerganov/llama.cpp/compare/f47edb8...983559d
- **Example PR with similar updates:** https://github.com/mozilla-ai/llamafile/pull/847
