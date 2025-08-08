# Making a Llamafile Release

There are a few steps in making a Llamafile release which will be detailed in this document.

## What's New in This Release

### LoRA Adapter Support

This release adds comprehensive support for LoRA (Low-Rank Adaptation) adapters, enabling fine-tuning capabilities compatible with llama.cpp. Key features include:

- **Multiple LoRA Adapter Support**: Load and apply multiple LoRA adapters simultaneously with individual scaling factors
- **Server Integration**: Full integration with the llamafile server (`--server` mode) for LoRA-enhanced inference
- **Compatible Flags**: 
  - `--lora [FNAME]`: Apply a LoRA adapter with default scale (1.0)
  - `--lora-scaled [FNAME] [SCALE]`: Apply a LoRA adapter with custom scaling factor
  - `--lora-base [FNAME]`: Optional base model for LoRA adapter (advanced use cases)
- **Automatic Optimizations**: Memory mapping is automatically disabled when using LoRA adapters for optimal compatibility
- **Clean Resource Management**: Proper loading, application, and cleanup of LoRA adapters across server lifecycle

Example usage:
```bash
# Single adapter with default scale
llamafile -m base_model.gguf --lora adapter.gguf --server

# Multiple adapters with different scales
llamafile -m base_model.gguf --lora adapter1.gguf --lora-scaled adapter2.gguf 0.5 --server
```

This implementation follows llama.cpp patterns for maximum compatibility and provides a foundation for advanced fine-tuning workflows.

The two primary artifacts of the release are the `llamafile-<version>.zip` and the binaries for the GitHub release.

## Release Process

Note: Step 2 and 3 are only needed if you are making a new release of the ggml-cuda.so and ggml-rocm.so shared libraries. You only need to do this when you are making changes to the CUDA code or the API's surrounding it. Otherwise you can use the previous release of the shared libraries.

1. Update the version number in `version.h`
2. Build the ggml-cuda.so and ggml-rocm.so shared libraries on Linux. You need to do this for Llamafile and LocalScore. Llamafile uses TINYBLAS as a default and LocalScore uses CUBLAS as a default for CUDA.
    - For Llamafile you can do this by running the script `./llamafile/cuda.sh` and `./llamafile/rocm.sh` respectively.
    - For LocalScore you can do this by running the script `./localscore/cuda.sh`.
    - The files will be built and placed your home directory.
3. Build the ggml-cuda.dll and ggml-rocm.dll shared libraries on Windows. You need to do this for Llamafile and LocalScore.
    - You can do this by running the script `./llamafile/cuda.bat` and `./llamafile/rocm.bat` respectively.
    - For LocalScore you can do this by running the script `./localscore/cuda.bat`.
    - The files will be built and placed in the `build/release` directory.
4. Build the project with `make -j8`
5. Install the built project to your /usr/local/bin directory with `sudo make install PREFIX=/usr/local`

### Llamafile Release Zip

The easiest way to create the release zip is to:

`make install PREFIX=<preferred_dir>/llamafile-<version>`

After the directory is created, you will want to bundle the built shared libraries into the following release binaries:

- `llamafile`
- `localscore`
- `whisperfile`

You can do this for each binary with a command like the following:

Note: You MUST put the shared libraries in the same directory as the binary you are creating.

For llamafile and whisperfile you can do the following:

`zipalign -j0 llamafile ggml-cuda.so ggml-rocm.so ggml-cuda.dll ggml-rocm.dll`
`zipalign -j0 whisperfile ggml-cuda.so ggml-rocm.so ggml-cuda.dll ggml-rocm.dll`

After doing this, delete the ggml-cuda.so and ggml-cuda.dll files from the directory, and copy + rename the ggml-cuda.localscore.so and ggml-cuda.localscore.dll files to the directory.

```
rm <path_to>/llamafile-<version>/bin/ggml-cuda.so <path_to>/llamafile-<version>/bin/ggml-cuda.dll
cp ~/ggml-cuda.localscore.so <path_to>/llamafile-<version>/bin/ggml-cuda.so
cp ~/ggml-cuda.localscore.dll <path_to>/llamafile-<version>/bin/ggml-cuda.dll
```

For localscore you can now package it:

`zipalign -j0 localscore ggml-cuda.so ggml-rocm.so ggml-cuda.dll ggml-rocm.dll`

After you have done this for all the binaries, you will want to get the existing PDFs (from the prior release) and add them to the directory:

`cp <path_to>/doc/*.pdf <path_to>/llamafile-<version>/share/doc/llamafile/`

The zip is structured as follows.

```
llamafile-<version>
|-- README.md
|-- bin
|   |-- llamafile
|   |-- llamafile-bench
|   |-- llamafile-convert
|   |-- llamafile-imatrix
|   |-- llamafile-perplexity
|   |-- llamafile-quantize
|   |-- llamafile-tokenize
|   |-- llamafile-upgrade-engine
|   |-- llamafiler
|   |-- llava-quantize
|   |-- localscore
|   |-- sdfile
|   |-- whisperfile
|   `-- zipalign
`-- share
    |-- doc
    |   `-- llamafile
    |       |-- llamafile-imatrix.pdf
    |       |-- llamafile-perplexity.pdf
    |       |-- llamafile-quantize.pdf
    |       |-- llamafile.pdf
    |       |-- llamafiler.pdf
    |       |-- llava-quantize.pdf
    |       |-- whisperfile.pdf
    |       `-- zipalign.pdf
    `-- man
        `-- man1
            |-- llamafile-imatrix.1
            |-- llamafile-perplexity.1
            |-- llamafile-quantize.1
            |-- llamafile.1
            |-- llamafiler.1
            |-- llava-quantize.1
            |-- whisperfile.1
            `-- zipalign.1
```

Before you zip the directory, you will want to remove the shared libraries from the directory.

`rm *.so *.dll`

You can zip the directory with the following command:

`zip -r llamafile-<version>.zip llamafile-<version>`

### Llamafile Release Binaries

After you have built the zip it is quite easy to create the release binaries.

The following binaries are part of the release:

- `llamafile`
- `llamafile-bench`
- `llamafiler`
- `sdfile`
- `localscore`
- `whisperfile`
- `zipalign`

You can use the script to create the appropriately named binaries:

`./llamafile/release.sh -v <version> -s <source_dir> -d <dest_dir>`

Make sure to move the llamafile-<version>.zip file to the <dest_dir> as well, and you are good to release after you've tested.