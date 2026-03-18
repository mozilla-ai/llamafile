# Making a llamafile Release

There are a few steps in making a llamafile release which will be detailed in this document.

The two primary artifacts of the release are the `llamafile-<version>.zip` and the binaries for the GitHub release.

## Release Process

Note: Step 2 is only needed if you are making a new release of the ggml-cuda.so and ggml-rocm.so shared libraries. 
You only need to do this when you are making changes to the CUDA code or the APIs surrounding it.
Otherwise you can use the previous release of the shared libraries.

### Preparing the Build Environment

Before building, ensure all dependencies are initialized and configured:

```sh
make setup
```

This initializes git submodules (e.g., llama.cpp) and applies llamafile patches.
The patches integrate dependencies with llamafile's build system and add llamafile-specific functionality.

### Release Steps

1. Update the version number in `version.h`
2. Build the ggml-cuda.so and ggml-rocm.so shared libraries on Linux. llamafile uses TINYBLAS as a default, even if some model families (e.g. Qwen3.5) use CUBLAS as a default for CUDA.
    - You can do this by running the script `./llamafile/cuda.sh` and `./llamafile/rocm.sh` respectively.
    - The files will be built and placed your home directory.
3. Build the project with `make -j8`
4. Install the built project to your /usr/local/bin directory with `sudo make install PREFIX=/usr/local`

### llamafile Release Zip

The easiest way to create the release zip is to:

`make install PREFIX=<preferred_dir>/llamafile-<version>`

After the directory is created, you will want to bundle the built shared libraries into the release binaries (at the moment, llamafile only).

You can do this for each binary with a command like the following:

`zipalign -j0 llamafile ggml-cuda.so ggml-rocm.so`

The zip is structured as follows.

```
llamafile-<version>
|-- README.md
|-- bin
|   |-- llamafile
|   |-- whisperfile
|   `-- zipalign
`-- share
    `-- man
        `-- man1
            |-- whisperfile.1
            `-- zipalign.1
```

Before you zip the directory, you will want to remove the shared libraries from the directory (if present).

`rm *.so *.dll`

You can zip the directory with the following command:

`zip -r llamafile-<version>.zip llamafile-<version>`

### llamafile Release Binaries

After you have built the zip it is quite easy to create the release binaries.

The following binaries are part of the release:

- `llamafile`
- `whisperfile`
- `zipalign`

You can use the script to create the appropriately named binaries:

`./llamafile/release.sh -v <version> -s <source_dir> -d <dest_dir>`

Make sure to move the llamafile-<version>.zip file to the <dest_dir> as well, and you are good to release after you've tested.
