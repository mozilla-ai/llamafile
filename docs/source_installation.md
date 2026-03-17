Developing on llamafile requires a modern version of the GNU `make`
command (called `gmake` on some systems), `sha256sum` (otherwise `cc`
will be used to build it), `wget` (or `curl`), and `unzip` available at
[https://cosmo.zip/pub/cosmos/bin/](https://cosmo.zip/pub/cosmos/bin/).
Windows users need [cosmos bash](https://justine.lol/cosmo3/) shell too.

### Dependency Setup

Some dependencies are managed as git submodules with llamafile-specific patches.
Before building, you need to initialize and configure these dependencies:

```sh
make setup
```

The patches modify code in the git submodules. These modifications remain as local
changes in the submodule working directories.

`make setup` also downloads the [Cosmopolitan](https://github.com/jart/cosmopolitan/)
C compiler for you, saving it under the `.cosmocc` directory.

### Building

```sh
.cosmocc/4.0.2/bin/make -j8
sudo .cosmocc/4.0.2/bin/make install PREFIX=/usr/local
```

Build outputs will appear in the `./o` directory, e.g.:

- `./o/llama.cpp/server/llama-server`: the original llama.cpp inference server, compiled with cosmocc
- `o/llamafile/llamafile`: the llamafile executable, running both as a TUI and a server (with the `--server` flag)
- `o/third_party/zipalign/zipalign`: the zipalign tool used to bundle llamafile executable, model weights, and default args into llamafiles

> **NOTE**: Calling `make` should automatically run cosmocc's make when required.
If that does not happen for any reason, you can still directly run the one provided
by cosmocc: `.cosmocc/4.0.2/bin/make`.

### Testing

Optionally, you can verify the build with:

```sh
make check
```

This runs our unit tests to ensure everything is built correctly.

Some integration tests in `tests/integration` are available to test llamafile
with real models. Check the [README](/tests/integration/README.md) to learn how to run them. 

### Running llamafile

After the build, you can run llamafile as:

```sh
./o/llamafile/llamafile --model <gguf_model>
```

or just the llama.cpp server as:

```sh
./o/llamafile/llamafile --model <gguf_model> --server
```

or the llamafile CLI command as:

```sh
./o/llamafile/llamafile --model <gguf_model> --cli -p "Hello world"
```

## Documentation

There's a manual page for each of the llamafile programs installed when you
run `sudo make install`. Most commands will also display that information when
passing the `--help` flag.
