Developing on llamafile requires a modern version of the GNU `make`
command (called `gmake` on some systems), `sha256sum` (otherwise `cc`
will be used to build it), `wget` (or `curl`), and `unzip` available at
[https://cosmo.zip/pub/cosmos/bin/](https://cosmo.zip/pub/cosmos/bin/).
Windows users need [cosmos bash](https://justine.lol/cosmo3/) shell too.

### Dependency Setup

Some dependencies are managed as git submodules with
llamafile-specific patches. Before building, you need to initialize and
configure these dependencies:

```sh
make setup
```

The patches modify dependencies. These modifications remain as local
changes in the submodule working directories.

### Building

```sh
make -j8
sudo make install PREFIX=/usr/local
```

## Documentation

There's a manual page for each of the llamafile programs installed when you
run `sudo make install`. The command manuals are also typeset as PDF
files that you can download from our GitHub releases page. Lastly, most
commands will display that information when passing the `--help` flag.
