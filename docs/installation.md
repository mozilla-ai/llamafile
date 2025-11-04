# Installation

This page covers how to build llamafile from source.

## Prerequisites

Developing on llamafile requires:

- A modern version of the GNU `make` command (called `gmake` on some systems)
- `sha256sum` (otherwise `cc` will be used to build it)
- `wget` (or `curl`)
- `unzip` available at [https://cosmo.zip/pub/cosmos/bin/](https://cosmo.zip/pub/cosmos/bin/)
- Windows users need [cosmos bash](https://justine.lol/cosmo3/) shell too

## Dependency Setup

Some dependencies are managed as git submodules with llamafile-specific patches. Before building, you need to initialize and configure these dependencies:

```sh
make setup
```

The patches modify dependencies. These modifications remain as local changes in the submodule working directories.

## Building

Build llamafile and install it to your system:

```sh
make -j8
sudo make install PREFIX=/usr/local
```

## Using built llamafile

After building and installing, you can use llamafile with any GGUF model weights. Here are some examples:

### Example with Mistral

Here's a similar example that utilizes Mistral-7B-Instruct weights for prose composition:

```sh
llamafile -ngl 9999 \
  -m mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  -p '[INST]Write a story about llamas[/INST]'
```

### Example with WizardCoder-Python

Here's an example of how to generate code for a libc function using the llama.cpp command line interface, utilizing WizardCoder-Python-13B weights:

```sh
llamafile \
  -m wizardcoder-python-13b-v1.0.Q8_0.gguf \
  --temp 0 -r '}\n' -r '```\n' \
  -e -p '```c\nvoid *memcpy(void *dst, const void *src, size_t size) {\n'
```

## Next steps

- See [Usage Guide](usage.md) for more examples
- See [Creating llamafiles](usage.md#creating-llamafiles) to package your own models
- See [GPU Support](gpu-support.md) to enable GPU acceleration
