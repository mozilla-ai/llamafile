This branch is a work in progress.

It started with the goal of replicating a cosmopolitan llama.cpp build from scratch,
so we could get the best of two worlds. On the one hand, some of the characteristic
features of llamafiles, that is portability across different systems and architectures
and the possibility of bundling model weights within llamafile executables. On the
other hand, the features and the model support made available by the most recent
versions of llama.cpp.

We realise that what makes a llamafile is not just an APE executable, so we are now
bringing back other of its features. We have started with the llamafile TUI and the
latest PR introduces dynamic GPU support (starting with Metal first). 

While we are adding llamafile features back, some parts of the build might fail
(for instance, the current build does not support whisperfile and stable diffusion
yet... but [we are working on it](https://github.com/mozilla-ai/llamafile/pull/880)!).
If you want the more stable build with older code you can still use the code
(and the instructions) you can find in the main branch, while here you'll find the 
most recent stuff.

# Building llamafile v0.10.0(alpha)

The code in this branch allows you to build a llamafile from a recent version
of llama.cpp (commit f47edb8, Jan 8 2026), and has the following features:

- llamafile TUI, with support for multimodal models (add images with the `/upload` command)
- all the features already present in llama.cpp server, including tool calling support
and Anthropic messages API
- Metal GPU support
- CUDA GPU support (tested on Linux)
- Optimizations for different CPU architectures

### 1. Clone the Repository

```
git clone https://github.com/mozilla-ai/llamafile.git
cd llamafile
``` 

### 2. Checkout the New Build Branch

Make sure you are working in this branch:

```
git checkout new_build_wip
```

### 3. Initialize Submodules and Apply Patches

```
make setup
```

This initializes the git submodules (llama.cpp, whisper.cpp, stable-diffusion.cpp) and applies llamafile-specific patches to them.

### 4. Download the Cosmopolitan Toolchain

```
build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
```

### 5. Build Everything

```
.cosmocc/4.0.2/bin/make -j8
```

Build outputs will appear in the `./o` directory, e.g.:

- `./o/llama.cpp/server/llama-server`: the original llama.cpp inference server, compiled with cosmocc
- `o/llamafile/llamafile`: the llamafile executable, running both as a TUI and a server (with the `--server` flag)
- `o/third_party/zipalign/zipalign`: the zipalign tool used to bundle llamafile executable, model weights, and default args into llamafiles

### 6. Verify the Build (Optional)

```
.cosmocc/4.0.2/bin/make check
```

This runs the test suite to ensure everything built correctly.


### 7. Run llamafile

After the build, you can run the llamafile TUI as

```
./o/llamafile/llamafile --model <gguf_model>
```

or the llama.cpp server as

```
./o/llamafile/llamafile --model <gguf_model> --server
```

> [!NOTE]
> If you want, you can build just the vanilla llama.cpp server as an APE with:
> 
> ```
> make -j8 o//llama.cpp/server/llama-server
> ```

# What's new

20260219
- Added [CPU optimizations](https://github.com/mozilla-ai/llamafile/pull/868)
- Fixed misc issues
  - server [timing out](https://github.com/mozilla-ai/llamafile/pull/876)
  - [mmap errors](https://github.com/mozilla-ai/llamafile/pull/882) when loading bundled models
  - [think mode in TUI](https://github.com/mozilla-ai/llamafile/pull/885)
- [Added "skill docs"](https://github.com/mozilla-ai/llamafile/pull/886) to be used with AI assistants

[20260202](https://github.com/mozilla-ai/llamafile/discussions/871)
- Added zipalign as a GitHub [submodule](https://github.com/mozilla-ai/llamafile/pull/848) (so we can get the latest updates from Justine’s repo)
- Brought back [cuda support](https://github.com/mozilla-ai/llamafile/pull/859) on Linux
- Added support for the [mtmd API](https://github.com/mozilla-ai/llamafile/pull/852) in the TUI (so you can now directly access modern multimodal models from the llamafile chat)
- Tested new llamafiles running models trained for tool calling (e.g. Qwen3, gpt-oss-20b) and multimodal models such as llava 1.6, Qwen3-VL and Ministral 3

[20251218](https://github.com/mozilla-ai/llamafile/discussions/845)
- added Metal support: GPU on MacOS ARM64 is supported by compiling a small module
using the Xcode Command Line Tools, which need to be installed. Check our docs at
https://mozilla-ai.github.io/llamafile/support/#gpu-support for more info.
- Metal works both in llamafile (called either as TUI or with the --server flag)
and in llama-server.

20251215
- added TUI support: you can now directly chat with the chosen LLM from
the terminal, or run the llama.cpp server using the `--server` parameter
- simplified build by removing all tools/deps except those required by
the new llamafile code (they will be added back in as soon as we reintroduce
functionalities)

20251209
- added BUILD.mk so we can do without cmake
- build works with cosmocc 4.0.2
- dependencies are all taken from llama.cpp/vendor directory
- building now works both on linux and mac

20251208
- updated to llama.cpp commit dbc15a79672e72e0b9c1832adddf3334f5c9229c

20251124
- first version, relying on cmake for the build

