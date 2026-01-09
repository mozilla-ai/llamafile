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
yet). If you want the more stable build with older code you can still use the code
(and the instructions) you can find in the main branch, while here you'll find the 
most recent stuff.

# Building llamafile v0.10.0(alpha)

The code currently contains the minimum set of changes required to rebuild llama.cpp
with cosmopolitan, and use it together with the llamafile TUI and Metal GPU support.
The code is based on llama.cpp commit dbc15a79672e72e0b9c1832adddf3334f5c9229c
(Dec, 6, 2025).

First of all, run `make setup` to pull from the git submodules and apply the required
patches.

Then, run the following to build the llamafile APE:

```
make -j8
```

> [!NOTE]
> If you get the "please use bin/make from cosmocc.zip rather than old xcode make" message
> (likely to happen on a Mac), run the following to download cosmocc into the `.cosmocc/4.0.2`
> directory:
>
> ```
> build/download-cosmocc.sh .cosmocc/4.0.2 4.0.2 85b8c37a406d862e656ad4ec14be9f6ce474c1b436b9615e91a55208aced3f44
> ```
>
> You can then either add the cosmocc binaries dir to your path or directly run:
>
> ```
> .cosmocc/4.0.2/bin/make -j8
> ```

You can then run the llamafile TUI as

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

20251218
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

