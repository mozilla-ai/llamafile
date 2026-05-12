llamafile 0.10.0 has been a work in progress for a while. Now that we are merging
its code with main, we want to leave this document available to document both the
reasons and the process behind it.

Everything started with the goal of replicating a cosmopolitan llama.cpp build from scratch,
so we could get the best of two worlds. On the one hand, some of the characteristic
features of llamafiles, that is portability across different systems and architectures
and the possibility of bundling model weights within llamafile executables. On the
other hand, the features and the model support made available by the most recent
versions of llama.cpp.

We realise that what makes a llamafile is not just an APE executable, so before
merging this code with main we wanted to bring back other of its features into the
new build. We believe there's still work to do, but now that the main features are
there we can let you play with a more modern llamafile and directly ask you what
you'd like to see the most in its future versions.

Older builds (and llamafiles built on them) will still be available, check out our
[releases](https://github.com/mozilla-ai/llamafile/releases) and our 
[Example Llamafiles](/docs/example_llamafiles.md) page.

# Updates

Here are the features we brought into our development branch before merging with main.
Most of them were brought in from previous versions of llamafile, and all credit goes
to their original authors <3. Some (including new build for easier sync with upstream
llama.cpp, mtmd API support, intregration tests, skill docs, HTTP chat client for combined
mode) are new.

20260317
- Updates to [skill documents](https://github.com/mozilla-ai/llamafile/pull/886)
- Added [whisper](https://github.com/mozilla-ai/llamafile/pull/880)
- Added support for [chat, cli, server](https://github.com/mozilla-ai/llamafile/pull/896) modalities
- [Updated llama.cpp](https://github.com/mozilla-ai/llamafile/pull/901) to 7f5ee54 (with support for qwen3.5 models)
- Added [integration tests](https://github.com/mozilla-ai/llamafile/pull/906)
- Added [`--image` support to CLI](https://github.com/mozilla-ai/llamafile/pull/912)


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
[our support docs](https://docs.mozilla.ai/llamafile/reference/support#gpu-support) for more info.
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

# What's missing

- GPU support for Windows (and for whisperfile)
- stable diffusion (the code is there, but has not been ported to the new build format yet)
- some features triggered by extra arguments in CLI mode
- pledge() SECCOMP sandboxing
- localscore
- llamafiler for embeddings (we rolled back to llama.cpp's embeddings endpoint instead)
- ... please help us track if there's anything missing you wish to see in the new build!