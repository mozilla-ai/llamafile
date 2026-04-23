# llamafile

<img src="docs/images/llamafile-640x640.png" width="320" height="320"
     alt="[line drawing of llama animal head in front of slightly open manilla folder filled with files]">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mozilla-ai/llamafile/blob/main/LICENSE)
[![ci status](https://github.com/mozilla-ai/llamafile/actions/workflows/ci.yml/badge.svg)](https://github.com/mozilla-ai/llamafile/actions/workflows/ci.yml)
[![Based on llama.cpp](https://img.shields.io/badge/llama.cpp-7f5ee54-orange.svg)](https://github.com/ggml-org/llama.cpp/commit/7f5ee54)
[![Based on whisper.cpp](https://img.shields.io/badge/whisper.cpp-2eeeba5-green.svg)](https://github.com/ggml-org/whisper.cpp/commit/2eeeba5)
[![Discord](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr)
[![Mozilla Builders](https://img.shields.io/badge/Builders-6E6E6E?logo=mozilla&logoColor=white&labelColor=4A4A4A)](https://builders.mozilla.org/)

**llamafile lets you distribute and run LLMs with a single file.**

llamafile is a [Mozilla Builders](https://builders.mozilla.org/) project (see its [announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/)), now revamped by [Mozilla.ai](https://www.mozilla.ai/open-tools/llamafile). 

Our goal is to make open LLMs much more
accessible to both developers and end users. We're doing that by
combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one
framework that collapses all the complexity of LLMs down to
a single-file executable (called a "llamafile") that runs
locally on most operating systems and CPU archiectures, with no installation.

llamafile also includes **[whisperfile](https://docs.mozilla.ai/llamafile/whisperfile)**, a single-file speech-to-text tool built on [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and the same Cosmopolitan packaging. It supports transcription and translation of audio files across all the same platforms, with no installation required.


## v0.10.0

**llamafile versions starting from 0.10.0 use a new build system**, aimed at keeping our code more easily 
aligned with the latest versions of llama.cpp. This means they support more recent models and functionalities,
but at the same time they might be missing some of
the features you were accustomed to (check out [this doc](README_0.10.0.md) for a high-level description of what has been done). If you liked
the "classic experience" more, you will always be able to access the previous versions from our
[releases](https://github.com/mozilla-ai/llamafile/releases) page. Our pre-built llamafiles always
show which version of the server they have been bundled with ([0.9.* example](https://huggingface.co/mozilla-ai/llava-v1.5-7b-llamafile), [0.10.* example](https://huggingface.co/mozilla-ai/llamafile_0.10.0)), so you will always know
which version of the software you are downloading.


> **We want to hear from you!**
Whether you are a new user or a long-time fan, please share what you find most valuable about llamafile and what would make it more useful for you.
[Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).


## Quick Start

Download and run your first llamafile in minutes:

```sh
# Download an example model (Qwen3.5 0.8B)
curl -LO https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile

# Make it executable (macOS/Linux/BSD)
chmod +x Qwen3.5-0.8B-Q8_0.llamafile

# Run it
./Qwen3.5-0.8B-Q8_0.llamafile
```

We chose this model because that's the smallest one we have
built a llamafile for, so most likely to work out-of-the-box for you.
If you have powerful hardware and/or GPUs, [feel free to choose](https://docs.mozilla.ai/llamafile/getting-started/example_llamafiles)
larger and more expressive models which should provide more accurate
responses.

**Windows users:** Rename the file to add `.exe` extension before running.

## Documentation

Check the full documentation at [docs.mozilla.ai/llamafile](https://docs.mozilla.ai/llamafile), or directly jump into one of the following subsections:

- [Quickstart](https://docs.mozilla.ai/llamafile/getting-started/quickstart)
- [Example llamafiles](https://docs.mozilla.ai/llamafile/getting-started/example_llamafiles)
- [Running a llamafile](https://docs.mozilla.ai/llamafile/using-llamafile/running_llamafile)
- [Creating llamafiles](https://docs.mozilla.ai/llamafile/using-llamafile/creating_llamafiles)
- [Source installation](https://docs.mozilla.ai/llamafile/using-llamafile/source_installation)
- [Technical details](https://docs.mozilla.ai/llamafile/reference/technical_details)
- [Supported Systems](https://docs.mozilla.ai/llamafile/reference/support)
- [Troubleshooting](https://docs.mozilla.ai/llamafile/reference/troubleshooting)
- [Whisperfile](https://docs.mozilla.ai/llamafile/whisperfile)


## Licensing

While the llamafile project is Apache 2.0-licensed, our changes
to llama.cpp and whisper.cpp are licensed under MIT (just like the projects
themselves) so as to remain compatible and upstreamable in the future,
should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.


[![Star History Chart](https://api.star-history.com/svg?repos=Mozilla-Ocho/llamafile&type=Date)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
