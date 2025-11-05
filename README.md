# llamafile

> **We want to hear from you!**
Mozilla.ai recently adopted the llamafile project, and we're planning an approach for codebase modernization. Please share what you find most valuable about llamafile and what would make it more useful for your work.
[Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).


[![ci status](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml/badge.svg)](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml)<br/>
[![](https://dcbadge.vercel.app/api/server/YuMNeuKStr)](https://discord.gg/YuMNeuKStr)<br/><br/>

<img src="llamafile/llamafile-640x640.png" width="320" height="320"
     alt="[line drawing of llama animal head in front of slightly open manilla folder filled with files]">

**llamafile lets you distribute and run LLMs with a single file. ([announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/))**

Our goal is to make open LLMs much more
accessible to both developers and end users. We're doing that by
combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one
framework that collapses all the complexity of LLMs down to
a single-file executable (called a "llamafile") that runs
locally on most computers, with no installation.<br/><br/>

<a href="https://builders.mozilla.org/"><img src="llamafile/mozilla-logo-bw-rgb.png" width="150"></a><br/>
llamafile is a <a href="https://builders.mozilla.org/">Mozilla Builders</a> project.<br/><br/>

## Quick Start

Download and run your first llamafile in minutes:

```sh
# Download an example model (LLaVA 1.5 7B)
curl -LO https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile

# Make it executable (macOS/Linux/BSD)
chmod +x llava-v1.5-7b-q4.llamafile

# Run it (opens browser automatically)
./llava-v1.5-7b-q4.llamafile
```

**Windows users:** Rename the file to add `.exe` extension before running.

## Documentation

Check the full documentation in the [docs/](docs/) folder or online at [mozilla-ai.github.io/llamafile](https://mozilla-ai.github.io/llamafile/), or directly jump into one of the following subsections:

- [Quickstart](https://mozilla-ai.github.io/llamafile/quickstart/)
- [Supported Systems](https://mozilla-ai.github.io/llamafile/support/)
- [Example llamafiles](https://mozilla-ai.github.io/llamafile/example_llamafiles/)
- [Creating llamafiles](https://mozilla-ai.github.io/llamafile/creating_llamafiles/)
- [Source installation](https://mozilla-ai.github.io/llamafile/source_installation/)
- [Technical details](https://mozilla-ai.github.io/llamafile/technical_details/)
- [Security](https://mozilla-ai.github.io/llamafile/security/)
- [Troubleshooting](https://mozilla-ai.github.io/llamafile/troubleshooting/)


## Licensing

While the llamafile project is Apache 2.0-licensed, our changes
to llama.cpp are licensed under MIT (just like the llama.cpp project
itself) so as to remain compatible and upstreamable in the future,
should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.


[![Star History Chart](https://api.star-history.com/svg?repos=Mozilla-Ocho/llamafile&type=Date)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
