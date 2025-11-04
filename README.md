# llamafile

> **We want to hear from you!**
Mozilla.ai recently adopted the llamafile project, and we're planning an approach for codebase modernization. Please share what you find most valuable about llamafile and what would make it more useful for your work.
[Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).

[![ci status](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml/badge.svg)](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml)
[![](https://dcbadge.vercel.app/api/server/YuMNeuKStr)](https://discord.gg/YuMNeuKStr)

<img src="llamafile/llamafile-640x640.png" width="320" height="320"
     alt="line drawing of llama animal head in front of slightly open manilla folder filled with files">

**llamafile lets you distribute and run LLMs with a single file.** ([announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/))

Our goal is to make open LLMs much more accessible to both developers and end users. We're doing that by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most computers, with no installation.

<a href="https://builders.mozilla.org/"><img src="llamafile/mozilla-logo-bw-rgb.png" width="150"></a>

llamafile is a <a href="https://builders.mozilla.org/">Mozilla Builders</a> project.

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

📚 **Full documentation is available in the [docs/](docs/) folder or online at [mozilla-ai.github.io/llamafile](https://mozilla-ai.github.io/llamafile/)**

- [Quickstart Guide](docs/quickstart.md) - Get up and running
- [Usage Guide](docs/usage.md) - Command-line examples and creating llamafiles
- [Installation](docs/installation.md) - Building from source
- [GPU Support](docs/gpu-support.md) - Enable GPU acceleration
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Technical Details](docs/technical-details.md) - How llamafile works under the hood

## Key Features

- **Cross-platform** - Runs on macOS, Windows, Linux, FreeBSD, OpenBSD, and NetBSD
- **Cross-architecture** - Works on AMD64 and ARM64 processors
- **No installation** - Single executable file, no dependencies
- **GPU acceleration** - Supports Apple Metal, NVIDIA, and AMD GPUs
- **OpenAI compatible API** - Drop-in replacement for OpenAI's API
- **Embedded weights** - Model weights can be bundled in the executable

## Supported Platforms

**Operating Systems:** Linux 2.6.18+, macOS 23.1.0+, Windows 10+, FreeBSD 13+, NetBSD 9.2+, OpenBSD 7+

**CPUs:** AMD64 (with AVX), ARM64 (ARMv8a+)

**GPUs:** Apple Metal, NVIDIA (via CUDA), AMD (via ROCm)

## Security

llamafile uses pledge() and SECCOMP sandboxing to limit what the process can access after startup. This helps protect your system even if there's a vulnerability in the LLM server. Sandboxing is enabled by default and can be disabled with `--unsecure` if needed.

## Contributing

See [RELEASE.md](docs/release.md) for information about the release process.

## License

While the llamafile project is Apache 2.0-licensed, our changes to llama.cpp are licensed under MIT (just like the llama.cpp project itself) so as to remain compatible and upstreamable in the future, should that be desired.

## A Note About Models

The example llamafiles provided in this documentation should not be interpreted as endorsements or recommendations of specific models, licenses, or data sets on the part of Mozilla.

---

[![Star History Chart](https://api.star-history.com/svg?repos=Mozilla-Ocho/llamafile&type=Date)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
