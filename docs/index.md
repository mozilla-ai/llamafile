# llamafile

> **We want to hear from you!**
Mozilla.ai recently adopted the llamafile project, and we're planning an approach for codebase modernization. Please share what you find most valuable about llamafile and what would make it more useful for your work.
[Read more via the blog](https://blog.mozilla.ai/llamafile-returns/) and add your voice to the discussion [here](https://github.com/mozilla-ai/llamafile/discussions/809).


[![ci status](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml/badge.svg)](https://github.com/Mozilla-Ocho/llamafile/actions/workflows/ci.yml)<br/>
[![](https://dcbadge.vercel.app/api/server/YuMNeuKStr)](https://discord.gg/YuMNeuKStr)<br/><br/>

<img src="images/llamafile-640x640.png" width="320" height="320"
     alt="[line drawing of llama animal head in front of slightly open manilla folder filled with files]">

**llamafile lets you distribute and run LLMs with a single file. ([announcement blog post](https://hacks.mozilla.org/2023/11/introducing-llamafile/))**

Our goal is to make open LLMs much more
accessible to both developers and end users. We're doing that by
combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one
framework that collapses all the complexity of LLMs down to
a single-file executable (called a "llamafile") that runs
locally on most computers, with no installation.<br/><br/>

<a href="https://builders.mozilla.org/"><img src="images/mozilla-logo-bw-rgb.png" width="150"></a><br/>
llamafile is a <a href="https://builders.mozilla.org/">Mozilla Builders</a> project.<br/><br/>



## How llamafile works

A llamafile is an executable LLM that you can run on your own
computer. It contains the weights for a given open LLM, as well
as everything needed to actually run that model on your computer.
There's nothing to install or configure (with a few caveats, discussed
in subsequent sections of this document).

This is all accomplished by combining llama.cpp with Cosmopolitan Libc,
which provides some useful capabilities:

1. llamafiles can run on multiple CPU microarchitectures. We
added runtime dispatching to llama.cpp that lets new Intel systems use
modern CPU features without trading away support for older computers.

2. llamafiles can run on multiple CPU architectures. We do
that by concatenating AMD64 and ARM64 builds with a shell script that
launches the appropriate one. Our file format is compatible with WIN32
and most UNIX shells. It's also able to be easily converted (by either
you or your users) to the platform-native format, whenever required.

3. llamafiles can run on six OSes (macOS, Windows, Linux,
FreeBSD, OpenBSD, and NetBSD). If you make your own llama files, you'll
only need to build your code once, using a Linux-style toolchain. The
GCC-based compiler we provide is itself an Actually Portable Executable,
so you can build your software for all six OSes from the comfort of
whichever one you prefer most for development.

4. The weights for an LLM can be embedded within the llamafile.
We added support for PKZIP to the GGML library. This lets uncompressed
weights be mapped directly into memory, similar to a self-extracting
archive. It enables quantized weights distributed online to be prefixed
with a compatible version of the llama.cpp software, thereby ensuring
its originally observed behaviors can be reproduced indefinitely.

5. Finally, with the tools included in this project you can create your
*own* llamafiles, using any compatible model weights you want. You can
then distribute these llamafiles to other people, who can easily make
use of them regardless of what kind of computer they have.


## Licensing

While the llamafile project is Apache 2.0-licensed, our changes
to llama.cpp are licensed under MIT (just like the llama.cpp project
itself) so as to remain compatible and upstreamable in the future,
should that be desired.

The llamafile logo on this page was generated with the assistance of DALL·E 3.


[![Star History Chart](https://api.star-history.com/svg?repos=Mozilla-Ocho/llamafile&type=Date)](https://star-history.com/#Mozilla-Ocho/llamafile&Date)
