# How to make a Whisperfile

Whisperfile is designed to be a single-file solution for speech-to-text.
This tutorial will explain how you can merge the whisperfile executable
and OpenAI's model weights into a unified executable.

We'll be using Cosmopolitan Libc's "ZipOS" read-only filesystem to achieve
this. Because whisperfile executables are valid ZIP files at the same time,
you can embed model weights directly inside the binary, and the runtime
will expose them under the `/zip/...` path prefix. We'll also
use the `.args` file convention to bake in default arguments so users don't
need to pass flags manually.

## Prerequisites

First, build the `zipalign` tool, which is used to embed files into the
executable without breaking its ZIP structure:

```bash
.cosmocc/4.0.2/bin/make -j8 o//third_party/zipalign
```

Next, either obtain a prebuilt `whisperfile` executable from the
[GitHub releases page](https://github.com/mozilla-ai/llamafile/releases),
or build one from source:

```bash
.cosmocc/4.0.2/bin/make -j8 o//whisperfile

# copy it with a more specific name
cp o//whisperfile/whisperfile whisper-tiny
```

## Instructions

Download the model weights you want to bundle. For this tutorial we'll use
the tiny q5\_1 quantized weights:

```bash
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin
```

Embed the weights inside your whisperfile. The `-0` flag disables PKZIP
DEFLATE compression, which isn't beneficial for binary weights files:

```bash
o//third_party/zipalign/zipalign -0 whisper-tiny ggml-tiny.en-q5_1.bin
```

Your weights are now embedded. You can verify with `unzip -vl whisper-tiny`.
Cosmopolitan Libc exposes embedded files under the synthetic `/zip/...`
directory, so a file named `ggml-tiny.en-q5_1.bin` is accessible at
`/zip/ggml-tiny.en-q5_1.bin`:

```bash
./whisper-tiny -m /zip/ggml-tiny.en-q5_1.bin -f whisper.cpp/samples/jfk.wav
```

(`jfk.wav` is a sample audio clip included in the repository.)

It's now safe to delete the original weights file:

```bash
rm -f ggml-tiny.en-q5_1.bin
```

To avoid passing `-m /zip/ggml-tiny.en-q5_1.bin` every time, create a
`.args` file that specifies default arguments. Each argument goes on its
own line — no shell quoting needed:

```text
-m
/zip/ggml-tiny.en-q5_1.bin
...
```

The `...` at the end is a special token that gets replaced with any
additional arguments the user passes at runtime.

Embed the `.args` file into your whisperfile:

```bash
o//third_party/zipalign/zipalign whisper-tiny .args
rm -f .args
```

You now have a self-contained whisperfile. Run it with just an audio file:

```bash
./whisper-tiny -f whisper.cpp/samples/jfk.wav
```
