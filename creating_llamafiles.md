# Creating a llamafile

A llamafile bundles the llamafile executable, model weights, and a set of
default arguments into a single self-contained file using the
[APE](https://justine.lol/ape.html) (Actually Portable Executable) format,
which supports ZIP as a container for extra data. If you have already
downloaded a llamafile, you can inspect its contents with
`unzip -vl <filename.llamafile>` (or on Windows, rename it to `.zip` and
open it in your ZIP GUI).

## Prerequisites

llamafile uses [zipalign](https://github.com/jart/zipalign) to bundle files
into the executable. It is included as a git submodule and built alongside
llamafile, so if you have already compiled llamafile you have the `zipalign`
executable in the `o//third_party/zipalign` folder. To build it on its own:

```sh
make o//third_party/zipalign
```

> [!NOTE]
> The zipalign tool referenced here is **not** the
> [Android zipalign](https://developer.android.com/tools/zipalign). See the
> GitHub repo above for an in-depth description and up-to-date code.

## What you need

- **The llamafile executable** — download a prebuilt binary from the
  [releases page](https://github.com/mozilla-ai/llamafile/releases), or build
  from source following
  [these instructions](source_installation.md).

- **Model weights in GGUF format** — download from Hugging Face
  ([search here](https://huggingface.co/models?library=gguf)), or use weights
  already on disk from
  [another application](quickstart.md#running-llamafile-with-models-downloaded-by-third-party-applications).

- **A `.args` file** — specifies default arguments (at minimum, the model
  path so it loads automatically).

## Examples

### TUI, text-only

Let's see how this works in practice with a simple, text-only language
model, e.g. Qwen3-0.6B:

- [Search](https://huggingface.co/models?library=gguf&sort=trending&search=qwen3-0.6b) for the model weights in GGUF format
(for the sake of this example we'll download [these](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) with Q8 quantization)
- Create a file named `.args` with the following content:

```text
-m
/zip/Qwen3-0.6B-Q8_0.gguf
-fa
on
--temp
0.6
--top-k
20
--top-p
0.95
--min-p
0
--presence-penalty
1.5
-c
40960
-n
32768
--no-context-shift
--no-mmap
...
```

> [!NOTE]
> There is one argument per line. Most arguments are optional — the model
> name is the only required one (the above replicates the parameters suggested
> [here](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)). The `/zip/` path
> prefix is required whenever referencing a file packaged inside the llamafile.
> The `...` token is replaced with any additional CLI arguments the user passes
> at runtime.

- Copy the llamafile executable and run zipalign to embed the weights and args:

```bash
cp o//llamafile/llamafile Qwen3-0.6B-Q8.llamafile

o//third_party/zipalign/zipalign -j0 \
  Qwen3-0.6B-Q8.llamafile \
  Qwen3-0.6B-Q8_0.gguf \
  .args

./Qwen3-0.6B-Q8.llamafile
```

Congratulations, you've just made your own LLM executable that's easy to
share with your friends!

Your new llamafile will start loading the Qwen model in the TUI. You can also
run it as a web server with:

```bash
./Qwen3-0.6B-Q8.llamafile --server
```

### Server, multimodal

Now, let us build another llamafile running a multimodal model served
via HTTP. If you want to be able to just say:

```bash
./llava.llamafile
```

...and have it run the web server without having to specify arguments,
embed both the weights and the following `.args` file
(weights used in this example are downloaded from [here](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf)):

```text
-m
/zip/llava-v1.6-mistral-7b.Q8_0.gguf
--mmproj
/zip/mmproj-model-f16.gguf
--server
--host
0.0.0.0
-ngl
9999
--no-mmap
...
```

Next, add both the weights and the argument file to the executable:

```bash
cp o//llamafile/llamafile llava.llamafile

o//third_party/zipalign/zipalign -j0 \
  llava.llamafile \
  llava-v1.6-mistral-7b.Q8_0.gguf \
  mmproj-model-f16.gguf \
  .args

./llava.llamafile
```

## Distribution

One good way to share a llamafile with your friends is by posting it on
Hugging Face. If you do that, then it's recommended that you mention in
your Hugging Face commit message what git revision or released version
of llamafile you used when building your llamafile. That way everyone
online will be able verify the provenance of its executable content. If
you've made changes to the llama.cpp or cosmopolitan source code, then
the Apache 2.0 license requires you to explain what changed. One way you
can do that is by embedding a notice in your llamafile using `zipalign`
that describes the changes, and mention it in your Hugging Face commit.
