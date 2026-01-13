llamafile uses [zipalign](https://github.com/jart/zipalign) to bundle its main
executable together with model weights and a set of default arguments.

We are including zipalign as a git submodule and building it together with
llamafile, so if you managed to successfully compile llamafile you also have
the `zipalign` executable in the `o/third_party/zipalign` folder. If you want
to build zipalign alone, just run

```sh
make o//third_party/zipalign
```

> **NOTE:**
The zipalign tool we are referring to here is not the
[Android](https://developer.android.com/tools/zipalign) one! Please refer
to the GitHub repo above for an in-depth description and up-to-date code.

# Creating a llamafile

All files using the `.llamafile` extension follow the APE
([Actually Portable Executable](https://justine.lol/ape.html)) format,
which supports ZIP as a container format for extra data files. In the case
of llamafiles, this is used to package the main executable (the program
actually serving the models) together with model weights and a set
of arguments that are passed by default to the executable when it is run.
If you have already downloaded a llamafile, you can run
`unzip -vl <filename.llamafile>` to see its contents (or, if you are
running Windows, you can change the file extension to `.zip` and open
it in your default ZIP GUI).

If you want to create a llamafile from scratch, the things you need are:

- the llamafile executable, which you can either download as a binary
([here](https://huggingface.co/mozilla-ai/llamafile_0.10.0_alpha) is a
the repository holding the most recent version, 0.10.0 alpha) or build
from source following
[these instructions](https://mozilla-ai.github.io/llamafile/source_installation/);

- model weights in GGUF format, which you can download from huggingface
(you can start your search [here](https://huggingface.co/models?library=gguf)),
or you can find on your disk if you have already downloaded models using
[another application](https://mozilla-ai.github.io/llamafile/quickstart/#running-llamafile-with-models-downloaded-by-third-party-applications);

- a `.args` file containing some default arguments (typically at least the model name so it is automatically loaded).

## TUI, text-only
Let's see how this works in practice with a simple, text-only language
model, e.g. Qwen3-0.6B:

- [search](https://huggingface.co/models?library=gguf&sort=trending&search=qwen3-0.6b) for the model weights in GGUF format
(for the sake of this example we'll download [these](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) with Q8 quantization)
- create a file named `.args` with the following content:

```
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

> NOTE: there is one argument per line.
Most of the arguments are
optional, except the model name (in this case we are replicating the
parameters suggested [here](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)).
The `/zip/` path is always necessary when one refers to a file packaged
within the llamafile.
The `...` argument optionally specifies where any additional CLI arguments
passed by the user are to be inserted.

- copy the llamafile executable to the current directory and run zipalign
to add weights and args. Assuming both llamafile and zipalign have just
been built:

```
cp ./o/llamafile/llamafile Qwen3-0.6B-Q8.llamafile

./o/third_party/zipalign/zipalign -j0 \
  Qwen3-0.6B-Q8.llamafile \
  Qwen3-0.6B-Q8_0.gguf \
  .args

./Qwen3-0.6B-Q8.llamafile
```

Congratulations, you've just made your own LLM executable that's easy to
share with your friends!

Your new llamafile will start loading the Qwen model in the TUI. Note that
you can still run it as a web server if you want, with:

```
./Qwen3-0.6B-Q8.llamafile --server
```

## Server, multimodal
Now, let us build another llamafile running a multimodal model served
via HTTP. If you want to be able to just say:

```sh
./llava.llamafile
```

...and have it run the web server without having to specify arguments,
then you can embed both the weights and the following `.args` file
(weights used in this example are downloaded from [here](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf)):

```sh
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

```sh
cp ./o/llamafile/llamafile llava.llamafile

./o/third_party/zipalign/zipalign -j0 \
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
