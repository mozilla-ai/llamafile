Developing on llamafile requires a modern version of the GNU `make`
command (called `gmake` on some systems), `sha256sum` (otherwise `cc`
will be used to build it), `wget` (or `curl`), and `unzip` available at
[https://cosmo.zip/pub/cosmos/bin/](https://cosmo.zip/pub/cosmos/bin/).
Windows users need [cosmos bash](https://justine.lol/cosmo3/) shell too.

### Dependency Setup

Some dependencies are managed as git submodules with
llamafile-specific patches. Before building, you need to initialize and
configure these dependencies:

```sh
make setup
```

The patches modify dependencies. These modifications remain as local
changes in the submodule working directories.

### Building

```sh
make -j8
sudo make install PREFIX=/usr/local
```

Here's an example of how to generate code for a libc function using the
llama.cpp command line interface, utilizing WizardCoder-Python-13B
weights:

```sh
llamafile \
  -m wizardcoder-python-13b-v1.0.Q8_0.gguf \
  --temp 0 -r '}\n' -r '```\n' \
  -e -p '```c\nvoid *memcpy(void *dst, const void *src, size_t size) {\n'
```

Here's a similar example that instead utilizes Mistral-7B-Instruct
weights for prose composition:

```sh
llamafile -ngl 9999 \
  -m mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  -p '[INST]Write a story about llamas[/INST]'
```

Here's an example of how llamafile can be used as an interactive chatbot
that lets you query knowledge contained in training data:

```sh
llamafile -m llama-65b-Q5_K.gguf -p '
The following is a conversation between a Researcher and their helpful AI assistant Digital Athena which is a large language model trained on the sum of human knowledge.
Researcher: Good morning.
Digital Athena: How can I help you today?
Researcher:' --interactive --color --batch_size 1024 --ctx_size 4096 \
--keep -1 --temp 0 --mirostat 2 --in-prefix ' ' --interactive-first \
--in-suffix 'Digital Athena:' --reverse-prompt 'Researcher:'
```

Here's an example of how you can use llamafile to summarize HTML URLs:

```sh
(
  echo '[INST]Summarize the following text:'
  links -codepage utf-8 \
        -force-html \
        -width 500 \
        -dump https://www.poetryfoundation.org/poems/48860/the-raven |
    sed 's/   */ /g'
  echo '[/INST]'
) | llamafile -ngl 9999 \
      -m mistral-7b-instruct-v0.2.Q5_K_M.gguf \
      -f /dev/stdin \
      -c 0 \
      --temp 0 \
      -n 500 \
      --no-display-prompt 2>/dev/null
```

Here's how you can use llamafile to describe a jpg/png/gif/bmp image:

```sh
llamafile -ngl 9999 --temp 0 \
  --image ~/Pictures/lemurs.jpg \
  -m llava-v1.5-7b-Q4_K.gguf \
  --mmproj llava-v1.5-7b-mmproj-Q4_0.gguf \
  -e -p '### User: What do you see?\n### Assistant: ' \
  --no-display-prompt 2>/dev/null
```

It's possible to use BNF grammar to enforce the output is predictable
and safe to use in your shell script. The simplest grammar would be
`--grammar 'root ::= "yes" | "no"'` to force the LLM to only print to
standard output either `"yes\n"` or `"no\n"`. Another example is if you
wanted to write a script to rename all your image files, you could say:

```sh
llamafile -ngl 9999 --temp 0 \
    --image lemurs.jpg \
    -m llava-v1.5-7b-Q4_K.gguf \
    --mmproj llava-v1.5-7b-mmproj-Q4_0.gguf \
    --grammar 'root ::= [a-z]+ (" " [a-z]+)+' \
    -e -p '### User: What do you see?\n### Assistant: ' \
    --no-display-prompt 2>/dev/null |
  sed -e's/ /_/g' -e's/$/.jpg/'
a_baby_monkey_on_the_back_of_a_mother.jpg
```

Here's an example of how to run llama.cpp's built-in HTTP server. This
example uses LLaVA v1.5-7B, a multimodal LLM that works with llama.cpp's
recently-added support for image inputs.

```sh
llamafile -ngl 9999 \
  -m llava-v1.5-7b-Q8_0.gguf \
  --mmproj llava-v1.5-7b-mmproj-Q8_0.gguf \
  --host 0.0.0.0
```

The above command will launch a browser tab on your personal computer to
display a web interface. It lets you chat with your LLM and upload
images to it.


## Documentation

There's a manual page for each of the llamafile programs installed when you
run `sudo make install`. The command manuals are also typeset as PDF
files that you can download from our GitHub releases page. Lastly, most
commands will display that information when passing the `--help` flag.
