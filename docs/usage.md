# Usage Guide

This guide covers various ways to use llamafile, from basic command-line usage to creating your own custom llamafiles.

## Using llamafile with external weights

Even though our example llamafiles have the weights built-in, you don't *have* to use llamafile that way. Instead, you can download *just* the llamafile software (without any weights included) from our releases page. You can then use it alongside any external weights you may have on hand. External weights are particularly useful for Windows users because they enable you to work around Windows' 4GB executable file size limit.

For Windows users, here's an example for the Mistral LLM:

```sh
curl -L -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.17/llamafile-0.8.17
curl -L -o mistral.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
./llamafile.exe -m mistral.gguf
```

Windows users may need to change `./llamafile.exe` to `.\llamafile.exe` when running the above command.

## Command-line usage examples

Here are various examples of using llamafile from the command line.

### Example with Mistral for code generation

Here's an example for the Mistral command-line llamafile:

```sh
./mistral-7b-instruct-v0.2.Q5_K_M.llamafile --temp 0.7 -p '[INST]Write a story about llamas[/INST]'
```

### Example with WizardCoder-Python

And here is an example for WizardCoder-Python command-line llamafile:

```sh
./wizardcoder-python-13b.llamafile --temp 0 -e -r '```\n' -p '```c\nvoid *memcpy_sse2(char *dst, const char *src, size_t size) {\n'
```

Here's another example that generates code for a libc function:

```sh
llamafile \
  -m wizardcoder-python-13b-v1.0.Q8_0.gguf \
  --temp 0 -r '}\n' -r '```\n' \
  -e -p '```c\nvoid *memcpy(void *dst, const void *src, size_t size) {\n'
```

### Example with LLaVA for image analysis

Here's an example for the LLaVA command-line llamafile (with image support):

```sh
./llava-v1.5-7b-q4.llamafile --temp 0.2 --image lemurs.jpg -e -p '### User: What do you see?\n### Assistant:'
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

### Interactive chatbot example

Here's an example of how llamafile can be used as an interactive chatbot:

```sh
llamafile -m llama-65b-Q5_K.gguf -p '
The following is a conversation between a Researcher and their helpful AI assistant Digital Athena which is a large language model trained on the sum of human knowledge.
Researcher: Good morning.
Digital Athena: How can I help you today?
Researcher:' --interactive --color --batch_size 1024 --ctx_size 4096 \
--keep -1 --temp 0 --mirostat 2 --in-prefix ' ' --interactive-first \
--in-suffix 'Digital Athena:' --reverse-prompt 'Researcher:'
```

### HTML summarization example

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

### Using BNF grammar for predictable output

It's possible to use BNF grammar to enforce that the output is predictable and safe to use in your shell script. The simplest grammar would be `--grammar 'root ::= "yes" | "no"'` to force the LLM to only print to standard output either `"yes\n"` or `"no\n"`. Another example is if you wanted to write a script to rename all your image files, you could say:

```sh
llamafile -ngl 9999 --temp 0 \
    --image lemurs.jpg \
    -m llava-v1.5-7b-Q4_K.gguf \
    --mmproj llava-v1.5-7b-mmproj-Q4_0.gguf \
    --grammar 'root ::= [a-z]+ (" " [a-z]+)+' \
    -e -p '### User: What do you see?\n### Assistant: ' \
    --no-display-prompt 2>/dev/null |
  sed -e's/ /_/g' -e's/$/.jpg/'
```

Output:
```
a_baby_monkey_on_the_back_of_a_mother.jpg
```

## Running the HTTP server

Here's an example of how to run llama.cpp's built-in HTTP server. This example uses LLaVA v1.5-7B, a multimodal LLM that works with llama.cpp's recently-added support for image inputs.

```sh
llamafile -ngl 9999 \
  -m llava-v1.5-7b-Q8_0.gguf \
  --mmproj llava-v1.5-7b-mmproj-Q8_0.gguf \
  --host 0.0.0.0
```

The above command will launch a browser tab on your personal computer to display a web interface. It lets you chat with your LLM and upload images to it.

## Creating llamafiles

If you want to be able to just say:

```sh
./llava.llamafile
```

...and have it run the web server without having to specify arguments, then you can embed both the weights and a special `.args` file inside, which specifies the default arguments. First, let's create a file named `.args` which has this content:

```sh
-m
llava-v1.5-7b-Q8_0.gguf
--mmproj
llava-v1.5-7b-mmproj-Q8_0.gguf
--host
0.0.0.0
-ngl
9999
...
```

As we can see above, there's one argument per line. The `...` argument optionally specifies where any additional CLI arguments passed by the user are to be inserted. Next, we'll add both the weights and the argument file to the executable:

```sh
cp /usr/local/bin/llamafile llava.llamafile

zipalign -j0 \
  llava.llamafile \
  llava-v1.5-7b-Q8_0.gguf \
  llava-v1.5-7b-mmproj-Q8_0.gguf \
  .args

./llava.llamafile
```

Congratulations! You've just made your own LLM executable that's easy to share with your friends.

## Distribution

One good way to share a llamafile with your friends is by posting it on Hugging Face. If you do that, then it's recommended that you mention in your Hugging Face commit message what git revision or released version of llamafile you used when building your llamafile. That way everyone online will be able to verify the provenance of its executable content. If you've made changes to the llama.cpp or cosmopolitan source code, then the Apache 2.0 license requires you to explain what changed. One way you can do that is by embedding a notice in your llamafile using `zipalign` that describes the changes, and mention it in your Hugging Face commit.

## Documentation

There's a manual page for each of the llamafile programs installed when you run `sudo make install`. The command manuals are also typeset as PDF files that you can download from our GitHub releases page. Lastly, most commands will display that information when passing the `--help` flag.
