# Getting Started with llamafile 

The easiest way to try it for yourself is to download our example llamafile
for the [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B/) model (license: 
[Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE)).
Qwen3.5 is a recent LLM that can do more than just chat; you can also upload
images and ask it questions about them. With llamafile, this all happens
locally: no data ever leaves your computer.

> **NOTE**: we chose this model because that's the smallest one we have
built a llamafile for, so most likely to work out-of-the-box for you.
Please let us know if you are still having issues with that! If, on the
other hand, you have powerful hardware and/or GPUs, [feel free to choose](example_llamafiles.md)
larger and more expressive models which should provide more accurate
responses.

1. Download [Qwen3.5-0.8B-Q8_0.llamafile](https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/Qwen3.5-0.8B-Q8_0.llamafile) (1.77 GB).

2. Open your computer's terminal.

    - If you're using macOS, Linux, or BSD, you'll need to grant permission
for your computer to execute this new file. (You only need to do this
once.)

      ```sh
      chmod +x Qwen3.5-0.8B-Q8_0.llamafile
      ```

    - If you're on Windows, rename the file by adding ".exe" on the end.

5. Run the llamafile. e.g.:

    ```sh
    ./Qwen3.5-0.8B-Q8_0.llamafile
    ```

6. A chat interface will open in the terminal window. That's it: you can immediately
start writing. You can also upload an image by using the `/upload` command and specifying the path to the image, or write
`/help` to see the available commands).

7. Note that when llamafile is running, you can also chat with it using
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s Web UI: just open a
browser window and connect to <http://localhost:8080/>. 

8. When you're done chatting, `Control-C` to shut down llamafile.


**Having trouble? See the [Troubleshooting](troubleshooting.md) page.**

## JSON API Quickstart

As llamafile relies on llama.cpp for serving models, it comes with all its
features. When it is started, in addition to hosting a web UI chat server at 
<http://127.0.0.1:8080/>, it also exposes an endpoint compatible with
[OpenAI API](https://platform.openai.com/docs/api-reference/chat)
and [Anthropic's Messages API](https://platform.claude.com/docs/en/api/messages).
For further details on what fields and endpoints are available, refer to the
APIs documentation and llama.cpp server's
[README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

<details>
<summary>Curl API Client Example</summary>

The simplest way to get started using the API is to copy and paste the
following curl command into your terminal.

```shell
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
  "model": "LLaMA_CPP",
  "messages": [
      {
          "role": "system",
          "content": "You are LLAMAfile, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
      },
      {
          "role": "user",
          "content": "Write a limerick about python exceptions"
      }
    ]
}' | python3 -c '
import json
import sys
json.dump(json.load(sys.stdin), sys.stdout, indent=2)
print()
'
```

The response that's printed should look like the following:

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "In the world of Python, where magic breaks and errors occur,\nA script fails when it should not have failed.\nWith a `KeyError`, I can't access the key,\nSo I tell you to use the `except` clause!"
      }
    }
  ],
  "created": 1773659260,
  "model": "Qwen3.5-0.8B-Q8_0.gguf",
  "system_fingerprint": "b1773565177-7f5ee5496",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 52,
    "prompt_tokens": 49,
    "total_tokens": 101
  },
  "id": "chatcmpl-KOqwN6C0oRzINGZuFqZ95bU1iPfc6RFO",
  "timings": {
    "cache_n": 0,
    "prompt_n": 49,
    "prompt_ms": 54.944,
    "prompt_per_token_ms": 1.1213061224489795,
    "prompt_per_second": 891.8171228887594,
    "predicted_n": 52,
    "predicted_ms": 405.856,
    "predicted_per_token_ms": 7.804923076923076,
    "predicted_per_second": 128.1242608215722
  }
}
```
</details>

<details>
<summary>Python API Client example</summary>

If you've already developed your software using the [`openai` Python
package](https://pypi.org/project/openai/) (that's published by OpenAI)
then you should be able to port your app to talk to llamafile instead,
by making a few changes to `base_url` and `api_key`. This example
assumes you've run `pip3 install openai` to install OpenAI's client
software, which is required by this example. Their package is just a
simple Python wrapper around the OpenAI API interface, which can be
implemented by any server.

```python
#!/usr/bin/env python3
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)
completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "Write a limerick about python exceptions"}
    ]
)
print(completion.choices[0].message)
```

The above code will return a Python object like this:

```python
ChatCompletionMessage(content="A script that crashes like a ghost,\nWhen it tries to solve the problem deep and fast.\nThe error message pops up in a bright light,\nAnd tells us what's wrong when we try to fix it.", refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)
```
</details>

## Using llamafile with external weights

Even though our example llamafiles have the weights built-in, you don't
*have* to use llamafile that way. Instead, you can download *just* the
llamafile software (without any weights included) from our releases page.
You can then use it alongside any external weights you may have on hand.
External weights are particularly useful for Windows users because they
enable you to work around Windows' 4GB executable file size limit.

For Windows users, here's an example for the gpt-oss LLM (whose size is >12GB):

```sh
curl -L -o llamafile.exe https://huggingface.co/mozilla-ai/llamafile_0.10.0/resolve/main/llamafile_0.10.0
curl -L -o gpt-oss.gguf https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q5_K_S.gguf
./llamafile.exe -m gpt-oss.gguf
```

Windows users may need to change `./llamafile.exe` to `.\llamafile.exe` when running the above command.


## Running llamafile with models downloaded by third-party applications

This section answers the question *"I already have a model downloaded locally by application X, can I use it with llamafile?"*. The general answer is "yes, as long as those models are locally stored in GGUF format" but its implementation can be more or less hacky depending on the application. A few examples (tested on a Mac) follow.

### LM Studio
[LM Studio](https://lmstudio.ai/) stores downloaded models in `~/.cache/lm-studio/models/lmstudio-community`, in subdirectories with the same name of the models, minus their quantization level. So if you have downloaded e.g. the `gpt-oss-20b-MXFP4.gguf` file, it will be stored in `~/.cache/lm-studio/models/lmstudio-community/gpt-oss-20b-GGUF/` and you can run llamafile as follows:

```bash
llamafile -m ~/.cache/lm-studio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf
```

### Ollama

When you download a new model with [ollama](https://ollama.com), all its metadata will be stored in a manifest file under `~/.ollama/models/manifests/registry.ollama.ai/library/`. The directory and manifest file name are the model name as returned by `ollama list`. For instance, for `llama3:latest` the manifest file will be named `.ollama/models/manifests/registry.ollama.ai/library/llama3/latest`.

The manifest maps each file related to the model (e.g. GGUF weights, license, prompt template, etc) to a sha256 digest. The digest corresponding to the element whose `mediaType` is `application/vnd.ollama.image.model` is the one referring to the model's GGUF file.

Each sha256 digest is also used as a filename in the `~/.ollama/models/blobs` directory (if you look into that directory you'll see *only* those sha256-* filenames). This means you can directly run llamafile by passing the sha256 digest as the model filename. So if e.g. the `llama3:latest` GGUF file digest is `sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29`, you can run llamafile as follows:

```bash
cd ~/.ollama/models/blobs
llamafile -m sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29
```
**Note** that Ollama's GGUF weights do not always work with llama.cpp (see e.g. [here](https://forums.developer.nvidia.com/t/nemotron-3-super-120b-on-gb10-llama-cpp-sm-121-build-ollama-gguf-incompatibility-fix/363459)), 
and as llamafile relies on llama.cpp this trick might not always work for you.