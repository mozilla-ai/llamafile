The easiest way to try it for yourself is to download our example
llamafile for the [LLaVA](https://llava-vl.github.io/) model (license: [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/),
[OpenAI](https://openai.com/policies/terms-of-use)). LLaVA is a new LLM that can do more
than just chat; you can also upload images and ask it questions
about them. With llamafile, this all happens locally; no data
ever leaves your computer.

1. Download [llava-v1.5-7b-q4.llamafile](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile?download=true) (4.29 GB).

2. Open your computer's terminal.

3. If you're using macOS, Linux, or BSD, you'll need to grant permission
for your computer to execute this new file. (You only need to do this
once.)

```sh
chmod +x llava-v1.5-7b-q4.llamafile
```

4. If you're on Windows, rename the file by adding ".exe" on the end.

5. Run the llamafile. e.g.:

```sh
./llava-v1.5-7b-q4.llamafile
```

6. Your browser should open automatically and display a chat interface.
(If it doesn't, just open your browser and point it at http://localhost:8080)

7. When you're done chatting, return to your terminal and hit
`Control-C` to shut down llamafile.

**Having trouble? See the [Troubleshooting](troubleshooting.md) page.**

### JSON API Quickstart

When llamafile is started, in addition to hosting a web
UI chat server at <http://127.0.0.1:8080/>, an [OpenAI
API](https://platform.openai.com/docs/api-reference/chat) compatible
chat completions endpoint is provided too. It's designed to support the
most common OpenAI API use cases, in a way that runs entirely locally.
We've also extended it to include llama.cpp specific features (e.g.
mirostat) that may also be used. For further details on what fields and
endpoints are available, refer to both the [OpenAI
documentation](https://platform.openai.com/docs/api-reference/chat/create)
and the [llamafile server
README](https://github.com/mozilla-ai/llamafile/blob/HEAD/llamafile/server/README.md#api-endpoints).

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
   "choices" : [
      {
         "finish_reason" : "stop",
         "index" : 0,
         "message" : {
            "content" : "There once was a programmer named Mike\nWho wrote code that would often choke\nHe used try and except\nTo handle each step\nAnd his program ran without any hike.",
            "role" : "assistant"
         }
      }
   ],
   "created" : 1704199256,
   "id" : "chatcmpl-Dt16ugf3vF8btUZj9psG7To5tc4murBU",
   "model" : "LLaMA_CPP",
   "object" : "chat.completion",
   "usage" : {
      "completion_tokens" : 38,
      "prompt_tokens" : 78,
      "total_tokens" : 116
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
ChatCompletionMessage(content='There once was a programmer named Mike\nWho wrote code that would often strike\nAn error would occur\nAnd he\'d shout "Oh no!"\nBut Python\'s exceptions made it all right.', role='assistant', function_call=None, tool_calls=None)
```

</details>


## New v2 Server

We have a new server that has a better web gui. It also implements
OpenAI API compatible endpoints, including embeddings. It's designed to
be more reliable. It's better able to recycle context windows across
multiple slots. To try it, run:

```
llamafile --server --v2 --help
llamafile --server --v2
```



## Using llamafile with external weights

Even though our example llamafiles have the weights built-in, you don't
*have* to use llamafile that way. Instead, you can download *just* the
llamafile software (without any weights included) from our releases page.
You can then use it alongside any external weights you may have on hand.
External weights are particularly useful for Windows users because they
enable you to work around Windows' 4GB executable file size limit.

For Windows users, here's an example for the Mistral LLM:

```sh
curl -L -o llamafile.exe https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.17/llamafile-0.8.17
curl -L -o mistral.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
./llamafile.exe -m mistral.gguf
```

Windows users may need to change `./llamafile.exe` to `.\llamafile.exe`
when running the above command.



## Running llamafile with models downloaded by third-party applications

This section answers the question *"I already have a model downloaded locally by application X, can I use it with llamafile?"*. The general answer is "yes, as long as those models are locally stored in GGUF format" but its implementation can be more or less hacky depending on the application. A few examples (tested on a Mac) follow.

### LM Studio
[LM Studio](https://lmstudio.ai/) stores downloaded models in `~/.cache/lm-studio/models`, in subdirectories with the same name of the models (following HuggingFace's `account_name/model_name` format), with the same filename you saw when you chose to download the file.

 So if you have downloaded e.g. the `llama-2-7b.Q2_K.gguf` file for `TheBloke/Llama-2-7B-GGUF`, you can run llamafile as follows:

```
cd ~/.cache/lm-studio/models/TheBloke/Llama-2-7B-GGUF
llamafile -m llama-2-7b.Q2_K.gguf
```

### Ollama

When you download a new model with [ollama](https://ollama.com), all its metadata will be stored in a manifest file under `~/.ollama/models/manifests/registry.ollama.ai/library/`. The directory and manifest file name are the model name as returned by `ollama list`. For instance, for `llama3:latest` the manifest file will be named `.ollama/models/manifests/registry.ollama.ai/library/llama3/latest`.

The manifest maps each file related to the model (e.g. GGUF weights, license, prompt template, etc) to a sha256 digest. The digest corresponding to the element whose `mediaType` is `application/vnd.ollama.image.model` is the one referring to the model's GGUF file.

Each sha256 digest is also used as a filename in the `~/.ollama/models/blobs` directory (if you look into that directory you'll see *only* those sha256-* filenames). This means you can directly run llamafile by passing the sha256 digest as the model filename. So if e.g. the `llama3:latest` GGUF file digest is `sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29`, you can run llamafile as follows:

```
cd ~/.ollama/models/blobs
llamafile -m sha256-00e1317cbf74d901080d7100f57580ba8dd8de57203072dc6f668324ba545f29
```
