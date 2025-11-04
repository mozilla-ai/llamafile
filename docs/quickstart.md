# Quickstart

The easiest way to try llamafile for yourself is to download our example llamafile for the [LLaVA](https://llava-vl.github.io/) model (license: [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), [OpenAI](https://openai.com/policies/terms-of-use)). LLaVA is a new LLM that can do more than just chat; you can also upload images and ask it questions about them. With llamafile, this all happens locally; no data ever leaves your computer.

## Running your first llamafile

1. Download [llava-v1.5-7b-q4.llamafile](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile?download=true) (4.29 GB).

2. Open your computer's terminal.

3. If you're using macOS, Linux, or BSD, you'll need to grant permission for your computer to execute this new file. (You only need to do this once.)

```sh
chmod +x llava-v1.5-7b-q4.llamafile
```

4. If you're on Windows, rename the file by adding ".exe" on the end.

5. Run the llamafile. e.g.:

```sh
./llava-v1.5-7b-q4.llamafile
```

6. Your browser should open automatically and display a chat interface. (If it doesn't, just open your browser and point it at http://localhost:8080)

7. When you're done chatting, return to your terminal and hit `Control-C` to shut down llamafile.

**Having trouble? See the [Troubleshooting](troubleshooting.md) page.**

## JSON API Quickstart

When llamafile is started, in addition to hosting a web UI chat server at <http://127.0.0.1:8080/>, an [OpenAI API](https://platform.openai.com/docs/api-reference/chat) compatible chat completions endpoint is provided too. It's designed to support the most common OpenAI API use cases, in a way that runs entirely locally. We've also extended it to include llama.cpp specific features (e.g. mirostat) that may also be used. For further details on what fields and endpoints are available, refer to both the [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create) and the [llamafile server README](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints).

### Curl API Client Example

The simplest way to get started using the API is to copy and paste the following curl command into your terminal.

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

### Python API Client Example

If you've already developed your software using the [`openai` Python package](https://pypi.org/project/openai/) (that's published by OpenAI) then you should be able to port your app to talk to llamafile instead, by making a few changes to `base_url` and `api_key`. This example assumes you've run `pip3 install openai` to install OpenAI's client software, which is required by this example. Their package is just a simple Python wrapper around the OpenAI API interface, which can be implemented by any server.

```python
#!/usr/bin/env python3
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
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

## Other example llamafiles

We also provide example llamafiles for other models, so you can easily try out llamafile with different kinds of LLMs.

| Model                   | Size     | License                                                                                                                            | llamafile                                                                                                                                                                                      | other quants                                                                        |
| ---                     | ---      | ---                                                                                                                                | ---                                                                                                                                                                                            | ---                                                                                 |
| LLaMA 3.2 1B Instruct   | 1.11 GB  | [LLaMA 3.2](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/LICENSE)                                      | [Llama-3.2-1B-Instruct.Q6\_K.llamafile](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/blob/main/Llama-3.2-1B-Instruct.Q6_K.llamafile?download=true)                           | [See HF repo](https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile)       |
| LLaMA 3.2 3B Instruct   | 2.62 GB  | [LLaMA 3.2](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/blob/main/LICENSE)                                      | [Llama-3.2-3B-Instruct.Q6\_K.llamafile](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile/blob/main/Llama-3.2-3B-Instruct.Q6_K.llamafile?download=true)                           | [See HF repo](https://huggingface.co/Mozilla/Llama-3.2-3B-Instruct-llamafile)       |
| LLaMA 3.1 8B Instruct   | 5.23 GB  | [LLaMA 3.1](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile/blob/main/LICENSE)                                 | [Llama-3.1-8B-Instruct.Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile/resolve/main/Meta-Llama-3.1-8B-Instruct.Q4_K_M.llamafile?download=true)         | [See HF repo](https://huggingface.co/Mozilla/Meta-Llama-3.1-8B-Instruct-llamafile)  |
| Gemma 3 1B Instruct     | 1.32 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-1b-it.Q6\_K.llamafile](https://huggingface.co/Mozilla/gemma-3-1b-it-llamafile/resolve/main/google_gemma-3-1b-it-Q6_K.llamafile?download=true)                                         | [See HF repo](https://huggingface.co/Mozilla/gemma-3-1b-it-llamafile)               |
| Gemma 3 4B Instruct     | 3.50 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-4b-it.Q6\_K.llamafile](https://huggingface.co/Mozilla/gemma-3-4b-it-llamafile/resolve/main/google_gemma-3-4b-it-Q6_K.llamafile?download=true)                                         | [See HF repo](https://huggingface.co/Mozilla/gemma-3-4b-it-llamafile)               |
| Gemma 3 12B Instruct    | 7.61 GB  | [Gemma 3](https://ai.google.dev/gemma/terms)                                                                                       | [gemma-3-12b-it.Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/gemma-3-12b-it-llamafile/resolve/main/google_gemma-3-12b-it-Q4_K_M.llamafile?download=true)                                 | [See HF repo](https://huggingface.co/Mozilla/gemma-3-12b-it-llamafile)              |
| QwQ 32B                 | 7.61 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [Qwen\_QwQ-32B-Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/QwQ-32B-llamafile/resolve/main/Qwen_QwQ-32B-Q4_K_M.llamafile?download=true)                                                  | [See HF repo](https://huggingface.co/Mozilla/QwQ-32B-llamafile)                     |
| R1 Distill Qwen 14B     | 9.30 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [DeepSeek-R1-Distill-Qwen-14B-Q4\_K\_M](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Qwen-14B-llamafile/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.llamafile?download=true)        | [See HF repo](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Qwen-14B-llamafile)|
| R1 Distill Llama 8B     | 5.23 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [DeepSeek-R1-Distill-Llama-8B-Q4\_K\_M](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Llama-8B-llamafile/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.llamafile?download=true)        | [See HF repo](https://huggingface.co/Mozilla/DeepSeek-R1-Distill-Llama-8B-llamafile)|
| LLaVA 1.5               | 3.97 GB  | [LLaMA 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)                                                     | [llava-v1.5-7b-q4.llamafile](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile?download=true)                                                     | [See HF repo](https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile)               |
| Mistral-7B-Instruct v0.3| 4.42 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mistral-7b-instruct-v0.3.Q4\_0.llamafile](https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile/resolve/main/Mistral-7B-Instruct-v0.3.Q4_0.llamafile?download=true)               | [See HF repo](https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile)    |
| Granite 3.2 8B Instruct | 5.25 GB  | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [granite-3.2-8b-instruct-Q4\_K\_M.llamafile](https://huggingface.co/Mozilla/granite-3.2-8b-instruct-llamafile/resolve/main/granite-3.2-8b-instruct-Q4_K_M.llamafile?download=true)             | [See HF repo](https://huggingface.co/Mozilla/granite-3.2-8b-instruct-llamafile)     |
| Phi-3-mini-4k-instruct  | 7.67 GB  | [Apache 2.0](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/blob/main/LICENSE)                                    | [Phi-3-mini-4k-instruct.F16.llamafile](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/resolve/main/Phi-3-mini-4k-instruct.F16.llamafile?download=true)                        | [See HF repo](https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile)      |
| Mixtral-8x7B-Instruct   | 30.03 GB | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mixtral-8x7b-instruct-v0.1.Q5\_K\_M.llamafile](https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.llamafile?download=true)    | [See HF repo](https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile)  |
| OLMo-7B                 | 5.68 GB  | [Apache 2.0](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile/blob/main/LICENSE)                                              | [OLMo-7B-0424.Q6\_K.llamafile](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile/resolve/main/OLMo-7B-0424.Q6_K.llamafile?download=true)                                                   | [See HF repo](https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile)                |
| *Text Embedding Models* |          |                                                                                                                                    |                                                                                                                                                                                                |                                                                                     |
| E5-Mistral-7B-Instruct  | 5.16 GB  | [MIT](https://choosealicense.com/licenses/mit/)                                                                                    | [e5-mistral-7b-instruct-Q5_K_M.llamafile](https://huggingface.co/Mozilla/e5-mistral-7b-instruct/resolve/main/e5-mistral-7b-instruct-Q5_K_M.llamafile?download=true)                            | [See HF repo](https://huggingface.co/Mozilla/e5-mistral-7b-instruct)                |
| mxbai-embed-large-v1    | 0.7 GB   | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)                                                                      | [mxbai-embed-large-v1-f16.llamafile](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true)                              | [See HF Repo](https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile)        |
