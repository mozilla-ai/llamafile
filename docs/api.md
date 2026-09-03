# API server

When llamafile runs in [server mode](running_llamafile.md#running-llamafile-in-server-mode), it exposes the HTTP APIs inherited from
`llama-server`, llama.cpp's server component. The default base address is
`http://127.0.0.1:8080`.

The server provides:

- OpenAI-compatible APIs under `/v1`.
- An Anthropic-compatible Messages API.
- Native llama.cpp endpoints for lower-level server features.
- An experimental internal endpoint for the bundled Web UI's
  [built-in tools](built-in-tools.md).

Note that the supported fields and behavior come from the llama.cpp version
bundled into a particular llamafile release and can change between releases.

## Address and authentication

Use `--host` and `--port` to change the listening address and port.
`--api-prefix` prepends a path to every endpoint. For example,
`--api-prefix /llama` changes `/v1/chat/completions` to
`/llama/v1/chat/completions`.

Configure authentication with `--api-key` or `--api-key-file`. Protected
requests accept either header:

```text
Authorization: Bearer YOUR_API_KEY
```

```text
X-Api-Key: YOUR_API_KEY
```

The second form is also compatible with clients that send Anthropic's
`x-api-key` header; HTTP header names are case-insensitive. When no API key is
configured, omit the header.

> [!WARNING]
> As a general rule, avoid making your server accessible to more clients than
> you actually need. Start with the default host binding, which listens on
> localhost only, and change it only if you need to access the server from
> another machine. When you do this, configure `--api-key` or
> `--api-key-file` so reachable clients still need authentication.
>
> CORS only affects which browser origins can read responses. It is not
> authentication.

## OpenAI-compatible APIs

### Chat Completions

`POST /v1/chat/completions` accepts OpenAI-style chat messages, e.g.:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a limerick about llamas."}
    ]
  }'
```

The `model` field is required by the OpenAI-style schema even when only one
model is loaded. Clients commonly use a placeholder such as `local-model`. Set
`--alias` if you want a stable model name in requests and responses.

OpenAI SDKs read their base URL and API key from environment variables:

```sh
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=no-key-required
```

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

The SDK reads both variables automatically. It requires a non-empty API key
value even when the llamafile server does not require authentication. If you
set an API key via the `--api-key` argument as described above, set it as your
`OPENAI_API_KEY`.

### Responses

`POST /v1/responses` accepts OpenAI Responses-style input. The server converts
it internally to a Chat Completions request:

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "instructions": "You are a helpful assistant.",
    "input": "Explain why llamas hum.",
    "max_output_tokens": 256
  }'
```

`input` can be a string or an array of supported input items. Streaming is
available with `"stream": true`. `previous_response_id` is not supported;
send the conversation history needed for the next request in `input`.

### Other OpenAI-compatible endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | List model metadata and capabilities. |
| `POST /v1/completions` | Generate text from a raw prompt. |
| `POST /v1/embeddings` | Create embeddings with a pooling-enabled model. |
| `POST /v1/rerank` | Rank documents with a reranker model. `/v1/reranking` is an alias. |
| `POST /v1/audio/transcriptions` | Transcribe audio with a compatible model. |
| `POST /v1/chat/completions/input_tokens` | Count tokens for a Chat Completions request. |
| `POST /v1/responses/input_tokens` | Count tokens for a Responses request. |
| `POST /v1/chat/completions/control` | Control an in-progress chat completion that enabled realtime reasoning control. |

Model and build capabilities determine whether specialized endpoints such as
embeddings, reranking, multimodal input, and transcription can succeed.

## Anthropic-compatible API

### Messages

`POST /v1/messages` accepts Anthropic Messages-style requests, including a
system prompt, content blocks, stop sequences, streaming, and tools:

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "max_tokens": 512,
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Explain why llamas hum."}
    ]
  }'
```

The main supported request fields are:

| Field | Description |
| --- | --- |
| `model` | Model identifier. Required by the Anthropic-style schema. |
| `messages` | Conversation messages. |
| `max_tokens` | Maximum generated tokens. Defaults to `4096` when omitted. |
| `system` | System prompt as a string or text-block array. |
| `temperature`, `top_p`, `top_k` | Sampling controls. |
| `stop_sequences` | Strings that stop generation. |
| `stream` | Return server-sent events when `true`. |
| `tools`, `tool_choice` | Anthropic-style function definitions and selection mode. |

`POST /v1/messages/count_tokens` accepts the same conversation shape and
returns its input-token count without generating a response:

```json
{
  "input_tokens": 10
}
```

## Tool calling

The OpenAI- and Anthropic-compatible APIs can ask a model to call functions,
but they do not execute those functions. A client must run the agent loop:

1. Send tool definitions and conversation messages to the model.
2. Inspect the response for requested tool calls.
3. Validate the arguments and ask the user for any required approval.
4. Execute each approved call in the client or through `POST /tools`.
5. Return each result in the selected API's tool-result format.
6. Call the model again and repeat until it returns a normal answer.

`--tools` is needed only when the client will execute llama.cpp's enabled
[built-in tools](built-in-tools.md) through `/tools`; client-defined functions
do not require it.

The three compatible APIs use different tool-call shapes:

| API | Send tool definitions as | Requested call appears as | Return the result as |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `tools` array with nested `function` objects | assistant `message.tool_calls[]` | next request message with `role: "tool"` and the matching `tool_call_id` |
| OpenAI Responses | `tools` array of function objects | `output[]` item with `type: "function_call"` | next request `input[]` item with `type: "function_call_output"` |
| Anthropic Messages | `tools` array with `input_schema` | assistant content block with `type: "tool_use"` | next user message content block with `type: "tool_result"` |

To execute one of llama.cpp's enabled built-in tools, call `POST /tools` with
the tool name and arguments selected by the model. This internal endpoint is
experimental and does not display the Web UI's permission prompt. See
[Built-in local tools](built-in-tools.md) for tool discovery, security notes,
and the current `/tools` interface.

## Native llama.cpp endpoints

Native endpoints expose lower-level features that are not part of the OpenAI
or Anthropic schemas:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Report readiness. `/v1/health` is an alias. |
| `GET /props`, `POST /props` | Read or update server properties. |
| `GET /models` | List models and their load state. |
| `POST /completion` | Generate from a raw prompt with native llama.cpp options. |
| `POST /tokenize`, `POST /detokenize` | Convert between text and token IDs. |
| `POST /apply-template` | Apply the active chat template without inference. |
| `POST /infill` | Perform fill-in-the-middle code completion. |
| `POST /embedding` | Create embeddings in the native response format. |
| `POST /rerank` | Rank documents in the native response format. |
| `GET /slots`, `POST /slots/:id_slot` | Inspect slots and manage prompt-cache state. |
| `GET /lora-adapters`, `POST /lora-adapters` | Inspect or change loaded LoRA adapter scales. |
| `GET /metrics` | Return Prometheus metrics when enabled with `--metrics`. |
| `GET /tools`, `POST /tools` | List and invoke experimental built-in or MCP tools. |

Consult the bundled server help and upstream server documentation before
building automation around native endpoints, which change more frequently than
the compatibility endpoints.

## Errors

API errors normally use an OpenAI-style envelope:

```json
{
  "error": {
    "code": 400,
    "message": "Description of the invalid request",
    "type": "invalid_request_error"
  }
}
```

Common error types include `invalid_request_error`, `authentication_error`,
`permission_error`, `not_found_error`, `server_error`, and
`not_supported_error`. The HTTP status code carries the same general category.

## Further reference

This page is an overview of the endpoints most useful to llamafile users. For
all accepted generation parameters and lower-level response fields, see the
[llama.cpp server README](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
and check `llamafile --server --help` for the options available in the current
executable.
