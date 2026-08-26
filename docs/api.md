# API server

When llamafile runs in server mode, it exposes the HTTP APIs inherited from
`llama-server`, llama.cpp's server component. The default base address is
`http://127.0.0.1:8080`.

```sh
llamafile -m model.gguf --server
```

A model-bundled llamafile can be started directly:

```sh
./ModelName.llamafile --server
```

The server provides:

- OpenAI-compatible APIs under `/v1`.
- An Anthropic-compatible Messages API.
- Native llama.cpp endpoints for lower-level server features.
- An experimental internal endpoint for the bundled Web UI's
  [built-in tools](built-in-tools.md).

Compatibility is practical rather than exact. The supported fields and
behavior come from the llama.cpp version bundled into a particular llamafile
release and can change between releases.

## Address and authentication

Use `--host` and `--port` to change the listening address. `--api-prefix`
prepends a path to every endpoint. For example, `--api-prefix /llama` changes
`/v1/chat/completions` to `/llama/v1/chat/completions`.

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
> CORS controls which browser origins can read responses. It is not
> authentication. Keep the default loopback binding or configure an API key
> before making the server reachable from another machine.

## OpenAI-compatible APIs

### Chat Completions

`POST /v1/chat/completions` accepts OpenAI-style chat messages and supports
streaming with `"stream": true`:

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

The `model` value selects a model in router mode. With a single loaded model,
clients commonly use a placeholder such as `local-model`. Set a stable model
name with `--alias` when an application depends on it.

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
value even when the llamafile server does not require authentication. Replace
`no-key-required` with the configured llamafile API key when authentication is
enabled.

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

Model-side tool calling requires Jinja and works best with a model whose chat
template supports tools. Do not start the server with `--no-jinja`.
`--tools` is needed only when the client will execute llama.cpp's
[built-in tools](built-in-tools.md) through `/tools`; client-defined functions
do not require it.

### Convert built-in tool definitions

Each item returned by `GET /tools` contains an OpenAI Chat Completions
`definition`. Convert it as follows:

| API | Tool definition to send |
| --- | --- |
| Chat Completions | Use the complete `definition` object. |
| Responses | Flatten `definition.function` into `{ "type": "function", "name": ..., "description": ..., "parameters": ... }`. |
| Anthropic Messages | Use `{ "name": ..., "description": ..., "input_schema": definition.function.parameters }`. |

The application's tool descriptions do not have to come from `/tools`.
Client-side functions use the same schemas.

### OpenAI Chat Completions format

Send function definitions in the request's `tools` array:

```json
{
  "model": "local-model",
  "messages": [
    {"role": "user", "content": "What is today in UTC?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_datetime",
        "description": "Returns the current date and time in UTC",
        "parameters": {
          "type": "object",
          "properties": {
            "format": {"type": "string"}
          }
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

A requested function appears in the assistant message:

```json
{
  "finish_reason": "tool_calls",
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "get_datetime",
          "arguments": "{\"format\":\"%Y-%m-%d\"}"
        }
      }
    ]
  }
}
```

`function.arguments` is a JSON-encoded string. Parse and validate it before
execution. Send the model another request containing the original messages,
the complete assistant tool-call message, and the result with the matching ID:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "{\"result\":\"2026-08-26\"}"
}
```

Supported `tool_choice` strings are `auto`, `none`, and `required`.
`parallel_tool_calls` controls whether a compatible template can request more
than one function at once. With `"stream": true`, call fragments arrive in
`choices[].delta.tool_calls`; join the argument fragments before parsing them.

### OpenAI Responses format

Responses-style function definitions are not nested:

```json
{
  "model": "local-model",
  "input": "What is today in UTC?",
  "tools": [
    {
      "type": "function",
      "name": "get_datetime",
      "description": "Returns the current date and time in UTC",
      "parameters": {
        "type": "object",
        "properties": {
          "format": {"type": "string"}
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

The response places a requested function in `output`:

```json
{
  "type": "function_call",
  "call_id": "call_123",
  "name": "get_datetime",
  "arguments": "{\"format\":\"%Y-%m-%d\"}",
  "status": "completed"
}
```

After execution, call `/v1/responses` again with the relevant history,
function call, and output:

```json
{
  "model": "local-model",
  "input": [
    {"role": "user", "content": "What is today in UTC?"},
    {
      "type": "function_call",
      "call_id": "call_123",
      "name": "get_datetime",
      "arguments": "{\"format\":\"%Y-%m-%d\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_123",
      "output": "{\"result\":\"2026-08-26\"}"
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "get_datetime",
      "description": "Returns the current date and time in UTC",
      "parameters": {
        "type": "object",
        "properties": {"format": {"type": "string"}}
      }
    }
  ]
}
```

Only function tools are translated to Chat Completions. Unsupported
non-function tool types are skipped.

### Anthropic Messages format

Anthropic-style definitions use `input_schema`:

```json
{
  "model": "local-model",
  "max_tokens": 512,
  "messages": [
    {"role": "user", "content": "What is today in UTC?"}
  ],
  "tools": [
    {
      "name": "get_datetime",
      "description": "Returns the current date and time in UTC",
      "input_schema": {
        "type": "object",
        "properties": {
          "format": {"type": "string"}
        }
      }
    }
  ],
  "tool_choice": {"type": "auto"}
}
```

A requested function is a `tool_use` content block:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "call_123",
      "name": "get_datetime",
      "input": {"format": "%Y-%m-%d"}
    }
  ],
  "stop_reason": "tool_use"
}
```

Preserve that assistant message and append a user message containing the
result with the matching ID:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_123",
      "content": "{\"result\":\"2026-08-26\"}"
    }
  ]
}
```

Accepted tool-choice object types are `auto`, `any`, and `tool`. The current
compatibility conversion treats both `any` and a named `tool` choice as
requiring a tool call; it does not preserve a named-tool restriction.

With `"stream": true`, tool use arrives through `content_block_start`,
`content_block_delta` events containing `input_json_delta`, and
`content_block_stop`. The following `message_delta` has a `stop_reason` of
`tool_use`.

### Execute a built-in tool

To execute one of llama.cpp's enabled built-in tools, parse the model's call
into this internal request:

```sh
curl http://127.0.0.1:8080/tools \
  -H 'Content-Type: application/json' \
  -d '{
    "tool": "get_datetime",
    "params": {"format": "%Y-%m-%d"}
  }'
```

`POST /tools` is experimental and does not display the Web UI's permission
prompt. See [Built-in local tools](built-in-tools.md#internal-tools-api) for
its request, response, streaming, and security details.

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

Router mode adds model loading, unloading, download, and event endpoints under
`/models`. Consult the bundled server help and upstream server documentation
before building automation around native or router APIs, which change more
frequently than compatibility endpoints.

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
