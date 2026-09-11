# Built-in local tools

**llama.cpp's built-in tools are experimental** and can change between
releases. In llamafile, they are primarily intended for the bundled Web UI,
though a custom agent loop can use them too.

The Web UI oversees the following:

- making the enabled tool definitions available to the model
- asking for permission before executing any requested tool
- running the tool through the server
- sending the result back to the model

The regular chat completion API does not automatically execute arbitrary tool
calls.

> [!WARNING]
> Built-in tools act with the operating-system permissions of the llamafile
> process. Some can read or modify local files or execute arbitrary commands.
> They are disabled by default. Enable only the tools you need, and avoid
> exposing a tool-enabled server to an untrusted network.

## Enable built-in tools

Pass `--tools` in server mode. Use `all` to enable every built-in tool, or a
comma-separated list to enable selected tools (see the complete list below):

```sh
# Enable every built-in tool
llamafile -m model.gguf --server --tools all

# Enable a read-only subset
llamafile -m model.gguf --server \
  --tools read_file,file_glob_search,grep_search,get_datetime
```

For a model-bundled llamafile, use its filename in place of `llamafile -m
model.gguf`:

```sh
./ModelName.llamafile --server --tools read_file,grep_search
```

Open the Web UI at <http://localhost:8080/> after the model loads.

`--agent` enables all built-in tools and the experimental MCP proxy. Prefer
`--tools` when you only need local tools, because it exposes a smaller surface.

## Available tools

| Tool | Access | Purpose |
| --- | --- | --- |
| `read_file` | Read | Read all or part of a file. |
| `file_glob_search` | Read | Find files whose paths match a glob. |
| `grep_search` | Read | Search files for matching text. |
| `exec_shell_command` | Execute | Run a host shell command. |
| `write_file` | Write | Create or replace a file. |
| `edit_file` | Write | Replace selected text in an existing file. |
| `get_datetime` | Read | Get the server's current date and time. |
| `get_info` | Read | Get the runtime operating system and working directory. |

Tool support in llama.cpp is experimental and subject to change. To inspect the
exact tool definitions and parameter signatures exposed by a running server,
call `GET /tools`:

```sh
curl http://127.0.0.1:8080/tools
```

If the server was started with `--api-prefix`, prepend that prefix here too.
When `--api-key` or `--api-key-file` is configured, `/tools` uses the same
authentication as the rest of the API server.

## Security and sandbox behavior

Built-in tools are not a separate security boundary. Once they are enabled, any
client that can reach `/tools` and satisfy any configured API key can invoke an
enabled tool directly. The Web UI permission prompt is part of the interactive
browser flow, not server-side authorization.

Unless `--cors-origins` is set explicitly, enabling built-in tools changes the
allowed browser origin from `*` to localhost. CORS affects browser access only;
it is not authentication.

As a general rule, expose the server to no more clients than necessary. Keep
the default loopback binding when possible, and configure `--api-key` before
making a tool-enabled server reachable from another machine. For the broader
sandbox model, see [Security](security.md).

llamafile's sandbox changes which tools can succeed:

| Runtime mode | Effect on built-in tools |
| --- | --- |
| CPU-only `--server` on Linux or OpenBSD | The default sandbox permits reads but blocks file writes, process creation, and command execution, so `write_file`, `edit_file`, and `exec_shell_command` fail. |
| CPU-only `--server --confine-reads` | The same restrictions apply, and reads are limited to the executable and configured model, adapter, media, and static-file directories. |
| GPU server | The sandbox is skipped because GPU drivers need system access that the sandbox cannot allow. Enabled tools run with the process's normal permissions. |
| Default combined TUI and server mode | The sandbox is skipped because the in-process TUI needs an HTTP client connection. |
| `--unsecure` or an operating system without supported sandboxing | The sandbox is disabled or unavailable, so enabled tools run with the process's normal permissions. |

Enabling tools in an otherwise sandboxed server relaxes the network restriction
so the server can make outbound connections, but it does not add permission to
write files or execute programs.

## Use with API clients

The OpenAI-compatible Chat Completions and Responses APIs and the
Anthropic-compatible Messages API can ask a model to call tools, but clients
still execute those calls and return the results. See [API server: Tool
calling](api.md#tool-calling).

## Internal `/tools` API

The bundled Web UI uses `GET /tools` to discover tool definitions and `POST
/tools` to invoke them. This interface is experimental and intended primarily
for the Web UI, not as a stable downstream application API. Most users should
let the Web UI or another agent loop call it rather than invoking it manually.

If no built-in or MCP tools are enabled, `GET /tools` and `POST /tools` return
HTTP `403`.
