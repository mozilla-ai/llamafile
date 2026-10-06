# agentfile — design

agentfile is a self-contained, multi-platform APE binary that runs a focused
agent loop from a single CLI invocation. The product promise: preconfigure an
agent (model + system prompt + tool allowlist) and share it as **one file**
that runs anywhere, llamafile-style.

Upstream llama.cpp ships the tools (server tools, isolation runtimes,
`permission_write` metadata) and a thin client-side agent loop in its web
UI. What it does not have, and what agentfile is for: a serverless one-shot
CLI, single-file packaged agents, session/trace recording, and KV-warm
packaging. agentfile is **the file format for shipping agents**, not
another harness.

## Decisions

1. **agent.cpp** provides the loop and the model wrapper: a submodule
   plus local patches (`agent.cpp.patches/`), no vendored fork, no
   upstream PRs for now.
2. **Tools come from llama.cpp's server tools** through one adapter.
   agentfile's own tools are written as `server_tool` subclasses, so the
   adapter is the only bridge.
3. **Interaction**: unix-tool one-shot by default, `-i/--interactive`
   opt-in. No TUI.
4. **Network**: `http_fetch` and `web_search` are built in, over HTTPS
   via mbedtls. No host `curl`: it would break the single-file promise.
5. **Packaging**: `.args` plus zipalign, as llamafile does.
6. **Records**: two existing formats, nothing invented: pi's session
   format and OTLP/JSON spans, written without the OTel SDK.
7. **API drift**: depending on server-tools internals is a conscious
   choice. We can pin, or start vendoring, if they churn too hard.

## Architecture

One translation unit, `agentfile.cpp`, plus headers:

- `server_tools_adapter.h`: `ServerToolbox` owns llama.cpp's
  `server_tools` registry plus the native tools; `ServerToolAdapter`
  exposes one tool as an `agent_cpp::Tool`. Every call goes through
  `server_tools::handle_post`, the handler behind llama-server's
  `POST /tools`: upstream drops the keys the model must not set (`cwd`,
  `runtime`, `resp_type`), routes the call to the `--tools-runtime`
  isolate and maps exceptions to error responses. The request body is
  also the json boundary: agent.cpp speaks `nlohmann::json`, server
  tools llama.cpp's `common_json`.
- `tools/`: native tools (`get_datetime`, `http_fetch`, `web_search`).
  They stay here because llama.cpp keeps its server tools to minimal I/O
  and shell tools, and sends date/time and web search to MCP or the web
  UI.
- `callbacks/`: agent.cpp callbacks, registered in this order:
  session recorder, confirmation, progress, iteration cap, trace, error
  recovery. The trace comes after the confirmation so tool spans
  measure execution, not the time the user spent deciding, and after the
  iteration cap so the call the cap refuses gets no chat span. Error
  recovery comes after the observers so they record a failed call as
  failed before it is handed back to the model.
- `util.h`: terminal escaping (`terminal_text`), ids, timestamps.

The toolset is built and checked before the model loads, so a bad
`--tools` or `--tools-runtime` value fails fast; so do unwritable
`--session`/`--trace` paths.

### agent.cpp patches

Pinned at upstream `63d23da` (v0.4.0). The build compiles `src/model.cpp`
and `src/agent.cpp` against llamafile's `llama.cpp/` (never agent.cpp's own
`deps/llama.cpp`). When a llama.cpp bump breaks them, fix it in the local
patch. The patches (`src_model.{cpp,h}.patch`) do the following, and all
of it could go upstream:

- **Sampling**: `common_sampler` with the chat template's tool-call
  grammar. A user GBNF rides along as `COMMON_GRAMMAR_TYPE_USER` (caveat:
  `ModelConfig::grammar_root` is ignored, so grammars must use "root").
  The template's thinking tags become the sampler's reasoning-budget tags,
  so a lazy tool-call grammar stays dormant inside `<think>`, as in
  llama-server.
- **Reasoning round-trip**: replies are parsed with reasoning extraction,
  so `<think>` blocks land in `reasoning_content`. Re-renders then stay
  byte-stable and the KV prefix survives turns; on hybrid models
  (Qwen3.5) a rewind is a full re-prefill. Rejected alternative: an
  append-only token stream that never re-renders history. It bloats the
  context with old reasoning, fights the template's seams, and breaks as
  soon as context trimming edits history.
- **Parse failures** don't end the loop: on refusal the content is kept,
  the reasoning is split off, and `<tool_call>{json}</tool_call>` blocks
  are lifted out by a heuristic scan. The scan runs only when the parser
  failed, the template has no tool-call grammar, or the reply ended
  inside a reasoning block it never closed. A `<tool_call>` the scan
  can't lift rethrows the parse error (exit 2). Recovered arguments keep
  the model's key order, so the next prompt matches the KV cache.
- **Tool-call ids**: every call gets one (`call_` + 8 hex) when the
  template's parser leaves it empty, as llama-server does, so every tool
  result names its call whichever callbacks run.
- **KV reuse**: on recurrent state, where `llama_memory_seq_rm` returns
  false, fall back to `llama_memory_clear` and a full re-decode.
  `tokenize()` always adds special tokens; keying BOS on an empty cache
  dropped it after the first call and broke the prefix match.
- `ModelConfig::enable_thinking`, for `--think`.

## CLI

- One-shot by default. The final answer goes to **stdout**; progress,
  confirmations and errors go to **stderr**, so the answer pipes cleanly.
  On a terminal the answer is escaped with `terminal_text`, so model text
  can't erase or restyle the tool lines above it; piped output is
  unchanged.
- No `-p` and stdin not a tty: stdin is the prompt.
- `-i/--interactive`: after each answer, prompt on stderr and read the
  next message from the terminal (`/dev/tty` when stdin was a pipe).
  Empty line or EOF ends the session. Turns reuse the same `messages`, so
  the KV prefix carries over. `--max-iterations` is a per-turn budget.
- Parsing is **last-wins**, and every mode flag has an inverse
  (`--confirm` for `--yes`, `--no-interactive`, `--no-think`), so
  packaged `.args` defaults (prepended by `cosmo_args`) can always be
  overridden.
- `--system-file PATH` reads the system prompt from a file, `/zip/` paths
  included.
- `-c/--ctx-size` defaults to `kDefaultCtx` (32768, enough for one full
  `http_fetch` result plus history); `0` means the model's native
  context. `-c` and `--max-iterations` take non-negative integers only.
- Exit codes: `0` ok, `1` usage error, `2` agent error, `3` other error,
  `4` `--max-iterations` reached.
  - Declining a confirmation is not an exit: the model gets
    `{"skipped": "user declined"}` and decides how to continue.
  - No answer at all (no terminal, or EOF at the prompt) ends the run
    (exit 2): every later guarded call would go unanswered too.
  - A failed tool call (a tool the model doesn't have, arguments that
    are not JSON) goes back to the model as an error result. Four
    failures in a row end the run (exit 2).

### stderr verbosity

- `--quiet`: nothing except, under `--yes`, a one-line `[tool --yes]
  args` audit record per destructive call.
- Default: a progress line before and after each tool call; `--think`
  reasoning, dimmed.
- `-v`: also truncated tool results, and llama.cpp's `common_log`
  warnings (such as a reply the chat parser refused). Those can quote
  model output and `common_log` prints them unescaped, hence only on
  request.

Model and tool text on stderr goes through `terminal_text`: under `--yes`
these lines are the only record of a call.

## Session records & tracing

### Session records: pi session-format v3 (JSONL)

`--session FILE` writes the conversation in [pi](https://pi.dev)'s session
format (`pi/packages/coding-agent/docs/session-format.md`, `version: 3`),
so sessions open in pi (`/import`), export to HTML, and can be compared
with pi's own runs.

- Header `{"type":"session","version":3,"id":…,"timestamp":…,"cwd":…}`,
  then tree-linked `user`, `assistant` and `toolResult` entries. The
  history is linear: each `parentId` is the previous entry.
- `provider` is `"agentfile"`, `model` the GGUF basename. Costs are zero,
  and so is token usage: agent.cpp doesn't expose per-call counts.
- A failed tool call is `isError: true` with the message, pi's own shape.
  The model gets the same error as `{"error": true, "tool": …,
  "message": …}`.
- Risk: an application format owned by Earendil, with migrations. We pin
  v3 and revisit when pi bumps.

### Spans: OTLP/JSON with GenAI semconv

`--trace FILE` writes one `ExportTraceServiceRequest` per line, OTLP's
JSON encoding. The OTel Collector ingests it with the `otlpjsonfile`
receiver.

- Span structure and attributes follow agent.cpp's `examples/tracing`:
  `invoke_agent` → `chat` / `execute_tool`, GenAI semantic conventions.
- No OTel SDK, protobuf or curl: OTLP/JSON needs only a JSON library.
  Span and trace ids are random hex.

## Tools

The set comes from the toolbox (`-h` lists it). Tools with
`permission_write` (writes, shell, network) ask for confirmation before
each call unless `--yes`. `--tools` takes `all`, `read_only` (every tool
without `permission_write`) or a list of names; an unknown or unavailable
name is an error, not a smaller toolset.

### http_fetch

GET only. The body is capped at `kMaxBody` (64 KB; the download stops
there, the result says `truncated`). 30 s timeout. Redirects are reported
as `redirect_to`, not followed, so every fetched URL is one the model asked
for and the confirmation prompt showed. The prompt also names the host the
URL connects to. The model gets the raw body: no HTML-to-text. A binary
body (by `Content-Type`, or a NUL byte when the type is missing or
`application/octet-stream`) stops the download and is not returned: the
result has status, headers and a note instead, since escaped binary would
overflow the context.

### web_search (SearXNG)

- Arguments: `query` (required), `page` → `pageno`, `time_range`
  (`day|month|year`), `categories`.
- The instance is **not** a tool argument, so the model can't choose
  where queries go: `--searxng-url URL` or `SEARXNG_URL`. With neither,
  the tool isn't registered (no dead tool in the schema).
- `GET {base}/search?format=json&…`. Returns one page of results as
  `title`, `url`, `snippet`, `engine`; snippets are capped at
  `kMaxSnippet` bytes.
- HTTP 403 gets a message saying the instance must enable `json` under
  `search: formats:` in `settings.yml` (public instances usually don't;
  self-hosted is the expected deployment).

## Security model

agentfile does **not** sandbox itself: there is no `pledge()`/`unveil()`
call in it, so every tool runs with the invoking user's permissions.

- The confirmation prompt for `permission_write` tools is the only gate.
  `--yes` removes it; under `--yes --quiet` each such call is still logged
  to stderr, one line each.
- Tool inputs go through llama-server's own handler, which drops
  model-supplied `runtime`, `cwd` and `resp_type`. The confirmation prompt
  shows the arguments as the tool will get them.
- Isolation, when wanted, comes from `--tools-runtime`, with llama-server's
  specs, checked at startup:
  - `docker-container:ID` attaches to a running container;
  - `docker:IMAGE` starts a container and stops it on exit (podman
    likewise);
  - `ssh:TARGET` runs on a remote host.

  That is the right boundary for an agent whose purpose is running shell
  commands against a filesystem. `http_fetch` and `web_search` ignore the
  runtime: they always connect from this host, localhost services and
  cloud metadata endpoints included. Leave them out of `--tools` when
  network access must be contained too.

Why not llamafile's pledge sandbox now:
- On macOS it is a no-op, and the GPU gate skips it anyway.
- A policy derived from the enabled toolset would make "is the sandbox
  on?" depend on `--tools`, `--session`, `--trace`, GPU, OS and
  `--unsecure` at once.

Follow-up, in its own PR: one fixed rule modelled on the server table in
`docs/built-in-tools.md`, a status line at every start, `--unsecure` to opt
out, and a startup check naming the enabled tools that can't work under it.

## Packaging

`cosmo_args("/zip/.args", &argv)` plus zipalign embed the agent in the
binary:

```
cp o//agentfile/agentfile my-agent
o//third_party/zipalign/zipalign -j0 my-agent model.gguf system.md .args
# .args, one argument per line:
#   -m /zip/model.gguf
#   --system-file /zip/system.md
#   --tools read_file,grep_search,web_search
#   --max-iterations 25
#   --searxng-url http://localhost:8888
```

`--session` and `--trace` stay off in `.args`: recording is the user's
opt-in. **Never** put `--yes` in `.args`: a packaged agent must not
silently pre-authorize destructive tools.

## Roadmap

- **Packaging**:
  - smoke-test the recipe end to end on a second machine;
  - `--name`/`--description`, so an agent describes itself in `-h`;
  - a prewarmed KV cache: at pack time, run `load_or_create_cache` over
    the system prompt and tool definitions, zip the state
    (`/zip/prompt.cache`), and on load validate it against model and
    context parameters, falling back to live warming on a mismatch.
    Flags `--warm-cache PATH` / `--load-cache PATH`;
  - later, an `agentfile pack` subcommand wrapping the recipe.
- **Full llama.cpp parameter surface**, mirroring llamafile
  (`llamafile/args.cpp`):
  - parse agentfile's own flags first and hand the rest to
    `common_params_parse(..., LLAMA_EXAMPLE_CLI)`;
  - two-tier help: short agentfile help, then `--help` delegates to
    llama.cpp's full catalogue. Delegating, not copying, keeps it from
    drifting;
  - `-hf REPO[:TAG]` and URLs through `common_download_model` (HF cache,
    `--offline`), with download progress on stderr. A packaged agent can
    then carry `-hf repo:tag` instead of an embedded model;
  - open: agent.cpp's `ModelConfig` covers a subset of `common_params`.
    Either extend agent.cpp's Model to take `common_params`, or document
    the flags it ignores (`-ngl`, flash-attn, …).
- **Tests**: `tests/integration/tests/test_agentfile.py` covers the tools,
  sessions, confirmations and flag validation. Still missing: a
  `--max-iterations` run, a trace schema check, a packaged `.args` run,
  and a compile-only `o//agentfile` check on llama.cpp bumps.
- `--otlp-endpoint URL`: POST the same OTLP/JSON to a collector
  (`/v1/traces`) via cpp-httplib.

## Open items (defaults chosen, veto anytime)

- `http_fetch`: an optional `max_bytes` argument (clamped server-side)?
  `--insecure` for self-signed local instances? Is 64 KB the right cap?
  Server tools keep 16 KB for their own outputs.
- `web_search` snippet length.
- Context management: `-c` only sizes the window, so a long tool-heavy
  session still ends with "context size exceeded". Real fix: a trimming
  callback in `before_llm_call` that drops or summarizes old tool
  results (agent.cpp's ContextTrimmerCallback pattern).
- No per-call token cap (deferred 2026-10-06). With greedy decoding, a
  model stuck repeating itself fills the whole context before the call
  fails. `--max-iterations` caps calls, not tokens.
- Token usage in session files is zero until agent.cpp exposes per-call
  counts (upstream or local patch).
- Trace field names: freeze them after the first real consumer.
