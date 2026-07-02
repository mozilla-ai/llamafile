# agentfile — design & implementation plan

Status: draft, 2026-07-01. Decisions recorded from design discussion.

agentfile is a self-contained, multi-platform APE binary that runs a focused
agent loop from a single CLI invocation. The product promise: preconfigure an
agent (model + system prompt + tool allowlist) and share it as **one file**
that runs anywhere, llamafile-style.

## Decisions

1. **agent.cpp**: keep the submodule + patch pipeline; maintain the patches
   locally (no upstream PRs for now). No vendored fork, no ad-hoc rewrite.
   The build already compiles `agent.cpp/src` against **llamafile's**
   llama.cpp (never agent.cpp's own `deps/llama.cpp`, which `make setup`
   does not initialize).
2. **Interaction model**: unix-tool one-shot by default; `-i/--interactive`
   opt-in continuation. No TUI.
3. **Network**: keep `http_fetch`; wire HTTPS via mbedtls into cpp-httplib.
   Do not rely on host `curl` (breaks the self-contained promise).
4. **Packaging**: in scope now — `.args`-driven, llamafile-style zipalign.
5. **Tracing**: two standard artifacts, no invented formats —
   (a) **session records** in pi's session-format v3 JSONL (interop with
   https://pi.dev: `/import`, HTML export, branching UI);
   (b) **spans** in the OTLP/JSON encoding with GenAI semconv attributes
   (the standard OTLP wire format, JSON flavor — collector-ingestible),
   emitted without the OTel SDK (protobuf/curl violate the single-binary
   constraint; OTLP's JSON encoding needs only nlohmann/json).

---

## 1. agent.cpp strategy

- Keep the submodule + `agent.cpp.patches/` pipeline; maintain the two
  patches locally. Upstreaming is deliberately skipped for now (both
  patches are generic fixes and remain PR-able later if that changes).
- The pin (`7b75852`) is already upstream HEAD; sync opportunistically
  when upstream moves, re-applying patches.
- llama.cpp is **not** pulled twice: `make setup` initializes agent.cpp
  non-recursively and `agent.cpp.patches/llamafile-files/BUILD.mk`
  compiles `src/model.cpp` + `src/agent.cpp` against llamafile's
  `llama.cpp/` includes. (A locally-initialized `agent.cpp/deps/llama.cpp`
  checkout from earlier experiments can be dropped with
  `git -C agent.cpp submodule deinit -f deps/llama.cpp`.)
- Version-skew policy: when llamafile bumps its llama.cpp pin and
  agent.cpp's two TUs break, fix via a local patch; the drift is small and
  surfaces at compile time.

## 2. CLI UX

- Default: one-shot. Final answer on **stdout** only; all progress,
  confirmations, and errors on **stderr**. Pipe-friendly.
- If `-p` is absent and stdin is not a tty, read the prompt from stdin
  (`echo "..." | agentfile -m m.gguf`).
- `-i / --interactive`: after each answer, prompt on stderr and read the
  next user message from `/dev/tty` (stdin may be a pipe). Empty line or
  EOF exits. Reuses the same `messages` vector — KV-prefix diffing makes
  the continuation nearly free.
- Flag parsing must be **last-wins**, and every mode flag needs an inverse
  (`--confirm` to undo `--yes`, `--no-interactive`, …) so packaged `.args`
  defaults can always be overridden on the command line (cosmo_args
  prepends `.args` before user args).
- `--system-file PATH`: read system prompt from a file (works with
  `/zip/...` paths). Long prompts don't belong in `.args` one-liners.
- Exit codes: `0` success, `2` agent_cpp error, `3` other error,
  `4` max-iterations exceeded, `5` user declined a tool confirmation and
  the loop could not continue.

## 3. Session records & tracing

Two artifacts, both existing standards — nothing invented:

### 3a. Session record — pi session-format v3 (JSONL)

- `--session FILE`: a `SessionRecorderCallback` writes the conversation in
  [pi](https://pi.dev)'s documented session format
  (`pi/packages/coding-agent/docs/session-format.md`, `version: 3`):
  header line `{"type":"session","version":3,"id":…,"timestamp":…,"cwd":…}`
  followed by tree-linked entries (`id`/`parentId`) — `user`, `assistant`
  (content blocks, tool calls, `model`, `usage` tokens, `stopReason`),
  `toolResult` (tool call id, name, content, status).
- Interop payoff: agentfile sessions can be opened in pi (`/import`),
  exported to HTML, branched, and shared — and pi sessions of the same
  shape can be compared against agentfile runs.
- Mapping notes: `provider` = `"agentfile"`, `model` = GGUF basename,
  cost fields zero/omitted, agentfile's linear history is a degenerate
  tree (each entry's `parentId` = previous entry).
- Risk to track: it's an application format owned by Earendil with version
  migrations; we pin v3 and revisit when pi bumps.

### 3b. Spans — OTLP/JSON with GenAI semconv

- `--trace FILE`: emit spans in the **OTLP JSON encoding**
  (`ExportTraceServiceRequest`, one JSON object per line) — the standard
  OTel wire format in its JSON flavor. Any OTel Collector ingests it via
  the `otlpjsonfile` receiver and forwards to Jaeger/Tempo/etc.
- Span structure and attributes copy agent.cpp's own
  `examples/tracing` callback exactly: nested `invoke_agent` → `chat` →
  `execute_tool` spans per the GenAI semantic conventions — so agentfile
  traces are semantically identical to agent.cpp's, minus the SDK.
- No OTel SDK/protobuf/curl in the binary: OTLP/JSON needs only
  nlohmann/json (already linked). Span/trace ids are random hex; timestamps
  from `chrono`.
- **Later, after the mbedtls work (§4)**: optional `--otlp-endpoint URL`
  that POSTs the same OTLP/JSON to `http://…:4318/v1/traces` via
  cpp-httplib — fully standard live OTLP/HTTP export, still no protobuf
  (JSON encoding is part of the OTLP spec).

### 3c. stderr verbosity

- Three levels: `--quiet` (nothing), default (current one-line tool
  progress), `-v` (args + truncated results). `ProgressCallback` stays;
  it just gains the `-v` branch.

## 4. Network: HTTPS + http_fetch

- ~~Build task: compile the HTTPLIB objects with
  `CPPHTTPLIB_MBEDTLS_SUPPORT`~~ **Done on main** (PR #1011, merged
  2026-07-03): cpp-httplib is built with its Mbed TLS backend against
  `third_party/mbedtls`; `common/http.h`'s https guard is now
  `CPPHTTPLIB_SSL_ENABLED`. agentfile's BUILD.mk defines the same macro
  (ABI: it changes httplib class layouts) and links `mbedtls.a`.
- Certificate verification: **confirmed working** (2026-07-02 smoke test:
  `https://example.com` fetches; `https://self-signed.badssl.com` fails
  with "SSL server verification failed"). Roots come from llamafile's
  vendored mbedtls. Open: whether to add `--insecure` as an escape hatch
  for self-signed local instances (e.g. a LAN SearXNG behind a
  self-signed cert).
- `http_fetch` keeps GET-only, 16 KB default cap; add an optional
  `max_bytes` argument (server-side clamped). HTML→text extraction is
  explicitly out of scope for v0 — the model gets raw bytes.
- Stays in the confirmation set (network = exfiltration surface).

## 5. web_search tool (searxng)

- Tool name `web_search`. Arguments (all model-visible):
  - `query` (string, required)
  - `page` (int, default 1) → `pageno`
  - `time_range` (enum `day|month|year`, optional)
  - `categories` (string, optional, comma-separated)
- The **instance URL is not a tool argument** — the model must not choose
  where queries go. Configure via `--searxng-url URL` or `SEARXNG_URL`
  env; if neither is set, the tool is simply **not registered** (no dead
  tool in the schema).
- Request: `GET {base}/search?q=…&format=json&…`. Response: top ~8 results
  as compact JSON (`title`, `url`, `content`, `engine`), size-capped.
- Error mapping: HTTP 403 → clear message that the instance must enable
  `json` in `settings.yml` `search: formats:` (public instances usually
  don't — self-hosted is the expected deployment).
- Shares the HTTP client helper with `http_fetch` (factor into
  `tools_common.h` or a small `http_client.h`). Works against
  `http://localhost:8888` before the HTTPS work lands, so it is **not**
  blocked on item 4.
- In the confirmation set by default, same rationale as `http_fetch`.

## 6. Packaging: the shareable single-file agent

Mechanism already exists: `cosmo_args("/zip/.args", &argv)` + llamafile's
`zipalign` to embed files in the APE. A packaged agent is:

```
cp o//agentfile/agentfile my-agent
o//llamafile/zipalign -j0 my-agent model.gguf system.md
# add .args:
#   -m /zip/model.gguf
#   --system-file /zip/system.md
#   --tools read_file,grep_search,web_search
#   --max-iterations 25
#   --searxng-url http://localhost:8888
```

`.args`-relevant parameters (existing + new):

| flag | status | packaging role |
|---|---|---|
| `-m /zip/model.gguf` | exists | embedded model |
| `--system-file /zip/system.md` | **new** | embedded persona/instructions |
| `--tools LIST` | exists | tool allowlist baked into the agent |
| `--max-iterations N` | exists | runaway cap for shared agents |
| `--searxng-url URL` | **new** | pre-pointed search instance |
| `--trace FILE` | **new** | off by default; user-side opt-in |
| `--session FILE` | **new** | off by default; user-side opt-in |
| `--name` / `--description TEXT` | **new** | agent self-describes in `-h` output |
| `--yes` | exists | **never** put in `.args` — packaged agents must not silently pre-authorize destructive tools |

- **Prewarmed KV cache** (committed feature, not stretch): at pack time,
  run `agent.load_or_create_cache` over system prompt + tool definitions
  and zip the resulting state file (e.g. `/zip/prompt.cache`) into the
  agent — instant first token on the recipient's machine. On load,
  validate against model + ctx params and fall back to live warming on
  mismatch. Needs a small `--warm-cache PATH` (generate) /
  `--load-cache PATH` (use, `/zip/`-aware) flag pair.
- Later ergonomics: an `agentfile pack` subcommand wrapping the zipalign
  recipe. v0: document the recipe.

---

## Implementation plan (ordered)

1. **CLI polish**: stdin prompt, last-wins + inverse flags, exit codes,
   `--system-file`. Small, unblocks packaging semantics.
2. **Session records + spans**: `SessionRecorderCallback` (pi v3 JSONL)
   and `TraceCallback` (OTLP/JSON); verbosity levels. Validate a recorded
   session by importing it into pi.
3. **HTTPS**: done — #1011 on main + agentfile BUILD.mk flags/mbedtls.a;
   smoke-tested (real fetch works, bad certs rejected).
4. **web_search**: done — validated against a mock and against a real
   SearXNG instance (`http://raspi:8888`, 2026-07-02: query → 8 capped
   results from 10, model cited them correctly).
5. **`--interactive`** continuation mode.
5b. **Full llama.cpp parameter surface** + two-tier help + `-hf`/HF cache
   (see follow-up section below).
6. **Packaging**: document + smoke-test the zipalign recipe end-to-end on
   a second machine; `--name/--description`; pack-time prewarmed KV cache
   (`--warm-cache`/`--load-cache`).
7. **Integration tests** in `tests/integration/` (absolute paths, models
   from `~/zipaligner_files`): one-shot run with `--tools read_only`,
   tool-call round-trip, `--max-iterations`, session/trace schema checks,
   web_search against a mocked searxng (fixed JSON on localhost).
8. *(post-HTTPS, optional)* `--otlp-endpoint` live OTLP/HTTP-JSON export.

Dropped for now: upstream PRs to agent.cpp (patches maintained locally;
pin already at upstream HEAD).

## Follow-up: full llama.cpp parameter surface + two-tier help

Decision (2026-07-02): don't cherry-pick `-hf` — adopt **all** of
llama.cpp's parameters, mirroring llamafile's design (see
`llamafile/main.cpp:106-155`):

- **Arg split**: parse agentfile-specific flags first (`-p`, `-s`,
  `--system-file`, `--tools`, `--yes/--confirm`, `--session`, `--trace`,
  `--searxng-url`, `--max-iterations`, `-i`, verbosity) and strip them;
  feed the remainder to `common_params_parse(..., LLAMA_EXAMPLE_CLI)`
  into a `common_params`. (llamafile does this split in
  `llamafile/args.cpp` — reuse the approach.)
- **Two-tier help**: run with no/insufficient args → short hand-written
  agentfile help only. Explicit `--help`/`-h` → short help, then delegate
  to `common_params_parse()` which prints the full categorized llama.cpp
  catalogue and exits. Delegation (not a copy) is what prevents drift.
- **Model resolution**: `-hf REPO[:TAG]` / `-m` / URLs come free via
  `params.model`; call `common_download_model(params.model, opts)`
  (common/download.h — HF cache for repos, ETag cache for URLs, split
  GGUFs, `--offline` support; `common_list_cached_models()` for a
  future `--list-models`). Feed `result.model_path` to
  `ModelWeights::create`. HTTPS (#1011) is what makes downloads work.
- **Main integration question**: agent.cpp's `ModelConfig` is a narrow
  subset of `common_params` (temp/top_k/top_p/min_p/seed/n_ctx/n_batch/
  cache types). Map what exists; for the rest (`-ngl`, flash-attn, etc.)
  either extend agent.cpp's Model to accept `common_params`
  (upstream-worthy) or document unsupported flags. Decide at
  implementation time.
- **Caveats**: make sure download progress reaches stderr despite the
  silenced llama log callback; packaging variant — a shared agent's
  `.args` can carry `-hf repo:tag` instead of an embedded model (small
  file, fetched/cached on first run).

## Open items (defaults chosen, veto anytime)

- 16 KB body/output caps — right default? (kept from server-tools)
- `web_search` result count (8) and snippet truncation length.
- Whether `get_datetime` belongs in `read_only` preset (yes for now).
- Trace format field names — freeze only after first real consumer.
- Qwen3.5's empty `<think>\n\n</think>` block leaks into assistant
  content (visible in stdout and session files): the chat parser's
  reasoning syntax isn't configured in agent.cpp's Model. Follow-up:
  wire `reasoning_format`/`enable_thinking` through `ModelConfig`.
- Token usage in session files is recorded as zeros — agent.cpp's Model
  doesn't expose per-call token counts to callbacks. Candidate upstream
  patch (or local patch 0003).

## Status (2026-07-02)

Implemented and verified on this branch: CLI polish (stdin prompt,
`--system-file`, inverse flags, exit codes 0/1/2/3/4), `--session` (pi v3;
validated by parsing with pi's own SessionManager — id/tree/context all
resolve), `--trace` (OTLP/JSON; parent/child spans + aborted-span closing
verified), `web_search` (validated end-to-end against a mock SearXNG over
http; real-instance + https test pending the mbedtls work). Note:
llamafile's `third_party/mbedtls` already ships `ssl_cli.c`, x509, and
`sslroots.c` (CA roots) — the HTTPS task is wiring cpp-httplib to it, not
importing a TLS stack.
