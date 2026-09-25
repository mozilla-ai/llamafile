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
- Exit codes: `0` success, `1` usage error, `2` agent_cpp error, `3` other
  error, `4` max-iterations exceeded. Declining a confirmation is not an
  exit: agent.cpp catches `ToolExecutionSkipped` inside the loop, the model
  receives `{"skipped": "user declined"}` as the tool result and decides
  how to continue. No answer at all (no terminal to ask on, or EOF at the
  prompt) does end the run, with exit 2: every later guarded call would
  go unanswered too, and the model would retry them without end.

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

- Three levels: `--quiet` (nothing, except a one-line `[tool --yes] args`
  audit record for each destructive tool run under `--yes`), default
  (current one-line tool progress), `-v` (args + truncated results).
  `ProgressCallback` stays; it just gains the `-v` branch.

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
- `http_fetch` keeps GET-only with a 64 KB body cap (the download stops
  there; server-tools use 16 KB for their own outputs). Redirects are
  reported as `redirect_to`, not followed, so each fetched URL is one the
  model asked for and the confirmation prompt showed. Still to do: an
  optional `max_bytes` argument (server-side clamped). HTML→text
  extraction is explicitly out of scope for v0 — the model gets raw bytes.
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

## 7. Security model (decided 2026-09-23)

agentfile does **not** sandbox itself: there is no `pledge()`/`unveil()`
call anywhere in it, so every tool runs with the invoking user's full
permissions, on macOS as on Linux. What stands between the model and the
host:

- The confirmation prompt for `permission_write` tools (writes, shell,
  network) is the only gate. `--yes` removes it; under `--yes --quiet`
  each such call is still logged to stderr, one line each.
- Tool inputs are defended at the adapter: model-supplied `runtime`,
  `cwd` and `resp_type` keys are stripped before a server tool runs
  (llama-server does the same in its HTTP handler). `http_fetch` reports
  redirects instead of following them and stops downloading at 64 KB.
- Isolation, when wanted, comes from `--tools-runtime`: tools then run
  inside an already-running container (or over ssh). That is the right
  boundary for an agent whose purpose is running shell commands against
  a filesystem. The exceptions are `http_fetch` and `web_search`, which
  are agentfile-native and ignore the runtime: they always connect from
  this host, localhost services and cloud metadata endpoints included.
  Leave them out of `--tools` when network access must be contained too.

Why not llamafile's pledge sandbox now: on macOS it is a no-op and the
GPU gate skips it anyway, so it would protect nobody on the platform
agentfile is developed on; and a policy derived from the enabled toolset
would make "is the sandbox on?" depend on `--tools`, `--session`,
`--trace`, GPU, OS and `--unsecure` at once. Follow-up, in its own PR:
one fixed rule modelled on the server table in `docs/built-in-tools.md`,
a status line printed at every start, `--unsecure` to opt out, and a
startup check naming the enabled tools that cannot work under it.

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
5. **`--interactive`** continuation mode: done (2026-09-10). `-i` prompts
   on stderr after each answer; follow-ups read from /dev/tty when stdin
   was a pipe; empty line/EOF ends the session; `--no-interactive` is the
   `.args`-friendly inverse. `--max-iterations` is now a per-turn budget
   (reset each run_loop). Verified under a pty: two turns, KV prefix
   reused, both turns in the session file.
6. **agent.cpp sync to v0.4.0**: done (2026-09-10). Pin 7b75852 →
   63d23da; upstream #22's PEG-parser work replaced the old patch-0001
   loading half. Remaining local patches (per-file,
   `src_model.{cpp,h}.patch`, ~500 lines): (a) common_sampler switch +
   chat-template tool-call grammar (user GBNF rides along via
   `COMMON_GRAMMAR_TYPE_USER`; caveat: `ModelConfig::grammar_root` is
   ignored — grammars must use "root"). The new sampler is built before
   the old one is freed; a user grammar is re-initialized on every call
   (`common_sampler_reset` leaves a finished grammar finished); the
   template's thinking tags go to the sampler as reasoning-budget
   start/end, so a lazy tool-call grammar stays dormant inside
   `<think>` (as llama-server does); (b) parse-failure fallback +
   heuristic `<tool_call>` recovery (upstream now throws ModelError,
   which would kill the loop). Recovery scans the content only (the
   reasoning is split off first), and only when the parser failed or
   the template has no tool-call grammar; a `<tool_call>` it cannot
   lift rethrows the parse error (exit 2) instead of becoming the
   answer; recovered arguments keep the model's key order, so the next
   prompt matches the KV cache. A reply that ends inside a reasoning
   block it never closed returns that block as the content; (c)
   hybrid-model rewind fix (`llama_memory_seq_rm` returns false on
   recurrent state → fall back to `llama_memory_clear` + full
   re-decode); (d) `tokenize()` always adds special tokens: keying BOS
   on an empty cache dropped it on every call after the first, which
   broke the KV prefix match for BOS models; (e)
   `ModelConfig::enable_thinking` for `--think`. All are upstream PR
   candidates. Verified: tool-call grammar path, hybrid interactive,
   web_search live, session/trace, patch round-trip.
7. **server-tools adapter**: done (2026-09-10). `server_tools_adapter.h`
   bridges llama.cpp's `server_tool` into `agent_cpp::Tool`
   (`ServerToolAdapter` per tool + owning `ServerToolbox`); all eight
   vendored tool copies deleted — implementations now come from
   `llama.cpp/tools/server/server-tools.cpp` (gaining `get_info`,
   losing `apply_diff`, which upstream doesn't have). http_fetch and
   web_search rewritten as `server_tool` subclasses (upstream-shaped,
   foldable into llama.cpp by file move). Confirmations now driven by
   the tools' own `permission_write` metadata instead of a hardcoded
   list. `--tools-runtime SPEC` plumbs the isolation spec in as
   `params["runtime"]` — the same mechanism llama-server's handler uses
   (flag wired; not yet tested against a live podman/docker isolate).
   json boundary note: server tools speak `nlohmann::ordered_json`,
   agent.cpp speaks `nlohmann::json`; the adapter converts via
   dump/parse. Links five extra server TUs (tools, common, queue, mcp,
   http); an empty `server_mcp` keeps MCP inert.
8. **Ownership exercise** (pre-packaging, gates any push; decided
   2026-09-10). Four phases, each gating the next:
   - *Simplify with approved cuts*: full-tree pass (structure, naming,
     comment calibration); a proposed-cuts list goes to Davide for
     per-item approval before anything is removed.
   - *Live walkthrough*: file-by-file in conversation, dependency order,
     simplifying inline as we read.
   - *Reverse Q&A*: Davide explains the design back; gaps get flagged.
     Exit criterion: every decision defensible without help.
   - *Fresh curated branch*: rebuild from current main as ~8-12 small
     commits telling the story in order, plain-language messages in
     main's conventional style; Davide reviews each commit before push.
     Current branch kept as development reference.
9. **Packaging**: document + smoke-test the zipalign recipe end-to-end on
   a second machine; `--name/--description`; pack-time prewarmed KV cache
   (`--warm-cache`/`--load-cache`).
10. **Full llama.cpp parameter surface** + two-tier help + `-hf`/HF cache
   (see follow-up section below; moved after packaging).
11. **Integration tests** in `tests/integration/` (absolute paths, models
   from `~/.llamafile/models`): one-shot run with `--tools read_only`,
   tool-call round-trip, `--max-iterations`, session/trace schema checks,
   web_search against a mocked searxng (fixed JSON on localhost). Plus a
   compile-only `o//agentfile` check on llama.cpp-bump PRs.
12. *(optional)* `--otlp-endpoint` live OTLP/HTTP-JSON export.

Upstream PRs: agent.cpp patch 0002 is a strong candidate after the v0.4.0
sync (composes with upstream #20); SearXNG web_search to llama.cpp
server-tools deferred until the adapter proves out.

## Strategy: tool layer & positioning (decided 2026-09-10)

Context: upstream llama.cpp now ships built-in server tools
(`tools/server/server-tools.{h,cpp}`, `--tools`, `-ag/--agent`, a `/tools`
HTTP catalogue), including tool **isolation runtimes** (`--tools-runtime`:
podman rootless / ssh remote), MCP-sourced tools, and `permission_write`
metadata. The agent loop upstream is thin and client-side (Web UI;
`agent.py` in this repo is a 300-line replacement). What upstream does NOT
have: serverless one-shot CLI, single-file packaged agents, session/trace
recording, KV-warm-start packaging — agentfile's identity is **the file
format for shipping agents**, not another harness.

Decisions:

1. **Inherit upstream tools via an adapter**: one `server_tool →
   agent_cpp::Tool` bridge (the interfaces are nearly isomorphic:
   `get_definition()/invoke(json)` vs `get_definition()/execute(json)`).
   Upstream maintains tool semantics; we inherit isolation runtimes,
   new tools, and `permission_write` (which replaces the hardcoded
   destructive-tools set in the confirmation callback).
2. **Write agentfile-native tools as `server_tool` subclasses** (not
   `agent_cpp::Tool`), so the adapter is the single bridge and folding a
   tool upstream later is a file move, not a rewrite. web_search is the
   first candidate; offering it upstream: **later**, after the adapter
   proves out.
3. **API-drift posture**: accepted consciously — we can lock the pin or
   start vendoring at any moment if server-tools internals churn too hard.
4. **agent.cpp: sync to v0.4.0 (63d23da)** — the delta beyond llama.cpp
   bumps is two commits: #22 upstreams the PEG-parser-loading half of our
   patch 0001 (drop that half, keep the `<tool_call>` fallback extractor
   as a smaller patch); #20 adds user-supplied GBNF grammar, which does
   NOT overlap our 0002 (template-generated tool grammar via
   common_sampler) — 0002 stays and is now a stronger upstream PR
   candidate since it composes with #20's plumbing.

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

- 64 KB `http_fetch` body cap (server-tools keep 16 KB for read_file and
  shell output) — right default?
- `web_search` result count (8) and snippet truncation length.
- Whether `get_datetime` belongs in `read_only` preset (yes for now).
- Trace format field names — freeze only after first real consumer.
- Context management: `-c/--ctx-size` (added 2026-09-10; default 32768
  via `kDefaultCtx`, overriding agent.cpp's own 10240; `0` = model-native)
  only sizes the window — a long
  tool-heavy session still dies with "context size exceeded" instead of
  degrading. Real fix: a trimming callback in `before_llm_call` (agent.cpp's
  ContextTrimmerCallback pattern) that drops/summarizes old tool results.
- Qwen3.5's `<think>` blocks leak into assistant content (the chat
  parser's reasoning syntax isn't configured in agent.cpp's Model).
  **Upgraded from cosmetic to performance-critical (2026-09-10)**: the
  leak makes every re-render diverge from the decoded stream (the
  template re-strips think blocks each turn), which forces a KV rewind
  per turn — and on recurrent/hybrid models (Qwen3.5) a rewind is a
  full re-prefill. Fixing the reasoning round-trip makes re-renders
  byte-stable, so the warm cache survives interactive turns. Rejected
  alternative, kept in the back pocket: append-only token stream (never
  re-render history) — avoids rewinds entirely but bloats context with
  old reasoning, fights the template's seams, and breaks as soon as
  context trimming edits history.
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
