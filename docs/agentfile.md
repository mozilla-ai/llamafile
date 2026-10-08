# Agentfile

Agentfile runs a small, focused AI agent from a single command: one
executable, one model, a prompt in, an answer out. The model can use tools
— read and write files, run shell commands, fetch pages, search the web —
and every run works the same way on Linux, macOS, Windows and BSD, with no
installation and no server.

```sh
# Plain question, no tools
./agentfile -m Qwen3-4B.gguf -p "Explain git rebase in two sentences."

# Let it use the web (asks before each network call; --yes skips asking)
./agentfile -m Qwen3-4B.gguf --tools all --searxng-url http://localhost:8888 \
    -p "Find the latest llamafile release and summarize what changed."

# Keep the conversation going
./agentfile -m Qwen3-4B.gguf --tools read_only -i -p "What is in this directory?"
```

The answer goes to stdout; everything else (tool activity, prompts,
warnings) goes to stderr. `./agentfile result.txt 2>/dev/null` style
piping works as you would expect.

## Tools

**Tools are off by default.** Tool descriptions take up context the model
could use for your task, and every tool is capability you may not want to
hand out — so you enable exactly what a run needs:

```sh
--tools read_only                      # look, don't touch
--tools all                            # everything available
--tools read_file,exec_shell_command   # a specific set
```

The tools (run `agentfile -h` to see this list live):

| Tool | What it does | Asks first? |
|---|---|---|
| `read_file`, `file_glob_search`, `grep_search` | read and search files | no |
| `get_datetime`, `get_info` | date/time, system info | no |
| `write_file`, `edit_file` | create and change files | yes |
| `exec_shell_command` | run shell commands | yes |
| `http_fetch` | GET a URL (http/https) | yes |
| `web_search` | search via SearXNG | yes |

"Asks first" tools print the exact call and wait for your `y` before
running. `--yes` turns the prompts off for unattended runs (each call is
still logged, one line each); `--confirm` turns them back on, which
matters for packaged agents that bake in `--yes`.

Most tools are llama.cpp's own [built-in tools](built-in-tools.md), so
they behave exactly as they do in llama-server. `get_datetime`,
`http_fetch` and `web_search` are agentfile additions.

### Web search

`web_search` talks to a [SearXNG](https://docs.searxng.org) instance you
point it at with `--searxng-url URL` (or the `SEARXNG_URL` environment
variable). Without an instance, the tool is simply not available. The
instance must allow the JSON API: add `json` under `search: formats:` in
its `settings.yml` — most public instances don't, so self-hosting is the
expected setup. The model picks the query; it can never pick the instance.

### Sandboxing

By default tools run with your permissions on your machine, and the
confirmation prompt is the only gate. For real isolation, run the tools
somewhere expendable:

```sh
--tools-runtime docker-container:NAME   # a container you already started
--tools-runtime docker:IMAGE            # started for this run, stopped after
--tools-runtime ssh:TARGET              # a remote machine
```

Shell and file tools then operate inside the sandbox (podman works like
docker). `http_fetch` and `web_search` still connect from this host.

## Recording runs

- `--session FILE` writes the whole conversation as a
  [pi](https://pi.dev) session (JSONL). You can open it with pi, branch
  it, export it to HTML, or parse it with a few lines of Python.
- `--trace FILE` writes timing spans (model calls, tool calls) as
  OTLP/JSON lines, which an OpenTelemetry collector ingests directly —
  useful to see where a slow run spent its time.

Both are off unless asked for, and both overwrite their file.

## Useful flags

- `-i` / `--interactive`: after the answer, agentfile asks for a
  follow-up on the terminal; an empty line ends the session. Follow-ups
  reuse the work already done, so they start fast. Tip: models with a
  hybrid/recurrent design (Mamba-style, e.g. Qwen3.5) re-process the
  whole conversation on each follow-up — plain transformers feel snappier
  here.
- `--think`: lets reasoning models think; the reasoning shows dimmed on
  stderr and never mixes into the answer on stdout.
- `-c N`: context window (default 65536, never more than the model's
  maximum; `0` means the model's maximum). Web-heavy tasks want room — a
  single fetched page can be ~16k tokens.
- `--max-iterations N`: a per-reply budget of model calls; exceeding it
  exits with code 4. Good seatbelt for unattended runs.
- `-s TEXT` / `--system-file PATH`: the system prompt, inline or from a
  file.
- `--quiet` / `-v` / `-vv`: from silent to llama.cpp's full logs. `-v`
  also prints the enabled-tools roster and truncated tool results.

Exit codes: `0` ok, `1` usage error, `2` agent error, `3` other error,
`4` iteration cap reached — so scripts can tell "the model failed" from
"the run was cut short".
