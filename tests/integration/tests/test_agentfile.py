"""Ad hoc integration tests for agentfile.

Run from tests/integration, unsandboxed (they bind a loopback HTTP server):

  AGENTFILE_EXECUTABLE=$PWD/../../o/agentfile/agentfile \
  AGENTFILE_MODEL=~/ggufs/Qwen3.5-9B-Q5_K_S.gguf \
  uv run pytest tests/test_agentfile.py -v

The model drives the tool calls at temperature 0, so assertions target the
recorded tool results (--session JSONL) and stderr, not the model's prose.
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

EXE = os.environ.get("AGENTFILE_EXECUTABLE")
MODEL = os.path.expanduser(os.environ.get("AGENTFILE_MODEL", ""))

pytestmark = pytest.mark.skipif(
    not EXE or not MODEL,
    reason="set AGENTFILE_EXECUTABLE and AGENTFILE_MODEL to run agentfile tests",
)

CAP = 64 * 1024              # http_fetch body cap (kMaxBody)
BIG_TOTAL = 256 * 1024       # /big body size: > CAP, small enough to be quick
BIG_CHUNK = 16 * 1024
BIG_DELAY = 0.1              # per chunk, so an early abort is measurable
SECRET = "secret-token-7f3a9"
RUN_TIMEOUT = 300
LATIN1_TEXT = "Café crème brûlée costs 4 euros"
SNIPPET_CAP = 512            # web_search snippet clip (kMaxSnippet)


@dataclass
class ServerState:
    requests: list = field(default_factory=list)
    big_bytes_sent: int = 0
    big_disconnected: bool = False


def _make_handler(state: ServerState, base_url_box: list):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def _bytes(self, code: int, data: bytes, ctype: str = "text/plain",
                   extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _text(self, code: int, body: str, extra: dict | None = None):
            self._bytes(code, body.encode(), extra=extra)

        def do_GET(self):
            state.requests.append(self.path)
            if self.path.startswith("/search"):
                # SearXNG JSON API. The snippet is cut at SNIPPET_CAP bytes,
                # which lands inside a two-byte "é" (odd offset).
                body = {"results": [{"title": "Crème brûlée", "url": "http://example.com/",
                                     "content": "x" + "é" * SNIPPET_CAP, "engine": "mock"}]}
                self._bytes(200, json.dumps(body).encode(), "application/json")
            elif self.path.startswith("/?ref="):
                self._text(200, "at-ok")
            elif self.path == "/latin1":
                self._bytes(200, LATIN1_TEXT.encode("latin-1"),
                            "text/plain; charset=iso-8859-1")
            elif self.path == "/cut":
                # UTF-8, with a two-byte "é" straddling the CAP boundary.
                self._bytes(200, b"x" * (CAP - 1) + "é".encode() * 8,
                            "text/plain; charset=utf-8")
            elif self.path == "/redir":
                self._text(302, "", {"Location": base_url_box[0] + "/target"})
            elif self.path == "/target":
                self._text(200, SECRET)
            elif self.path == "/small":
                self._text(200, "hello from small")
            elif self.path == "/big":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(BIG_TOTAL))
                self.end_headers()
                sent = 0
                try:
                    while sent < BIG_TOTAL:
                        chunk = b"x" * min(BIG_CHUNK, BIG_TOTAL - sent)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        sent += len(chunk)
                        state.big_bytes_sent = sent
                        time.sleep(BIG_DELAY)
                except (BrokenPipeError, ConnectionResetError):
                    state.big_disconnected = True
            else:
                self._text(404, "not found")

    return Handler


@pytest.fixture
def http_server():
    state = ServerState()
    base_url_box = [""]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state, base_url_box))
    base_url_box[0] = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield base_url_box[0], state
    finally:
        srv.shutdown()
        srv.server_close()


@dataclass
class AgentRun:
    proc: subprocess.CompletedProcess
    tool_calls: list
    tool_results: list

    def results_for(self, tool: str) -> list:
        return [r for r in self.tool_results if r.get("toolName") == tool]

    @staticmethod
    def result_text(r: dict) -> str:
        return "".join(c.get("text", "") for c in r.get("content", []))


def run_agentfile(prompt: str, tools: str, tmp_path: Path, extra=(), cwd=None) -> AgentRun:
    session = tmp_path / "session.jsonl"
    # APE binaries need a shell launcher on macOS (kernel rejects the format).
    cmd = [*(["sh"] if os.name != "nt" else []), EXE, "-m", MODEL, "-p", prompt, "--tools", tools,
           "--session", str(session), "--no-think", *extra]
    # No stdin and a new session (no controlling terminal): a confirmation
    # prompt gets no answer instead of waiting on the keyboard.
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          start_new_session=True, cwd=cwd or tmp_path, timeout=RUN_TIMEOUT)
    tool_calls, tool_results = [], []
    if session.exists():
        for line in session.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", entry)
            if msg.get("role") == "assistant":
                tool_calls += [c for c in msg.get("content", []) if c.get("type") == "toolCall"]
            elif msg.get("role") == "toolResult":
                tool_results.append(msg)
    return AgentRun(proc, tool_calls, tool_results)


def _dump(run: AgentRun) -> str:
    return (f"\nexit={run.proc.returncode}\n--- stdout\n{run.proc.stdout}"
            f"\n--- stderr\n{run.proc.stderr}\n--- tool results\n"
            + "\n".join(AgentRun.result_text(r)[:500] for r in run.tool_results))


# --- http_fetch -------------------------------------------------------------

def test_http_fetch_reports_redirect_without_following(http_server, tmp_path):
    base, state = http_server
    run = run_agentfile(
        f"Fetch {base}/redir with http_fetch and reply with the text content of the page.",
        "http_fetch", tmp_path, extra=["--yes"])
    fetches = run.results_for("http_fetch")
    assert fetches, "model did not call http_fetch" + _dump(run)
    first = json.loads(AgentRun.result_text(fetches[0]))
    assert first.get("status") == 302, _dump(run)
    assert first.get("redirect_to", "").endswith("/target"), _dump(run)
    assert "/redir" in state.requests
    # The redirect is reported, not followed: the model has to fetch the
    # target itself in a second call (which is what the confirmation prompt
    # would gate in an interactive run).
    assert "/target" in state.requests, "model did not follow up on the reported redirect" + _dump(run)
    if SECRET not in run.proc.stdout:
        pytest.xfail("model fetched the target but did not quote it (prose check, non-fatal)")


def test_http_fetch_stops_reading_at_cap(http_server, tmp_path):
    base, state = http_server
    run = run_agentfile(
        f"Use the http_fetch tool to fetch {base}/big and tell me whether the body was truncated.",
        "http_fetch", tmp_path, extra=["--yes"])
    fetches = run.results_for("http_fetch")
    assert fetches, "model did not call http_fetch" + _dump(run)
    first = json.loads(AgentRun.result_text(fetches[0]))
    assert first.get("status") == 200, _dump(run)
    assert first.get("truncated") is True, _dump(run)
    assert len(first.get("body", "")) == CAP, _dump(run)
    # The download must stop at the cap instead of buffering the whole body.
    assert state.big_bytes_sent <= BIG_TOTAL // 2, (
        f"server sent {state.big_bytes_sent} of {BIG_TOTAL} bytes; client did not abort at the cap")


# Tool results used to go through a strict json dump(), so a body with
# invalid UTF-8 reached the model as a type_error.316 message instead.
def test_http_fetch_latin1_page_is_returned(http_server, tmp_path):
    base, _ = http_server
    run = run_agentfile(
        f"Fetch {base}/latin1 with http_fetch and tell me what the page says.",
        "http_fetch", tmp_path, extra=["--yes"])
    fetches = run.results_for("http_fetch")
    assert fetches, "model did not call http_fetch" + _dump(run)
    first = json.loads(AgentRun.result_text(fetches[0]))
    assert "error" not in first, _dump(run)
    assert first.get("status") == 200, _dump(run)
    assert "Caf" in first.get("body", ""), _dump(run)


def test_http_fetch_cap_inside_utf8_char(http_server, tmp_path):
    base, _ = http_server
    run = run_agentfile(
        f"Use the http_fetch tool to fetch {base}/cut and tell me whether the body was truncated.",
        "http_fetch", tmp_path, extra=["--yes"])
    fetches = run.results_for("http_fetch")
    assert fetches, "model did not call http_fetch" + _dump(run)
    first = json.loads(AgentRun.result_text(fetches[0]))
    assert "error" not in first, _dump(run)
    assert first.get("status") == 200, _dump(run)
    assert first.get("truncated") is True, _dump(run)


# common_http_parse_url took the first '@' anywhere after the scheme as the
# end of the userinfo, so this URL connected to 127.0.0.2 while the
# confirmation prompt showed 127.0.0.1.
def test_http_fetch_at_sign_stays_on_displayed_host(http_server, tmp_path):
    base, state = http_server
    port = base.rsplit(":", 1)[1]
    url = f"{base}/?ref=a@127.0.0.2:{port}/"
    run = run_agentfile(
        f"Call http_fetch once with exactly this url, unchanged: {url} and report the status code.",
        "http_fetch", tmp_path, extra=["--yes"])
    fetches = run.results_for("http_fetch")
    assert fetches, "model did not call http_fetch" + _dump(run)
    assert any(p.startswith("/?ref=a") for p in state.requests), (
        f"request did not reach the displayed host: {state.requests}" + _dump(run))
    first = json.loads(AgentRun.result_text(fetches[0]))
    assert first.get("status") == 200, _dump(run)


# --- web_search -------------------------------------------------------------

def test_web_search_snippet_cut_inside_utf8_char(http_server, tmp_path):
    base, state = http_server
    run = run_agentfile(
        "Use web_search to search for creme brulee and tell me the title of the first result.",
        "web_search", tmp_path, extra=["--yes", "--searxng-url", base])
    searches = run.results_for("web_search")
    assert searches, "model did not call web_search" + _dump(run)
    assert any(p.startswith("/search") for p in state.requests)
    first = json.loads(AgentRun.result_text(searches[0]))
    assert "error" not in first, _dump(run)
    assert first["results"][0]["snippet"].endswith("…"), _dump(run)


# --- --tools ----------------------------------------------------------------

# An empty keep-set means "all tools" to make_adapters, so a list naming
# nothing used to enable every tool, exec_shell_command included.
@pytest.mark.parametrize("spec", [",", " ", ""])
def test_tools_list_naming_nothing_is_rejected(tmp_path, spec):
    run = run_agentfile("hi", spec, tmp_path, extra=["--yes"])
    assert run.proc.returncode == 1, _dump(run)
    assert "--tools: no tool names" in run.proc.stderr, _dump(run)
    assert not run.tool_calls, _dump(run)


# --- --session --------------------------------------------------------------

# The session recorder used a strict json dump(): a tool result with invalid
# UTF-8 (here a Latin-1 file) killed the run with exit 3.
def test_session_records_latin1_file(tmp_path):
    (tmp_path / "menu.txt").write_bytes(LATIN1_TEXT.encode("latin-1"))
    run = run_agentfile("Use read_file to read menu.txt and tell me what it says.",
                        "read_file", tmp_path)
    assert run.proc.returncode == 0, _dump(run)
    reads = run.results_for("read_file")
    assert reads, "model did not call read_file (or the result was not recorded)" + _dump(run)
    text = AgentRun.result_text(reads[0])
    assert "Caf" in text and "�" in text, _dump(run)


# --- --quiet --yes audit ----------------------------------------------------

WRITE_PROMPT = ("Use the write_file tool to create a file named note.txt in the "
                "current directory containing exactly the text hello. Then reply done.")


def test_quiet_yes_prints_audit_line(tmp_path):
    run = run_agentfile(WRITE_PROMPT, "write_file", tmp_path, extra=["--yes", "--quiet"])
    assert (tmp_path / "note.txt").exists(), _dump(run)
    assert "[write_file --yes]" in run.proc.stderr, _dump(run)
    assert "[tool: write_file" not in run.proc.stderr, "progress output leaked under --quiet" + _dump(run)


def test_default_verbosity_has_progress_not_audit(tmp_path):
    run = run_agentfile(WRITE_PROMPT, "write_file", tmp_path, extra=["--yes"])
    assert (tmp_path / "note.txt").exists(), _dump(run)
    assert "[tool: write_file" in run.proc.stderr, _dump(run)
    assert "--yes]" not in run.proc.stderr, "duplicate audit line at default verbosity" + _dump(run)


# --- confirmation without a terminal ----------------------------------------

# With no terminal to confirm on, a guarded call used to come back as
# "user cancelled" and the model retried it without end.
def test_guarded_call_without_terminal_ends_run(tmp_path):
    run = run_agentfile(WRITE_PROMPT, "write_file", tmp_path)
    assert run.proc.returncode == 2, _dump(run)
    assert "cannot confirm write_file" in run.proc.stderr, _dump(run)
    assert run.proc.stderr.count("Allow? [y/N]") == 1, _dump(run)
    assert not (tmp_path / "note.txt").exists(), _dump(run)
