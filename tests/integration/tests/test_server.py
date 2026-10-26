"""Server mode integration tests."""

import os
import socket
import sys
import time

import pytest

from utils.llamafile import LlamafileRunner
from utils.prompts import ADD_2_2, GREETING_PROMPT


@pytest.mark.server
class TestServerBasic:
    """Basic server mode tests."""

    def test_server_starts_and_responds(self, llamafile, server_port, timeouts):
        """Test that server starts and responds to health check."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready in time"
        finally:
            proc.terminate()
            proc.wait()

    def test_server_chat_completion(self, llamafile, server_port, timeouts):
        """Test basic chat completion endpoint."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": GREETING_PROMPT}],
                timeout=timeouts.http_request,
            )

            assert "choices" in response
            assert len(response["choices"]) > 0
            content = response["choices"][0]["message"]["content"]
            assert len(content.strip()) > 0

        finally:
            proc.terminate()
            proc.wait()

    def test_server_chat_completion_math(self, llamafile, server_port, timeouts):
        """Test chat completion with a math question."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": ADD_2_2.prompt}],
                timeout=timeouts.http_request,
            )

            content = response["choices"][0]["message"]["content"]
            assert ADD_2_2.check(content), f"Expected {ADD_2_2.describe()} in content: {content}"

        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.server
@pytest.mark.cli
class TestServerParameters:
    """Test server with various parameters."""

    @pytest.mark.determinism
    def test_server_with_temperature_zero(self, llamafile, server_port, timeouts):
        """Test that temperature=0 produces consistent output.

        Determinism is checked cold-vs-cold: each completion runs against its
        own freshly-started server, so no state carries between them. This is
        deliberate. Two requests to the *same* running server are not a fair
        determinism check: the first does a full prompt prefill while the
        second reuses the cached prefix KV (cache_prompt defaults to true), and
        the different batch composition perturbs the logits by ~1 ULP -- enough
        to flip a knife's-edge argmax at temperature 0. That is inherent
        floating-point non-associativity in llama.cpp's cached-prefix path, not
        a llamafile bug, so we isolate the kernel's own determinism by giving
        each request an identical cold start.
        """
        messages = [
            {
                "role": "user",
                "content": "Hello",
            }
        ]

        def cold_completion() -> str:
            """Run one completion against a fresh server, then tear it down."""
            proc = llamafile.start_server(port=server_port)
            try:
                assert LlamafileRunner.wait_for_server(
                    server_port, timeout=timeouts.server_ready, proc=proc
                ), "Server did not become ready"
                # Use streaming with time limit to handle slow/thinking models
                return LlamafileRunner.chat_completion_streaming(
                    port=server_port,
                    messages=messages,
                    temperature=0.0,
                    collect_timeout=20.0,
                )
            finally:
                proc.terminate()
                proc.wait()

        content1 = cold_completion()
        content2 = cold_completion()

        # Compare the shorter response - it should match the prefix of the longer
        # (they may differ in length if one timed out earlier)
        min_len = min(len(content1), len(content2))
        assert min_len > 0, "No content received from either response"

        assert content1[:min_len] == content2[:min_len], (
            f"Expected consistent output with temperature=0.\n"
            f"Response 1: {content1[:200]!r}...\n"
            f"Response 2: {content2[:200]!r}..."
        )

    def test_server_with_max_tokens(self, llamafile, server_port, timeouts):
        """Test that max_tokens parameter limits output."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Count from 1 to 100"}],
                max_tokens=10,
                timeout=timeouts.http_request,
            )

            # Output should be limited
            content = response["choices"][0]["message"]["content"]
            # With max_tokens=10, we shouldn't get to 100
            assert "100" not in content or len(content) < 50

        finally:
            proc.terminate()
            proc.wait()


def _unix_http_get(sock_path: str, path: str, timeout: float) -> tuple[int, bytes]:
    """Minimal HTTP/1.1 GET over a Unix domain socket: (status, body)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(sock_path)
        s.sendall(
            f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode()
        )
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    return int(head.split(b" ", 2)[1]), body


@pytest.mark.server
@pytest.mark.skipif(sys.platform == "win32", reason="the test client needs AF_UNIX on a POSIX host")
class TestServerUnixSocket:
    """--host <path>.sock serves HTTP over a Unix domain socket.

    --unsecure keeps these independent of the pledge() sandbox, which
    test_sandbox.py covers.
    """

    # "server.sock" relative to the server's cwd is 11 bytes: shorter than
    # the 14 a path needs for 2 + strlen(path) to reach sizeof(struct
    # sockaddr), below which Cosmopolitan's bind() returns EINVAL.
    @pytest.mark.parametrize(
        "relative", [False, True], ids=["absolute-path", "short-relative-path"]
    )
    def test_server_on_unix_socket(
        self, executable, model, gpu_mode, server_port, timeouts, tmp_path, monkeypatch, relative
    ):
        # absolute paths so the runner is unaffected by the chdir below
        runner = LlamafileRunner(
            executable=os.path.abspath(executable),
            model=os.path.abspath(model) if model else None,
            gpu=gpu_mode,
        )
        monkeypatch.chdir(tmp_path)  # the server inherits it
        host = "server.sock" if relative else str(tmp_path / "server.sock")
        sock_path = str(tmp_path / "server.sock")
        if len(sock_path) >= 108:
            pytest.skip("tmp_path too long for sun_path")

        proc = runner.start_server(port=server_port, extra_args=["--host", host, "--unsecure"])
        try:
            deadline = time.time() + timeouts.server_ready
            status = None
            while time.time() < deadline and proc.poll() is None:
                if os.path.exists(sock_path):
                    try:
                        status, _ = _unix_http_get(sock_path, "/health", 5)
                        if status == 200:
                            break
                    except OSError:
                        pass
                time.sleep(0.5)
            assert status == 200, f"server did not answer /health on --host {host}"

            status, body = _unix_http_get(sock_path, "/v1/models", timeouts.http_request)
            assert status == 200 and b'"data"' in body
        finally:
            proc.terminate()
            proc.wait()
