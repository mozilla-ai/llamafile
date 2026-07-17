"""Server mode integration tests."""

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
