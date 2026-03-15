"""Server mode integration tests."""

import pytest

from utils.llamafile import LlamafileRunner


@pytest.mark.server
class TestServerBasic:
    """Basic server mode tests."""

    def test_server_starts_and_responds(self, llamafile, server_port, timeouts):
        """Test that server starts and responds to health check."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
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
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello in one word."}],
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
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[
                    {"role": "user", "content": "What is 2+2? Answer with just the number."}
                ],
                timeout=timeouts.http_request,
            )

            content = response["choices"][0]["message"]["content"]
            assert "4" in content

        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.server
@pytest.mark.cli
class TestServerParameters:
    """Test server with various parameters."""

    def test_server_with_temperature_zero(self, llamafile, server_port, timeouts):
        """Test that temperature=0 produces consistent output."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            messages = [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ]

            # Use streaming with time limit to handle slow/thinking models
            content1 = LlamafileRunner.chat_completion_streaming(
                port=server_port,
                messages=messages,
                temperature=0.0,
                collect_timeout=20.0,
            )
            content2 = LlamafileRunner.chat_completion_streaming(
                port=server_port,
                messages=messages,
                temperature=0.0,
                collect_timeout=20.0,
            )

            # Compare the shorter response - it should match the prefix of the longer
            # (they may differ in length if one timed out earlier)
            min_len = min(len(content1), len(content2))
            assert min_len > 0, "No content received from either response"

            assert content1[:min_len] == content2[:min_len], (
                f"Expected consistent output with temperature=0.\n"
                f"Response 1: {content1[:200]!r}...\n"
                f"Response 2: {content2[:200]!r}..."
            )

        finally:
            proc.terminate()
            proc.wait()

    def test_server_with_max_tokens(self, llamafile, server_port, timeouts):
        """Test that max_tokens parameter limits output."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
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
