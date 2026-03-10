"""Server mode integration tests."""

import pytest

from utils.llamafile import LlamafileRunner


@pytest.mark.server
class TestServerBasic:
    """Basic server mode tests."""

    def test_server_starts_and_responds(self, llamafile, server_port):
        """Test that server starts and responds to health check."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(server_port, timeout=120)
            assert ready, "Server did not become ready in time"
        finally:
            proc.terminate()
            proc.wait()

    def test_server_chat_completion(self, llamafile, server_port):
        """Test basic chat completion endpoint."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(server_port, timeout=120)
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello in one word."}],
            )

            assert "choices" in response
            assert len(response["choices"]) > 0
            content = response["choices"][0]["message"]["content"]
            assert len(content.strip()) > 0

        finally:
            proc.terminate()
            proc.wait()

    def test_server_chat_completion_math(self, llamafile, server_port):
        """Test chat completion with a math question."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(server_port, timeout=120)
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[
                    {"role": "user", "content": "What is 2+2? Answer with just the number."}
                ],
            )

            content = response["choices"][0]["message"]["content"]
            assert "4" in content

        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.server
class TestServerParameters:
    """Test server with various parameters."""

    def test_server_with_temperature(self, llamafile, server_port):
        """Test that temperature parameter is accepted."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(server_port, timeout=120)
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                temperature=0.0,
            )

            assert "choices" in response

        finally:
            proc.terminate()
            proc.wait()

    def test_server_with_max_tokens(self, llamafile, server_port):
        """Test that max_tokens parameter limits output."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(server_port, timeout=120)
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Count from 1 to 100"}],
                max_tokens=10,
            )

            # Output should be limited
            content = response["choices"][0]["message"]["content"]
            # With max_tokens=10, we shouldn't get to 100
            assert "100" not in content or len(content) < 50

        finally:
            proc.terminate()
            proc.wait()
