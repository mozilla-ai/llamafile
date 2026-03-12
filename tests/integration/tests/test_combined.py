"""Combined TUI+Server mode integration tests."""

import time

import pytest

from utils.llamafile import LlamafileRunner


@pytest.mark.tui
@pytest.mark.server
class TestCombinedMode:
    """Tests for simultaneous TUI and Server mode."""

    def test_combined_server_responds(self, llamafile, server_port, timeouts):
        """Test that server works in combined mode."""
        proc = llamafile.start_combined(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready in combined mode"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                timeout=timeouts.http_request,
            )

            assert "choices" in response
            assert len(response["choices"][0]["message"]["content"]) > 0

        finally:
            proc.terminate()
            proc.wait()

    def test_combined_tui_and_server_simultaneously(self, llamafile, server_port, timeouts):
        """Test that both TUI and server can be used at the same time."""
        proc = llamafile.start_combined(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready"

            # Send a request via server API
            response1 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "What is 1+1?"}],
                timeout=timeouts.http_request,
            )
            assert "2" in response1["choices"][0]["message"]["content"]

            # Send TUI input
            proc.stdin.write("What is 2+2?\n")
            proc.stdin.flush()

            # Give it time to process
            time.sleep(2 * timeouts.multiplier)

            # Send another server request (should still work)
            response2 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "What is 3+3?"}],
                timeout=timeouts.http_request,
            )
            assert "6" in response2["choices"][0]["message"]["content"]

        finally:
            proc.terminate()
            proc.wait()
