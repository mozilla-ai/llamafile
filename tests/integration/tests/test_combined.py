"""Combined TUI+Server mode integration tests."""

import pytest

from utils.llamafile import LlamafileRunner, read_until_idle, stop_tui


@pytest.mark.tui
@pytest.mark.server
@pytest.mark.combined
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
            stop_tui(proc)

    def test_combined_tui_and_server_simultaneously(self, llamafile, server_port, timeouts):
        """Test that both TUI and server can be used at the same time."""
        proc = llamafile.start_combined(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready"

            # Clear any startup output from TUI
            _ = read_until_idle(proc.stdout, idle_timeout=0.5, max_timeout=5.0)

            # Test 1: Send a request via server API
            response1 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "What is 1+1?"}],
                timeout=timeouts.http_request,
            )
            assert "2" in response1["choices"][0]["message"]["content"]

            # Test 2: Send TUI input and verify response
            proc.stdin.write("What is 2+2?\n")
            proc.stdin.flush()

            # Read TUI output until model stops generating
            tui_output = read_until_idle(
                proc.stdout,
                idle_timeout=2.0 * timeouts.multiplier,
                max_timeout=timeouts.cli,
            )
            assert len(tui_output) > 0, "TUI produced no output"
            assert "4" in tui_output, f"Expected '4' in TUI output: {tui_output}"

            # Test 3: Server should still work after TUI interaction
            response2 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "What is 3+3?"}],
                timeout=timeouts.http_request,
            )
            assert "6" in response2["choices"][0]["message"]["content"]

        finally:
            stop_tui(proc)
