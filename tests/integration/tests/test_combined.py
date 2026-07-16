"""Combined TUI+Server mode integration tests."""

import pytest

from utils.llamafile import LlamafileRunner, read_until_idle, stop_tui
from utils.prompts import ADD_1_1, ADD_2_2, ADD_3_3, GREETING_PROMPT


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
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready in combined mode"

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": GREETING_PROMPT}],
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
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready"

            # Clear any startup output from TUI
            _ = read_until_idle(proc.stdout, idle_timeout=0.5, max_timeout=5.0)

            # Test 1: Send a request via server API
            response1 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": ADD_1_1.prompt}],
                timeout=timeouts.http_request,
            )
            content1 = response1["choices"][0]["message"]["content"]
            assert ADD_1_1.check(content1), f"Expected {ADD_1_1.describe()} in content: {content1}"

            # Test 2: Send TUI input and verify response
            proc.stdin.write(f"{ADD_2_2.prompt}\n")
            proc.stdin.flush()

            # Read TUI output until model stops generating
            tui_output = read_until_idle(
                proc.stdout,
                idle_timeout=2.0 * timeouts.multiplier,
                max_timeout=timeouts.cli,
            )
            assert len(tui_output) > 0, "TUI produced no output"
            assert ADD_2_2.check(tui_output), (
                f"Expected {ADD_2_2.describe()} in TUI output: {tui_output}"
            )

            # Test 3: Server should still work after TUI interaction
            response2 = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": ADD_3_3.prompt}],
                timeout=timeouts.http_request,
            )
            content2 = response2["choices"][0]["message"]["content"]
            assert ADD_3_3.check(content2), f"Expected {ADD_3_3.describe()} in content: {content2}"

        finally:
            stop_tui(proc)
