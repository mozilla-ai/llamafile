"""Combined TUI+Server mode integration tests."""

import pytest

from utils.llamafile import (
    LlamafileRunner,
    STATS_IDLE_LINE,
    assert_stats_core,
    read_until_idle,
    stop_tui,
    strip_ansi,
)
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


@pytest.mark.tui
@pytest.mark.server
@pytest.mark.combined
class TestCombinedStats:
    """/stats in combined mode, served by ApiBackend over HTTP."""

    def _stats_output(self, llamafile, server_port, timeouts, generate_first):
        """Drive the combined-mode TUI and return its output after /stats."""
        proc = llamafile.start_combined(port=server_port)
        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            )
            assert ready, "Server did not become ready in combined mode"

            # discard startup banner and model info block
            _ = read_until_idle(proc.stdout, idle_timeout=0.5, max_timeout=5.0)

            if generate_first:
                proc.stdin.write(f"{ADD_2_2.prompt}\n")
                proc.stdin.flush()
                read_until_idle(
                    proc.stdout,
                    idle_timeout=2.0 * timeouts.multiplier,
                    max_timeout=timeouts.cli,
                )

            proc.stdin.write("/stats\n")
            proc.stdin.flush()
            return read_until_idle(
                proc.stdout,
                idle_timeout=2.0 * timeouts.multiplier,
                max_timeout=timeouts.cli,
            )
        finally:
            stop_tui(proc)

    def test_stats_reports_perf_metrics(self, llamafile, server_port, timeouts):
        """/stats prints the same core block the --chat backend prints.

        ApiBackend reads these from the timings object the server attaches
        to the final SSE chunk; before unification it reported only token
        counts, with no durations or rates at all.
        """
        output = self._stats_output(
            llamafile, server_port, timeouts, generate_first=True
        )

        assert len(output) > 0, "TUI produced no output for /stats"
        assert_stats_core(output, where="combined mode")

    def test_stats_omits_direct_backend_extras(self, llamafile, server_port, timeouts):
        """Rows the server cannot source must not appear in combined mode."""
        output = self._stats_output(
            llamafile, server_port, timeouts, generate_first=True
        )
        # assert the core rendered first, so this cannot pass merely because
        # /stats printed nothing worth inspecting
        assert_stats_core(output, where="combined mode")
        clean = strip_ansi(output)

        for row in ("load", "graphs", "samplers"):
            assert f"  {row}" not in clean, (
                f"combined-mode /stats printed --chat-only row '{row}': {clean}"
            )

    def test_stats_when_idle_reports_nothing(self, llamafile, server_port, timeouts):
        """An idle /stats reads the same here as it does in --chat."""
        clean = strip_ansi(
            self._stats_output(llamafile, server_port, timeouts, generate_first=False)
        )

        assert STATS_IDLE_LINE in clean, f"/stats did not report an idle session: {clean}"
