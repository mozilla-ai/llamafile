"""TUI (chat) mode integration tests."""

import pytest

from utils.llamafile import STATS_IDLE_LINE, assert_stats_core, strip_ansi
from utils.prompts import ADD_2_2


@pytest.mark.tui
class TestTUIBasic:
    """Basic TUI mode tests with piped input."""

    def test_tui_responds_to_hello(self, llamafile, tmp_path, timeouts):
        """Test that TUI accepts input and generates a response."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Hello!\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0, f"TUI failed: {result.stderr}"
        # Should have some output (model response or at least the UI)
        assert len(result.stdout) > 0

    def test_tui_math_question(self, llamafile, tmp_path, timeouts):
        """Test TUI with a simple math question."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(f"{ADD_2_2.prompt}\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0
        assert ADD_2_2.check(result.stdout), (
            f"Expected {ADD_2_2.describe()} in output: {result.stdout}"
        )

    def test_tui_multi_turn(self, llamafile, tmp_path, timeouts):
        """Test TUI with multiple turns of conversation."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("My name is Alice.\nWhat is my name?\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0
        # Model should remember the name
        assert "Alice" in result.stdout

    def test_tui_exits_cleanly(self, llamafile, tmp_path, timeouts):
        """Test that /exit command works."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0


@pytest.mark.tui
class TestTUIStats:
    """/stats in --chat mode, served by DirectBackend."""

    def test_stats_reports_perf_metrics(self, llamafile, tmp_path, timeouts):
        """/stats prints the shared core block after generating.

        Regression test for #1047, where this printed nothing at all: the
        metrics went through LOG_INF(), which chat mode silences.
        """
        input_file = tmp_path / "input.txt"
        input_file.write_text(f"{ADD_2_2.prompt}\n/stats\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0, f"TUI failed: {result.stderr}"
        assert_stats_core(result.stdout, where="--chat")

    def test_stats_reports_direct_backend_extras(self, llamafile, tmp_path, timeouts):
        """--chat adds load time and graph reuse, which the server has no equivalent for."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(f"{ADD_2_2.prompt}\n/stats\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)
        clean = strip_ansi(result.stdout)

        assert result.returncode == 0
        for row in ("load", "graphs"):
            assert f"  {row}" in clean, f"/stats missing '{row}' row: {clean}"

    def test_stats_when_idle_reports_nothing(self, llamafile, tmp_path, timeouts):
        """/stats before generating says so, and prints no rows of its own.

        llama_perf_context() clamps its token counts with std::max(1, ...),
        so an unguarded implementation shows a phantom token on each row.
        """
        input_file = tmp_path / "input.txt"
        input_file.write_text("/stats\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)
        clean = strip_ansi(result.stdout)

        assert result.returncode == 0
        assert STATS_IDLE_LINE in clean, f"/stats did not report an idle session: {clean}"
        # no per-mode extras belong under an idle report
        for row in ("load", "graphs", "samplers"):
            assert f"  {row}" not in clean, f"idle /stats printed '{row}' row: {clean}"


@pytest.mark.tui
@pytest.mark.thinking
class TestTUIThinking:
    """TUI tests for thinking model styling."""

    def test_tui_thinking_visible(self, llamafile, tmp_path, timeouts):
        """Test that thinking content is visible in TUI output.

        For thinking models, the <think> block should be displayed
        with appropriate styling (darker color).
        """
        input_file = tmp_path / "input.txt"
        input_file.write_text("What is 15 * 23? Think carefully.\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0
        # Output should contain something (either thinking or answer)
        assert len(result.stdout) > 0

        # If it's a thinking model, there might be ANSI escape codes
        # for the darker styling, or <think> tags visible
        # This is a basic check - more specific checks depend on terminal output
