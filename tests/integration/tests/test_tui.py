"""TUI (chat) mode integration tests."""

import pytest


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
        input_file.write_text("What is 2+2?\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0
        # The response should contain "4" somewhere
        assert "4" in result.stdout

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
