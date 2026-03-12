"""CLI mode integration tests."""

import pytest


@pytest.mark.cli
class TestCLIBasic:
    """Basic CLI mode tests."""

    def test_cli_responds(self, llamafile, timeouts):
        """Test that CLI mode starts and generates a response."""
        result = llamafile.run_cli("Say hello in exactly one word.", timeout=timeouts.cli)

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "No output generated"

    def test_cli_math_question(self, llamafile, timeouts):
        """Test that CLI can answer a simple math question."""
        result = llamafile.run_cli(
            "What is 2+2? Answer with just the number.", timeout=timeouts.cli
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "4" in result.stdout, f"Expected '4' in output: {result.stdout}"

    def test_cli_exits_cleanly(self, llamafile, timeouts):
        """Test that CLI exits with code 0 after completion."""
        result = llamafile.run_cli("Say OK", timeout=timeouts.cli)

        assert result.returncode == 0


@pytest.mark.cli
@pytest.mark.thinking
class TestCLIThinking:
    """CLI tests for thinking models."""

    def test_thinking_enabled(self, llamafile, timeouts):
        """Test that the model has thinking enabled (otherwise --nothink tests make no sense)."""
        prompt = "What is 2+2? Think step by step then give the answer."

        # With thinking
        result_think = llamafile.run_cli(prompt, nothink=False, timeout=timeouts.cli)

        assert result_think.returncode == 0

        # nothink output should not contain think tags
        assert "<think>" in result_think.stdout

    def test_nothink_removes_thinking(self, llamafile, timeouts):
        """Test that --nothink removes thinking content from output."""
        prompt = "What is 2+2? Think step by step then give the answer."

        # Without thinking
        result_nothink = llamafile.run_cli(prompt, nothink=True, timeout=timeouts.cli)

        assert result_nothink.returncode == 0

        # nothink output should not contain think tags
        assert "<think>" not in result_nothink.stdout

    def test_nothink_shorter_output(self, llamafile, timeouts):
        """Test that --nothink produces shorter output (no thinking tokens)."""
        prompt = "Explain briefly why the sky is blue."

        result_think = llamafile.run_cli(prompt, nothink=False, timeout=timeouts.cli)
        result_nothink = llamafile.run_cli(prompt, nothink=True, timeout=timeouts.cli)

        # nothink should generally be shorter (no thinking block)
        # This may not always hold for very short responses
        if "<think>" in result_think.stdout:
            assert len(result_nothink.stdout) <= len(result_think.stdout)
