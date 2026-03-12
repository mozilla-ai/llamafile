"""Multimodal (vision) integration tests."""

import pytest

from utils.llamafile import LlamafileRunner


@pytest.mark.multimodal
class TestMultimodalCLI:
    """Multimodal tests using CLI mode."""

    # Note: CLI multimodal would need --image flag or similar
    # This depends on how llamafile CLI handles images
    pass


@pytest.mark.multimodal
@pytest.mark.tui
class TestMultimodalTUI:
    """Multimodal tests using TUI mode with /upload command."""

    def test_tui_describe_image(self, llamafile, test_image, tmp_path, timeouts):
        """Test that TUI can describe an uploaded image."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(f"/upload {test_image}\nDescribe this image briefly.\n/exit\n")

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0, f"TUI failed: {result.stderr}"
        # Should have generated some description
        assert len(result.stdout) > 0

    def test_tui_image_question(self, llamafile, test_image, tmp_path, timeouts):
        """Test asking a specific question about an image."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(
            f"/upload {test_image}\nWhat colors do you see in this image?\n/exit\n"
        )

        result = llamafile.run_tui(str(input_file), timeout=timeouts.tui)

        assert result.returncode == 0
        # Should mention some color
        output_lower = result.stdout.lower()
        color_words = ["red", "blue", "green", "white", "black", "yellow", "color"]
        assert any(color in output_lower for color in color_words)


@pytest.mark.multimodal
@pytest.mark.server
class TestMultimodalServer:
    """Multimodal tests using server mode with OpenAI API."""

    def test_server_describe_image(self, llamafile, test_image, server_port, timeouts):
        """Test image description via server API."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready, "Server did not become ready"

            response = LlamafileRunner.chat_completion_with_image(
                port=server_port,
                prompt="Describe this image in one sentence.",
                image_path=str(test_image),
                timeout=timeouts.http_request,
            )

            content = response["choices"][0]["message"]["content"]
            assert len(content.strip()) > 0

        finally:
            proc.terminate()
            proc.wait()

    def test_server_image_question(self, llamafile, test_image, server_port, timeouts):
        """Test asking a specific question about an image via server."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion_with_image(
                port=server_port,
                prompt="What colors are present in this image?",
                image_path=str(test_image),
                timeout=timeouts.http_request,
            )

            content = response["choices"][0]["message"]["content"].lower()
            color_words = ["red", "blue", "green", "white", "black", "yellow", "color"]
            assert any(color in content for color in color_words)

        finally:
            proc.terminate()
            proc.wait()
