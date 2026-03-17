"""Multimodal (vision) integration tests."""

import pytest

from utils.llamafile import LlamafileRunner


@pytest.mark.multimodal
@pytest.mark.cli
class TestMultimodalCLI:
    """Multimodal tests using CLI mode with --image flag.

    Works with both pre-built llamafiles (mmproj bundled) and
    separate executable + model (requires --mmproj).
    """

    def _image_args(self, mmproj, image_path):
        """Build extra args for image CLI invocation."""
        args = ["--image", str(image_path)]
        if mmproj:
            args.extend(["--mmproj", mmproj])
        return args

    def test_cli_describe_image(self, llamafile, mmproj, test_image, timeouts):
        """Test that CLI can describe an image."""
        result = llamafile.run_cli(
            "Describe this image briefly.",
            extra_args=self._image_args(mmproj, test_image),
            timeout=timeouts.cli,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "No output generated"

    def test_cli_image_question(self, llamafile, mmproj, test_image, timeouts):
        """Test asking a specific question about an image."""
        result = llamafile.run_cli(
            "What colors do you see in this image?",
            extra_args=self._image_args(mmproj, test_image),
            timeout=timeouts.cli,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        output_lower = result.stdout.lower()
        color_words = ["red", "blue", "green", "white", "black", "yellow", "color"]
        assert any(color in output_lower for color in color_words)

    def test_cli_multiple_images_with_markers(self, llamafile, mmproj, test_image, timeouts):
        """Test multiple images with explicit markers in the prompt."""
        two_images = f"{test_image},{test_image}"
        result = llamafile.run_cli(
            "<__media__> Describe the first image. <__media__> Describe the second image.",
            extra_args=self._image_args(mmproj, two_images),
            timeout=timeouts.cli,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "No output generated"

    def test_cli_multiple_images_marker_mismatch(self, llamafile, mmproj, test_image, timeouts):
        """Test that mismatched marker count and image count gives a clear error."""
        two_images = f"{test_image},{test_image}"
        result = llamafile.run_cli(
            "<__media__> Only one marker but two images.",
            extra_args=self._image_args(mmproj, two_images),
            timeout=timeouts.cli,
        )

        assert result.returncode != 0, "Should fail with marker/image mismatch"
        assert "markers" in result.stderr.lower() or "match" in result.stderr.lower()


@pytest.mark.multimodal
@pytest.mark.tui
class TestMultimodalTUI:
    """Multimodal tests using TUI mode with /upload command."""

    def _mmproj_args(self, mmproj):
        """Build extra args for mmproj if provided."""
        if mmproj:
            return ["--mmproj", mmproj]
        return None

    def test_tui_describe_image(self, llamafile, mmproj, test_image, tmp_path, timeouts):
        """Test that TUI can describe an uploaded image."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(f"/upload {test_image}\nDescribe this image briefly.\n/exit\n")

        result = llamafile.run_tui(str(input_file), extra_args=self._mmproj_args(mmproj),
                                   timeout=timeouts.tui)

        assert result.returncode == 0, f"TUI failed: {result.stderr}"
        # Should have generated some description
        assert len(result.stdout) > 0

    def test_tui_image_question(self, llamafile, mmproj, test_image, tmp_path, timeouts):
        """Test asking a specific question about an image."""
        input_file = tmp_path / "input.txt"
        input_file.write_text(
            f"/upload {test_image}\nWhat colors do you see in this image?\n/exit\n"
        )

        result = llamafile.run_tui(str(input_file), extra_args=self._mmproj_args(mmproj),
                                   timeout=timeouts.tui)

        assert result.returncode == 0
        # Should mention some color
        output_lower = result.stdout.lower()
        color_words = ["red", "blue", "green", "white", "black", "yellow", "color"]
        assert any(color in output_lower for color in color_words)


@pytest.mark.multimodal
@pytest.mark.server
class TestMultimodalServer:
    """Multimodal tests using server mode with OpenAI API."""

    def _mmproj_args(self, mmproj):
        """Build extra args for mmproj if provided."""
        if mmproj:
            return ["--mmproj", mmproj]
        return None

    def test_server_describe_image(self, llamafile, mmproj, test_image, server_port, timeouts):
        """Test image description via server API."""
        proc = llamafile.start_server(port=server_port,
                                      extra_args=self._mmproj_args(mmproj))

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

    def test_server_image_question(self, llamafile, mmproj, test_image, server_port, timeouts):
        """Test asking a specific question about an image via server."""
        proc = llamafile.start_server(port=server_port,
                                      extra_args=self._mmproj_args(mmproj))

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
