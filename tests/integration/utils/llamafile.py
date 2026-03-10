"""Llamafile process runner for integration tests."""

import base64
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


class LlamafileRunner:
    """Wrapper for running llamafile in different modes.

    Supports both direct builds (executable + model) and pre-built llamafiles.

    Examples:
        # Direct build
        runner = LlamafileRunner("./o/llamafile/llamafile", model="model.gguf")

        # Pre-built llamafile
        runner = LlamafileRunner("./Qwen-QwQ.llamafile")
    """

    # On macOS, llamafiles need to be run via sh due to cosmopolitan format
    USE_SHELL = platform.system() == "Darwin"

    def __init__(
        self,
        executable: str,
        model: str | None = None,
        gpu: str | None = None,
    ):
        """Initialize the runner.

        Args:
            executable: Path to llamafile binary or pre-built .llamafile
            model: Path to model file (None for pre-built llamafiles)
            gpu: GPU mode - "auto", "apple", "amd", "nvidia", or None for CPU
        """
        self.executable = os.path.abspath(executable)
        self.model = os.path.abspath(model) if model else None
        self.gpu = gpu

        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Executable not found: {executable}")
        if self.model and not os.path.exists(self.model):
            raise FileNotFoundError(f"Model not found: {model}")

    def _base_args(self) -> list[str]:
        """Build base command arguments.

        On macOS, prepends 'sh' to run llamafiles via shell.
        """
        if self.USE_SHELL:
            args = ["sh", self.executable]
        else:
            args = [self.executable]
        if self.model:
            args.extend(["-m", self.model])
        if self.gpu:
            args.extend(["--gpu", self.gpu])
        return args

    def run_cli(
        self,
        prompt: str,
        nothink: bool = False,
        extra_args: list[str] | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        """Run llamafile in CLI mode with a prompt.

        Args:
            prompt: The prompt to send
            nothink: If True, disable thinking output
            extra_args: Additional command-line arguments
            timeout: Timeout in seconds

        Returns:
            CompletedProcess with stdout, stderr, returncode
        """
        args = self._base_args()
        args.extend(["--cli", "-p", prompt])

        if nothink:
            args.append("--nothink")

        if extra_args:
            args.extend(extra_args)

        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_tui(
        self,
        input_file: str,
        extra_args: list[str] | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        """Run llamafile in TUI/chat mode with piped input.

        Args:
            input_file: Path to file containing input to pipe to stdin
            extra_args: Additional command-line arguments
            timeout: Timeout in seconds

        Returns:
            CompletedProcess with stdout, stderr, returncode
        """
        args = self._base_args()
        args.append("--chat")

        if extra_args:
            args.extend(extra_args)

        with open(input_file, "r") as f:
            input_data = f.read()

        return subprocess.run(
            args,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def start_server(
        self,
        port: int = 8080,
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen:
        """Start llamafile in server mode.

        Args:
            port: Port to listen on
            extra_args: Additional command-line arguments

        Returns:
            Popen process handle (caller must terminate)
        """
        args = self._base_args()
        args.extend(["--server", "--port", str(port)])

        if extra_args:
            args.extend(extra_args)

        return subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def start_combined(
        self,
        port: int = 8080,
        extra_args: list[str] | None = None,
    ) -> subprocess.Popen:
        """Start llamafile in combined TUI+Server mode (default mode).

        Args:
            port: Port for the server component
            extra_args: Additional command-line arguments

        Returns:
            Popen process handle (caller must terminate)
        """
        args = self._base_args()
        args.extend(["--port", str(port)])

        if extra_args:
            args.extend(extra_args)

        return subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def wait_for_server(
        port: int,
        host: str = "127.0.0.1",
        timeout: int = 120,
        poll_interval: float = 0.5,
    ) -> bool:
        """Wait for server to become ready.

        Args:
            port: Server port
            host: Server host
            timeout: Maximum time to wait in seconds
            poll_interval: Time between health checks

        Returns:
            True if server is ready, False if timeout
        """
        url = f"http://{host}:{port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)

        return False

    @staticmethod
    def chat_completion(
        port: int,
        messages: list[dict[str, Any]],
        host: str = "127.0.0.1",
        stream: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a chat completion request to the server.

        Args:
            port: Server port
            messages: List of message dicts with "role" and "content"
            host: Server host
            stream: Whether to stream the response
            timeout: Request timeout
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response JSON as dict
        """
        url = f"http://{host}:{port}/v1/chat/completions"
        payload = {
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def chat_completion_with_image(
        port: int,
        prompt: str,
        image_path: str,
        host: str = "127.0.0.1",
        timeout: int = 60,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a multimodal chat completion with an image.

        Args:
            port: Server port
            prompt: Text prompt
            image_path: Path to image file
            host: Server host
            timeout: Request timeout
            **kwargs: Additional parameters

        Returns:
            Response JSON as dict
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect MIME type
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        },
                    },
                ],
            }
        ]

        return LlamafileRunner.chat_completion(
            port=port,
            messages=messages,
            host=host,
            timeout=timeout,
            **kwargs,
        )
