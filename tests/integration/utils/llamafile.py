"""Llamafile process runner for integration tests."""

import base64
import fcntl
import logging
import os
import platform
import select
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


def read_until_idle(fd, idle_timeout=1.0, max_timeout=60.0):
    """Read from file descriptor until output stops (model finished generating).

    Useful for reading streaming TUI output where tokens arrive one at a time.

    Args:
        fd: File object to read from (e.g., proc.stdout)
        idle_timeout: Time to wait with no new output before considering done
        max_timeout: Maximum total time to wait

    Returns:
        String of all collected output
    """
    fileno = fd.fileno()
    flags = fcntl.fcntl(fileno, fcntl.F_GETFL)
    fcntl.fcntl(fileno, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    chunks = []
    start_time = time.time()
    last_read_time = start_time

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                break

            idle_time = time.time() - last_read_time
            if idle_time > idle_timeout and chunks:
                # No new output for idle_timeout and we have some output
                break

            ready, _, _ = select.select([fileno], [], [], 0.1)
            if ready:
                try:
                    chunk = fd.read(4096)
                    if chunk:
                        chunks.append(chunk)
                        last_read_time = time.time()
                except (BlockingIOError, IOError):
                    pass

        return "".join(chunks)
    finally:
        # Restore blocking mode
        fcntl.fcntl(fileno, fcntl.F_SETFL, flags)


def stop_tui(proc, timeout=30):
    """Stop a process that has a TUI reading stdin.

    Sends /exit via stdin for a clean shutdown. Falls back to kill
    if the process doesn't exit in time.

    Args:
        proc: subprocess.Popen with stdin pipe
        timeout: Seconds to wait before falling back to kill
    """
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.write("/exit\n")
            proc.stdin.flush()
        proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
        proc.wait()


# Default timeout constants (in seconds)
TIMEOUT_CLI = 120
TIMEOUT_TUI = 120
TIMEOUT_SERVER_READY = 120
TIMEOUT_HTTP_REQUEST = 60
POLL_INTERVAL = 0.5


class LlamafileRunner:
    """Wrapper for running llamafile in different modes.

    Supports both direct builds (executable + model) and pre-built llamafiles.

    Examples:
        # Direct build
        runner = LlamafileRunner("./o/llamafile/llamafile", model="model.gguf")

        # Pre-built llamafile
        runner = LlamafileRunner("./Qwen-QwQ.llamafile")
    """

    # On Unix (macOS, Linux, BSD), run llamafiles via sh for portability.
    # Direct execution on Linux requires binfmt_misc configured for APE binaries.
    # On Windows, cosmopolitan binaries run directly (self-extract to .exe).
    USE_SHELL = platform.system() != "Windows"

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

        On Unix, prepends 'sh' to run llamafiles via shell for compatibility.
        On Windows, runs directly since cosmopolitan binaries self-extract.
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
        timeout: float = TIMEOUT_CLI,
        log_file: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run llamafile in CLI mode with a prompt.

        Args:
            prompt: The prompt to send
            nothink: If True, disable thinking output
            extra_args: Additional command-line arguments
            timeout: Timeout in seconds
            log_file: If provided, adds --log-file flag and stores log content
                      in result.log_output attribute after execution

        Returns:
            CompletedProcess with stdout, stderr, returncode.
            If log_file was provided, also has log_output attribute.
        """
        args = self._base_args()
        args.extend(["--cli", "-p", prompt])

        if nothink:
            args.append("--nothink")

        if log_file:
            args.extend(["--log-file", log_file])

        if extra_args:
            args.extend(extra_args)

        logger.info("CLI command: %s (timeout=%.1fs)", " ".join(args), timeout)
        result = subprocess.run(
            args,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        logger.info("CLI exit code: %d", result.returncode)
        logger.debug("CLI stdout:\n%s", result.stdout)
        if result.stderr:
            logger.debug("CLI stderr:\n%s", result.stderr)

        # Read log file if provided
        if log_file and os.path.exists(log_file):
            with open(log_file, "r", errors="replace") as f:
                result.log_output = f.read()
            logger.debug("Log file contents:\n%s", result.log_output)
        elif log_file:
            result.log_output = ""

        return result

    def run_tui(
        self,
        input_file: str,
        extra_args: list[str] | None = None,
        timeout: float = TIMEOUT_TUI,
        log_file: str | None = None,
    ) -> subprocess.CompletedProcess:
        """Run llamafile in TUI/chat mode with piped input.

        Args:
            input_file: Path to file containing input to pipe to stdin
            extra_args: Additional command-line arguments
            timeout: Timeout in seconds
            log_file: If provided, adds --log-file flag and stores log content
                      in result.log_output attribute after execution

        Returns:
            CompletedProcess with stdout, stderr, returncode.
            If log_file was provided, also has log_output attribute.
        """
        args = self._base_args()
        args.append("--chat")

        if log_file:
            args.extend(["--log-file", log_file])

        if extra_args:
            args.extend(extra_args)

        with open(input_file, "r") as f:
            input_data = f.read()

        logger.info("TUI command: %s (timeout=%.1fs)", " ".join(args), timeout)
        logger.debug("TUI input:\n%s", input_data)
        result = subprocess.run(
            args,
            input=input_data,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        logger.info("TUI exit code: %d", result.returncode)
        logger.debug("TUI stdout:\n%s", result.stdout)
        if result.stderr:
            logger.debug("TUI stderr:\n%s", result.stderr)

        # Read log file if provided
        if log_file and os.path.exists(log_file):
            with open(log_file, "r", errors="replace") as f:
                result.log_output = f.read()
            logger.debug("Log file contents:\n%s", result.log_output)
        elif log_file:
            result.log_output = ""

        return result

    def start_server(
        self,
        port: int = 8080,
        extra_args: list[str] | None = None,
        log_file: str | None = None,
    ) -> subprocess.Popen:
        """Start llamafile in server mode.

        Args:
            port: Port to listen on
            extra_args: Additional command-line arguments
            log_file: If provided, adds --log-file flag. Caller should read
                      the file after terminating the process.

        Returns:
            Popen process handle (caller must terminate)
        """
        args = self._base_args()
        args.extend(["--server", "--port", str(port)])

        if log_file:
            args.extend(["--log-file", log_file])

        if extra_args:
            args.extend(extra_args)

        logger.info("Starting server: %s", " ".join(args))
        return subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def start_combined(
        self,
        port: int = 8080,
        extra_args: list[str] | None = None,
        log_file: str | None = None,
    ) -> subprocess.Popen:
        """Start llamafile in combined TUI+Server mode (default mode).

        Args:
            port: Port for the server component
            extra_args: Additional command-line arguments
            log_file: If provided, adds --log-file flag. Caller should read
                      the file after terminating the process.

        Returns:
            Popen process handle (caller must terminate)
        """
        args = self._base_args()
        args.extend(["--port", str(port)])

        if log_file:
            args.extend(["--log-file", log_file])

        if extra_args:
            args.extend(extra_args)

        logger.info("Starting combined mode: %s", " ".join(args))
        return subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def read_log_file(log_file: str) -> str:
        """Read contents of a log file.

        Useful for reading log files after Popen processes terminate.

        Args:
            log_file: Path to the log file

        Returns:
            Log file contents, or empty string if file doesn't exist
        """
        if os.path.exists(log_file):
            with open(log_file, "r", errors="replace") as f:
                return f.read()
        return ""

    @staticmethod
    def wait_for_server(
        port: int,
        host: str = "127.0.0.1",
        timeout: float = TIMEOUT_SERVER_READY,
        poll_interval: float = POLL_INTERVAL,
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

        logger.info("Waiting for server at %s (timeout=%.1fs)", url, timeout)
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logger.info("Server ready")
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval)

        logger.warning("Server not ready after %.1fs", timeout)
        return False

    @staticmethod
    def chat_completion(
        port: int,
        messages: list[dict[str, Any]],
        host: str = "127.0.0.1",
        stream: bool = False,
        timeout: float = TIMEOUT_HTTP_REQUEST,
        retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Send a chat completion request to the server.

        Args:
            port: Server port
            messages: List of message dicts with "role" and "content"
            host: Server host
            stream: Whether to stream the response
            timeout: Request timeout
            retries: Number of retries on connection errors
            retry_delay: Delay between retries in seconds
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

        logger.info("POST %s (timeout=%.1fs)", url, timeout)
        logger.debug("Request payload: %s", payload)

        last_error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                logger.debug("Response: %s", result)
                return result
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < retries:
                    logger.warning(
                        "Connection error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, retries + 1, retry_delay, e
                    )
                    time.sleep(retry_delay)
                else:
                    raise
        raise last_error  # Should not reach here, but for type safety

    @staticmethod
    def chat_completion_streaming(
        port: int,
        messages: list[dict[str, Any]],
        host: str = "127.0.0.1",
        collect_timeout: float = 20.0,
        include_reasoning: bool = True,
        **kwargs,
    ) -> str:
        """Send a streaming chat completion and collect content up to a time limit.

        Useful for testing with large/slow models where you want to compare
        partial outputs without waiting for full completion.

        Args:
            port: Server port
            messages: List of message dicts with "role" and "content"
            host: Server host
            collect_timeout: Max time to collect streaming content (seconds)
            include_reasoning: If True, also collect reasoning_content from
                               thinking models (default True)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Collected content string (may be partial if timeout reached)
        """
        import json as json_module

        url = f"http://{host}:{port}/v1/chat/completions"
        payload = {
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        logger.info(
            "POST %s (streaming, collect_timeout=%.1fs)", url, collect_timeout
        )
        logger.debug("Request payload: %s", payload)

        content = ""
        start_time = time.time()

        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if time.time() - start_time > collect_timeout:
                    logger.info("Collect timeout reached after %.1fs", collect_timeout)
                    break

                if not line:
                    continue

                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue

                data = line_str[6:]  # Strip "data: " prefix
                if data == "[DONE]":
                    break

                try:
                    chunk = json_module.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Collect both content and reasoning_content (for thinking models)
                    chunk_content = delta.get("content")
                    if chunk_content:
                        content += chunk_content
                    if include_reasoning:
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            content += reasoning
                except json_module.JSONDecodeError:
                    continue

        logger.debug("Collected content (len=%d): %s", len(content), content[:200])
        return content

    @staticmethod
    def chat_completion_with_image(
        port: int,
        prompt: str,
        image_path: str,
        host: str = "127.0.0.1",
        timeout: float = TIMEOUT_HTTP_REQUEST,
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
