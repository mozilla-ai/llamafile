"""GPU acceleration integration tests."""

import os
import platform
import subprocess

import pytest

from utils.llamafile import LlamafileRunner


def get_available_gpu() -> str | None:
    """Detect available GPU type."""
    system = platform.system()

    if system == "Darwin":
        # Check for Apple Silicon
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if "Apple" in result.stdout:
                return "apple"
        except Exception:
            pass

    elif system == "Linux":
        # Check for NVIDIA
        if os.path.exists("/usr/bin/nvidia-smi"):
            try:
                subprocess.run(
                    ["nvidia-smi"], capture_output=True, check=True
                )
                return "nvidia"
            except Exception:
                pass

        # Check for AMD
        if os.path.exists("/opt/rocm"):
            return "amd"

    return None


@pytest.fixture
def available_gpu():
    """Fixture that provides the available GPU type or skips."""
    gpu = get_available_gpu()
    if gpu is None:
        pytest.skip("No GPU available")
    return gpu


@pytest.mark.gpu
class TestGPUAcceleration:
    """GPU acceleration tests."""

    def test_gpu_cli_responds(self, executable, model, available_gpu, timeouts):
        """Test that CLI works with GPU acceleration."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu=available_gpu,
        )

        result = runner.run_cli("Say hello", timeout=timeouts.cli)

        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_gpu_server_responds(self, executable, model, available_gpu, server_port, timeouts):
        """Test that server works with GPU acceleration."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu=available_gpu,
        )

        proc = runner.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                timeout=timeouts.http_request,
            )

            assert len(response["choices"][0]["message"]["content"]) > 0

        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.cpu
class TestCPUExecution:
    """CPU-only execution tests."""

    def test_cpu_cli_responds(self, executable, model, timeouts):
        """Test that CLI works without GPU (CPU only)."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu="disable",
        )

        result = runner.run_cli("Say hello", timeout=timeouts.cli)

        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_cpu_server_responds(self, executable, model, server_port, timeouts):
        """Test that server works without GPU (CPU only)."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu="disable",
        )

        proc = runner.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                timeout=timeouts.http_request,
            )

            assert len(response["choices"][0]["message"]["content"]) > 0

        finally:
            proc.terminate()
            proc.wait()
