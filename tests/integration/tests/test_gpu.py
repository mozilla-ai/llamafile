"""GPU acceleration integration tests."""

import os
import platform
import subprocess
import tempfile

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


def check_gpu_in_output(log_output: str) -> dict:
    """Check log output for GPU usage indicators.

    Uses inverted logic: detects GPU offloading rather than
    trying to parse all possible GPU types. This is more robust
    because new GPU types don't need special handling.

    Key patterns:
    - GPU enabled: "using device MTL0 (Apple M3 Max)"
                   "load_tensors: offloading X layers to GPU"
                   "load_tensors: offloaded X/Y layers to GPU"
    - CPU only: No "using device" or "offloading" messages

    Args:
        log_output: Log file contents from llamafile with --log-file

    Returns:
        Dict with:
        - 'devices': list of devices found (e.g., ["MTL0"], ["CPU"])
        - 'gpu_used': True if GPU offloading is detected
        - 'layers_offloaded': tuple of (offloaded, total) or None
    """
    devices = []
    gpu_used = False
    layers_offloaded = None

    for line in log_output.split("\n"):
        # Pattern: "using device MTL0 (Apple M3 Max)" or "using device CUDA0 (...)"
        if "using device" in line and "llama_model_load" in line:
            # Extract device name after "using device "
            parts = line.split("using device")
            if len(parts) > 1:
                device = parts[1].strip().split()[0]
                if device and device not in devices:
                    devices.append(device)
                gpu_used = True

        # Pattern: "load_tensors: offloaded 33/33 layers to GPU"
        if "offloaded" in line and "layers to GPU" in line:
            gpu_used = True
            # Try to extract X/Y
            parts = line.split("offloaded")
            if len(parts) > 1:
                fraction = parts[1].strip().split()[0]
                if "/" in fraction:
                    try:
                        offloaded, total = fraction.split("/")
                        layers_offloaded = (int(offloaded), int(total))
                    except (ValueError, IndexError):
                        pass

        # Pattern: "load_tensors: offloading X layers to GPU"
        if "offloading" in line and "to GPU" in line:
            gpu_used = True

    # If no GPU device found, assume CPU
    if not devices:
        devices = ["CPU"]

    return {
        "devices": devices,
        "gpu_used": gpu_used,
        "layers_offloaded": layers_offloaded,
    }


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

    def test_gpu_cli(self, executable, model, available_gpu, timeouts):
        """Test CLI mode with GPU: verify GPU is used and response works."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu=available_gpu,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            log_path = f.name

        try:
            result = runner.run_cli(
                "Say hello",
                timeout=timeouts.cli,
                log_file=log_path,
                extra_args=["--verbose"],
            )

            # Verify response works
            assert result.returncode == 0
            assert len(result.stdout.strip()) > 0

            # Verify GPU is used
            gpu_info = check_gpu_in_output(result.log_output)
            assert gpu_info["gpu_used"], (
                f"No GPU offloading detected. Devices: {gpu_info['devices']}"
            )
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_gpu_server(self, executable, model, available_gpu, server_port, timeouts):
        """Test server mode with GPU: verify GPU is used and response works."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu=available_gpu,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            log_path = f.name

        proc = runner.start_server(port=server_port, log_file=log_path,
                                    extra_args=["--verbose"])

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            # Verify response works
            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                timeout=timeouts.http_request,
            )
            assert len(response["choices"][0]["message"]["content"]) > 0

        finally:
            proc.terminate()
            proc.wait()

            # Verify GPU is used
            log_output = LlamafileRunner.read_log_file(log_path)
            gpu_info = check_gpu_in_output(log_output)
            assert gpu_info["gpu_used"], (
                f"No GPU offloading detected. Devices: {gpu_info['devices']}"
            )

            if os.path.exists(log_path):
                os.unlink(log_path)


@pytest.mark.cpu
class TestCPUExecution:
    """CPU-only execution tests."""

    def test_cpu_cli(self, executable, model, timeouts):
        """Test CLI mode with CPU only: verify no GPU and response works."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu="disable",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            log_path = f.name

        try:
            result = runner.run_cli(
                "Say hello",
                timeout=timeouts.cli,
                log_file=log_path,
                extra_args=["--verbose"],
            )

            # Verify response works
            assert result.returncode == 0
            assert len(result.stdout.strip()) > 0

            # Verify no GPU is used
            gpu_info = check_gpu_in_output(result.log_output)
            assert not gpu_info["gpu_used"], (
                f"GPU used when disabled. Devices: {gpu_info['devices']}"
            )
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_cpu_server(self, executable, model, server_port, timeouts):
        """Test server mode with CPU only: verify no GPU and response works."""
        runner = LlamafileRunner(
            executable=executable,
            model=model,
            gpu="disable",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            log_path = f.name

        proc = runner.start_server(port=server_port, log_file=log_path,
                                    extra_args=["--verbose"])

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            # Verify response works
            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello"}],
                timeout=timeouts.http_request,
            )
            assert len(response["choices"][0]["message"]["content"]) > 0

        finally:
            proc.terminate()
            proc.wait()

            # Verify no GPU is used
            log_output = LlamafileRunner.read_log_file(log_path)
            gpu_info = check_gpu_in_output(log_output)
            assert not gpu_info["gpu_used"], (
                f"GPU used when disabled. Devices: {gpu_info['devices']}"
            )

            if os.path.exists(log_path):
                os.unlink(log_path)
