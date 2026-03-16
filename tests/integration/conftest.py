"""Pytest fixtures for llamafile integration tests."""

import os
from pathlib import Path

import pytest

from utils.llamafile import (
    LlamafileRunner,
    TIMEOUT_CLI,
    TIMEOUT_TUI,
    TIMEOUT_SERVER_READY,
    TIMEOUT_HTTP_REQUEST,
    POLL_INTERVAL,
)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--executable",
        action="store",
        default=None,
        help="Path to llamafile executable or pre-built .llamafile",
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path to model file (not needed for pre-built llamafiles)",
    )
    parser.addoption(
        "--mmproj",
        action="store",
        default=None,
        help="Path to multimodal projector model file (for vision tests)",
    )
    parser.addoption(
        "--gpu",
        action="store",
        default=None,
        choices=["auto", "apple", "amd", "nvidia", "disable"],
        help="GPU mode to use (disable for CPU-only)",
    )
    parser.addoption(
        "--timeout-multiplier",
        action="store",
        default=1.0,
        type=float,
        help="Multiplier for all timeouts (e.g., 2.0 for slower models)",
    )


@pytest.fixture(scope="session")
def executable(request) -> str:
    """Get the llamafile executable path.

    Priority: --executable flag > LLAMAFILE_EXECUTABLE env var
    """
    exe = request.config.getoption("--executable")
    if exe:
        return exe

    exe = os.environ.get("LLAMAFILE_EXECUTABLE")
    if exe:
        return exe

    pytest.skip("No executable specified. Use --executable or LLAMAFILE_EXECUTABLE")


@pytest.fixture(scope="session")
def model(request) -> str | None:
    """Get the model path (optional for pre-built llamafiles).

    Priority: --model flag > LLAMAFILE_MODEL env var
    """
    model_path = request.config.getoption("--model")
    if model_path:
        return model_path

    return os.environ.get("LLAMAFILE_MODEL")


@pytest.fixture(scope="session")
def gpu_mode(request) -> str | None:
    """Get the GPU mode.

    Use --gpu disable for CPU-only execution.
    """
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def mmproj(request) -> str | None:
    """Get the multimodal projector model path.

    Priority: --mmproj flag > LLAMAFILE_MMPROJ env var
    """
    mmproj_path = request.config.getoption("--mmproj")
    if mmproj_path:
        return mmproj_path

    return os.environ.get("LLAMAFILE_MMPROJ")


@pytest.fixture(scope="session")
def timeout_multiplier(request) -> float:
    """Get the timeout multiplier for slower models."""
    return float(request.config.getoption("--timeout-multiplier"))


class Timeouts:
    """Scaled timeout values."""

    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        self.cli = TIMEOUT_CLI * multiplier
        self.tui = TIMEOUT_TUI * multiplier
        self.server_ready = TIMEOUT_SERVER_READY * multiplier
        self.http_request = TIMEOUT_HTTP_REQUEST * multiplier
        self.poll_interval = POLL_INTERVAL * multiplier


@pytest.fixture(scope="session")
def timeouts(timeout_multiplier) -> Timeouts:
    """Get scaled timeout values."""
    return Timeouts(timeout_multiplier)


@pytest.fixture(scope="session")
def llamafile(executable, model, gpu_mode) -> LlamafileRunner:
    """Create a LlamafileRunner instance for tests."""
    return LlamafileRunner(
        executable=executable,
        model=model,
        gpu=gpu_mode,
    )


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def test_image(fixtures_dir) -> Path:
    """Get the test image path."""
    image_path = fixtures_dir / "test_image.png"
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return image_path


@pytest.fixture
def server_port() -> int:
    """Get an available port for server tests.

    Uses PORT env var or defaults to 8080.
    """
    return int(os.environ.get("PORT", "8080"))
