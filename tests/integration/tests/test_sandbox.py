"""Sandbox (pledge/SECCOMP) integration tests.

llamafile sandboxes itself with Cosmopolitan's pledge(), which installs a
SECCOMP BPF filter on Linux (issue #930). These tests verify externally
that the sandbox is really in place: a pledged process carries one more
SECCOMP filter than its parent (/proc/<pid>/status, ``Seccomp_filters``).
The comparison is relative to the pytest process itself because container
runtimes (e.g. Docker's default profile) already wrap everything in a
filter of their own. Inference must keep working while sandboxed;
``--unsecure``, combined mode, and GPU mode must skip the sandbox and say
so in the log.

Enforcement is only possible on Linux and OpenBSD; on other hosts the
enforcement tests skip and we only check that llamafile reports the
sandbox status honestly. (tests/sandbox_test.c covers the syscall-level
allow/deny behavior of each promise set.)
"""

import os
import platform
import subprocess

import pytest

from utils.llamafile import LlamafileRunner

IS_LINUX = platform.system() == "Linux"

requires_linux = pytest.mark.skipif(
    not IS_LINUX, reason="SECCOMP enforcement is only observable on Linux"
)


def _status_field(pid: int, field: str) -> int | None:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return None


def _seccomp_filters(pid: int) -> int:
    """Number of SECCOMP filters attached to <pid>.

    Prefers the exact ``Seccomp_filters`` count (Linux 5.9+). On older
    kernels falls back to the ``Seccomp`` mode field, mapping FILTER mode
    to 1 and disabled to 0 — good enough for a relative comparison as
    long as the environment itself doesn't sandbox processes.
    """
    count = _status_field(pid, "Seccomp_filters")
    if count is not None:
        return count
    return 1 if _status_field(pid, "Seccomp") == 2 else 0


def _baseline_filters() -> int:
    """SECCOMP filters every spawned process inherits from this test
    runner (0 on bare metal, typically 1 under Docker's default
    profile)."""
    return _seccomp_filters(os.getpid())


def _resolve_server_pid(pid: int) -> int:
    """Follow the sh -> APE-loader chain down to the llamafile process.

    The runner launches ``sh <executable>``. The APE shell header normally
    exec()s in place so the PID stays the same, but if any stage forked we
    walk single-child links downward.
    """
    for _ in range(5):
        children = []
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    children.extend(int(p) for p in f.read().split())
        except OSError:
            break
        if not children:
            break
        pid = children[0]
    return pid


@pytest.fixture
def cpu_runner(executable, model) -> LlamafileRunner:
    """Runner pinned to CPU: a loaded GPU backend disables the sandbox
    (drivers need syscalls pledge forbids), so tests force --gpu disable
    to make the sandbox state deterministic on any host."""
    return LlamafileRunner(executable=executable, model=model, gpu="disable")


@pytest.mark.sandbox
@pytest.mark.server
@pytest.mark.cpu
class TestServerSandbox:
    @requires_linux
    def test_server_sandboxed_by_default(self, cpu_runner, server_port, timeouts, tmp_path):
        """--server gets a SECCOMP filter and still serves completions."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(port=server_port, log_file=log_file)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"

            pid = _resolve_server_pid(proc.pid)
            assert _seccomp_filters(pid) == _baseline_filters() + 1, (
                "expected exactly one additional SECCOMP filter on the "
                "server process; the pledge() sandbox is not installed"
            )

            # inference must keep working inside the sandbox
            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": "Say hello in one word."}],
                timeout=timeouts.http_request,
            )
            assert response["choices"][0]["message"]["content"].strip()
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert 'sandbox: pledge("' in log, "server should log the active promise set"

    @requires_linux
    def test_unsecure_flag_disables_sandbox(self, cpu_runner, server_port, timeouts, tmp_path):
        """--unsecure opts out: no SECCOMP filter, and the log says why."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(
            port=server_port, log_file=log_file, extra_args=["--unsecure"]
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"
            pid = _resolve_server_pid(proc.pid)
            assert _seccomp_filters(pid) == _baseline_filters(), (
                "--unsecure must not install a filter"
            )
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert "disabled by --unsecure" in log

    def test_sandbox_status_always_logged(self, cpu_runner, server_port, timeouts, tmp_path):
        """On every platform the server reports its sandbox state honestly
        (active on Linux/OpenBSD, an explicit skip reason elsewhere)."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(port=server_port, log_file=log_file)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert "sandbox:" in log, "server must log its sandbox status"
        if not IS_LINUX:
            assert "not supported on this OS" in log


@pytest.mark.sandbox
@pytest.mark.cli
@pytest.mark.cpu
class TestCliSandbox:
    @requires_linux
    def test_cli_sandboxed_and_functional(self, cpu_runner, timeouts):
        """--cli pledges "stdio rpath tty" after arg parsing and still
        generates a response. --verbose surfaces the status line on
        stderr (CLI keeps stdout clean for the model output)."""
        result = cpu_runner.run_cli(
            "Say hello.", extra_args=["--verbose"], timeout=timeouts.cli
        )
        assert result.returncode == 0
        assert "sandbox: active" in result.stderr
        assert result.stdout.strip()


def _stop_hard(proc):
    """Terminate a process without ever hanging the test: SIGTERM first
    (llamafile shuts down gracefully), SIGKILL if it doesn't oblige."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.mark.sandbox
@pytest.mark.combined
@pytest.mark.cpu
class TestCombinedModeSandbox:
    @requires_linux
    def test_combined_mode_skips_sandbox(self, cpu_runner, server_port, timeouts, tmp_path):
        """Combined mode hosts a TUI HTTP client in-process, which needs
        connect(); the accept-only sandbox is skipped and logged.

        The TUI (bestline) exits — taking the whole process down — unless
        stdin AND stdout are terminals, so the process runs on a pty.
        --verbose is required too: non-server modes turn the common log
        down to errors-only, which would swallow the skip notice.
        """
        import pty  # POSIX-only; fine under @requires_linux

        log_file = str(tmp_path / "combined.log")
        args = cpu_runner._base_args() + [
            "--port", str(server_port), "--verbose", "--log-file", log_file,
        ]
        master, slave = pty.openpty()
        proc = subprocess.Popen(
            args, stdin=slave, stdout=slave, stderr=subprocess.DEVNULL
        )
        os.close(slave)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"
            pid = _resolve_server_pid(proc.pid)
            assert _seccomp_filters(pid) == _baseline_filters()
        finally:
            _stop_hard(proc)
            os.close(master)

        log = LlamafileRunner.read_log_file(log_file)
        assert "disabled in combined mode" in log
