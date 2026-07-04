"""Sandbox (pledge/SECCOMP/unveil) integration tests.

llamafile sandboxes itself with Cosmopolitan's pledge() and unveil() (issue
#930). On Linux pledge() installs a SECCOMP BPF filter and unveil() uses
Landlock. These tests verify externally that the sandbox is really in place:

  - a pledged process carries one more SECCOMP filter than its parent
    (/proc/<pid>/status ``Seccomp_filters``); the comparison is relative to
    the pytest process because container runtimes (e.g. Docker's default
    profile) already wrap everything in a filter of their own;
  - every thread of the server carries that filter, not just the main one
    (guards against a pre-existing thread escaping the per-thread filter);
  - inference keeps working while sandboxed, including for a bundled
    ``/zip/`` llamafile whose weights load through the executable;
  - ``--unsecure``, combined mode, and GPU mode skip the sandbox and say so.

Enforcement is only observable on Linux; on other hosts the enforcement
tests skip and we only check that llamafile reports its status honestly.
tests/sandbox_test.c covers the syscall-level allow/deny behavior and the
unveil() path confinement directly.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from utils.llamafile import LlamafileRunner

IS_LINUX = platform.system() == "Linux"

requires_linux = pytest.mark.skipif(
    not IS_LINUX, reason="SECCOMP enforcement is only observable on Linux"
)


def _status_field(pid_or_tid_path: str, field: str) -> int | None:
    try:
        with open(pid_or_tid_path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _seccomp_filters(pid: int) -> int | None:
    """Exact count of SECCOMP filters on <pid>, or None on kernels older
    than 5.9 that don't expose ``Seccomp_filters``. Callers skip the strict
    comparison when this is None rather than guessing from the coarse
    ``Seccomp`` mode field (which can't tell a container filter apart from
    our own)."""
    return _status_field(f"/proc/{pid}/status", "Seccomp_filters")


def _thread_filter_counts(pid: int) -> dict[int, int]:
    """Per-thread SECCOMP filter counts for every thread of <pid>."""
    counts = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        n = _status_field(f"/proc/{pid}/task/{tid}/status", "Seccomp_filters")
        if n is not None:
            counts[int(tid)] = n
    return counts


def _baseline_filters() -> int | None:
    """SECCOMP filters every spawned process inherits from this test runner
    (0 on bare metal, typically 1 under Docker's default profile)."""
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


def _stop_hard(proc):
    """Terminate a process without ever hanging the test: SIGTERM first
    (llamafile shuts down gracefully), SIGKILL if it doesn't oblige."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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
        """--server installs a SECCOMP filter on every thread and still
        serves completions inside the sandbox."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(port=server_port, log_file=log_file)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"

            pid = _resolve_server_pid(proc.pid)
            filters, base = _seccomp_filters(pid), _baseline_filters()
            if filters is None or base is None:
                pytest.skip("kernel lacks Seccomp_filters; cannot count filters")
            # The sandbox adds at least the pledge filter (and, when unveil
            # confinement applies, cosmo's extra truncate/setxattr filter);
            # the exact count is cosmo's business, so assert presence not a
            # specific delta. The negative cases below pin the baseline.
            assert filters > base, (
                "expected the server process to gain a SECCOMP filter; "
                "the pledge() sandbox is not installed"
            )

            # Every thread must carry the filter, not just the main thread:
            # a pre-existing background thread (e.g. the log worker) would
            # otherwise escape the per-thread SECCOMP filter (issue #930).
            counts = _thread_filter_counts(pid)
            assert counts, "no thread filter counts available"
            assert set(counts.values()) == {filters}, (
                f"threads have mismatched SECCOMP filter counts {counts}; "
                f"a thread escaped the sandbox"
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
            filters, base = _seccomp_filters(pid), _baseline_filters()
            if filters is None or base is None:
                pytest.skip("kernel lacks Seccomp_filters; cannot count filters")
            assert filters == base, "--unsecure must not install a filter"
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


def _find_zipalign(executable: str) -> str | None:
    """Locate the zipalign tool for building a bundled llamafile: env
    override, then the sibling build tree of a direct build, then PATH."""
    env = os.environ.get("LLAMAFILE_ZIPALIGN")
    if env and os.path.exists(env):
        return env
    for parent in Path(executable).resolve().parents:
        cand = parent / "third_party" / "zipalign" / "zipalign"
        if cand.exists():
            return str(cand)
    return shutil.which("zipalign")


@pytest.mark.sandbox
@pytest.mark.server
@pytest.mark.cpu
class TestBundledLlamafileSandbox:
    @requires_linux
    def test_embedded_zip_model_loads_under_sandbox(
        self, executable, model, server_port, timeouts, tmp_path
    ):
        """A bundled llamafile references its weights as ``/zip/model.gguf``;
        the loader reads them by reopening the executable's zip store, which
        needs the rpath promise. This is the exact configuration issue #930's
        rpath fix targets — a naive 'embedded => no filesystem' policy makes
        the model fail to load. Build a real bundle and assert it serves."""
        if not model:
            pytest.skip("needs --model to embed into a bundle")
        zipalign = _find_zipalign(executable)
        if not zipalign:
            pytest.skip("zipalign not found (set LLAMAFILE_ZIPALIGN)")

        bundle = tmp_path / "bundled.llamafile"
        shutil.copy(executable, bundle)
        os.chmod(bundle, 0o755)
        model_name = os.path.basename(model)
        args = tmp_path / ".args"
        args.write_text(f"-m\n/zip/{model_name}\n--server\n--gpu\ndisable\n")
        # zipalign is an APE binary; run it through sh like the llamafile
        # itself, so hosts without binfmt_misc for APE still work.
        za = ["sh", zipalign] if os.name != "nt" else [zipalign]
        subprocess.run(za + ["-j0", str(bundle), model, str(args)], check=True)

        log_file = str(tmp_path / "bundled.log")
        proc = subprocess.Popen(
            ["sh", str(bundle), "--port", str(server_port), "--log-file", log_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "bundled /zip/ server failed to load under the sandbox"
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert 'sandbox: pledge("' in log
        assert "rpath" in log, "embedded weights still need the rpath promise"


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


@pytest.mark.sandbox
@pytest.mark.combined
@pytest.mark.cpu
class TestCombinedModeSandbox:
    @requires_linux
    def test_combined_mode_skips_sandbox(self, cpu_runner, server_port, timeouts, tmp_path):
        """Combined mode hosts a TUI HTTP client in-process, which needs
        connect(); the accept-only sandbox is skipped and logged.

        --verbose is required: non-server modes turn the common log down to
        errors-only, which would swallow the skip notice.
        """
        log_file = str(tmp_path / "combined.log")
        proc = cpu_runner.start_combined(
            port=server_port, log_file=log_file, extra_args=["--verbose"]
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Combined-mode server did not become ready"
            pid = _resolve_server_pid(proc.pid)
            filters, base = _seccomp_filters(pid), _baseline_filters()
            if filters is None or base is None:
                pytest.skip("kernel lacks Seccomp_filters; cannot count filters")
            assert filters == base, "combined mode must not install a filter"
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert "disabled in combined mode" in log
