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

pledge() is on by default; unveil() path confinement is opt-in via
``--confine-reads`` (the default must NOT confine, so files opened by path at
request time keep working). The confinement tests below check that the default
does not confine and that ``--confine-reads`` does; the syscall-level proof
that confinement denies reads outside the weights dirs is in sandbox_test.c.

Enforcement is only observable on Linux; on other hosts the enforcement
tests skip and we only check that llamafile reports its status honestly.
tests/sandbox_test.c covers the syscall-level allow/deny behavior and the
unveil() path confinement directly.
"""

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from utils.llamafile import LlamafileRunner
from utils.prompts import GREETING_PROMPT

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


def _seccomp_mode(pid: int) -> int | None:
    """The coarse ``Seccomp`` mode field: 2 (SECCOMP_MODE_FILTER) means at
    least one filter is installed, 0 means none. Available on every kernel
    with SECCOMP. Can't distinguish our filter from a container's, so it's
    only a positive 'a filter exists' check, used as a fallback when the
    exact count is unavailable so a fully-unsandboxed build still fails."""
    return _status_field(f"/proc/{pid}/status", "Seccomp")


def _assert_sandbox_installed(pid: int):
    """A sandboxed process gains at least one filter beyond the runner's
    baseline. Prefer the exact count; fall back to the mode field on old
    kernels so a totally unsandboxed regression can't skip through."""
    filters, base = _seccomp_filters(pid), _baseline_filters()
    if filters is not None and base is not None:
        assert filters > base, (
            "expected the process to gain a SECCOMP filter; the pledge() "
            "sandbox is not installed"
        )
    else:
        assert _seccomp_mode(pid) == 2, (
            "expected SECCOMP_MODE_FILTER; the pledge() sandbox is not installed"
        )


def _assert_no_added_filter(pid: int):
    """The process carries no filter beyond the runner's baseline (the
    sandbox was skipped). Needs the exact count to tell our filter from a
    container's; skip the check where it's unavailable."""
    filters, base = _seccomp_filters(pid), _baseline_filters()
    if filters is None or base is None:
        pytest.skip("kernel lacks Seccomp_filters; cannot count filters")
    assert filters == base, "expected no added SECCOMP filter"


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
    """Runner pinned to CPU, with --no-warmup.

    --gpu disable: a loaded GPU backend disables the sandbox (drivers need
    syscalls pledge forbids), so CPU is forced to make sandbox state
    deterministic on any host.

    --no-warmup: large models (e.g. 26B MXFP4 MoE) can take longer than the
    120s server-ready timeout when the kernel page cache is cold, because
    pledge may restrict the prefetch hints (posix_fadvise / MADV_WILLNEED)
    that speed up the initial memory-mapped read.  Sandbox tests care about
    whether the pledge filter is installed and whether inference works, not
    about the warmup pass itself.
    """
    return LlamafileRunner(executable=executable, model=model, gpu="disable", no_warmup=True)


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
            # The sandbox adds at least the pledge filter (and, when unveil
            # confinement applies, cosmo's extra truncate/setxattr filter);
            # the exact count is cosmo's business, so assert presence not a
            # specific delta. The negative cases below pin the baseline.
            _assert_sandbox_installed(pid)

            # Every thread must carry the same filter count, not just the
            # main thread: a pre-existing background thread (e.g. the log
            # worker) would otherwise escape the per-thread SECCOMP filter
            # (issue #930). Needs the exact per-thread count (5.9+); where
            # unavailable the presence check above still holds.
            counts = _thread_filter_counts(pid)
            if counts:
                base = _baseline_filters()
                assert set(counts.values()) == {max(counts.values())}, (
                    f"threads have mismatched SECCOMP filter counts {counts}; "
                    f"a thread escaped the sandbox"
                )
                if base is not None:
                    assert max(counts.values()) > base, "no thread is sandboxed"

            # inference must keep working inside the sandbox
            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": GREETING_PROMPT}],
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
            _assert_no_added_filter(_resolve_server_pid(proc.pid))
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
@pytest.mark.server
@pytest.mark.cpu
class TestConfineReads:
    """The pledge()/unveil() split: pledge() is on by default, unveil() path
    confinement is opt-in via --confine-reads."""

    @requires_linux
    def test_default_does_not_confine(self, cpu_runner, server_port, timeouts, tmp_path):
        """By default the server is pledged but NOT path-confined, so files
        opened by path at request time (multimodal media, etc.) keep working.
        The log must say active without claiming confinement."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(port=server_port, log_file=log_file)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"
        finally:
            _stop_hard(proc)
        log = LlamafileRunner.read_log_file(log_file)
        assert 'pledge("stdio anet rpath")' in log, "default should pledge, no confinement"
        assert "confined to weights dirs" not in log, "default must not confine reads"

    @requires_linux
    def test_confine_reads_confines_and_serves(self, cpu_runner, server_port, timeouts, tmp_path):
        """--confine-reads adds unveil() confinement and still serves."""
        log_file = str(tmp_path / "server.log")
        proc = cpu_runner.start_server(
            port=server_port, log_file=log_file, extra_args=["--confine-reads"]
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"
            _assert_sandbox_installed(_resolve_server_pid(proc.pid))
            resp = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[{"role": "user", "content": GREETING_PROMPT}],
                timeout=timeouts.http_request,
            )
            assert resp["choices"][0]["message"]["content"].strip()
        finally:
            _stop_hard(proc)
        log = LlamafileRunner.read_log_file(log_file)
        # This log line is emitted only after the governability probe confirms
        # a canary outside the rules is actually denied (see sandbox.c), so it
        # means confinement is installed and biting -- not merely requested.
        assert "reads confined to weights dirs" in log

    # Note: there is no integration test that drives a *live* server into
    # reading a file outside the weights dirs, because the HTTP API exposes no
    # arbitrary-read primitive to exploit -- static serving (httplib) and media
    # loading (fs_validate_filename) both reject traversal/symlinks before a
    # read is attempted, so unveil() never gets a chance to be the thing that
    # denies. unveil() is the defense-in-depth backstop *behind* those checks;
    # its enforcement (allowed inside, EACCES outside) is proven directly at
    # the syscall level in tests/sandbox_test.c (child_unveil).


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
            GREETING_PROMPT, extra_args=["--verbose"], timeout=timeouts.cli
        )
        assert result.returncode == 0
        assert "sandbox: active" in result.stderr
        assert result.stdout.strip()


@pytest.mark.sandbox
@pytest.mark.tui
@pytest.mark.cpu
class TestChatSandbox:
    @requires_linux
    def test_chat_all_threads_sandboxed(self, cpu_runner, timeouts, tmp_path):
        """Chat mode wraps the pledge in log pause/resume like the server,
        so no pre-existing thread (the log worker) escapes the per-thread
        filter. Launch headless with stdin held open so the REPL blocks
        after sandboxing, then confirm every thread carries the filter
        (issue #930 — the chat/cli path the round-1 tests missed)."""
        # The "sandbox:" status is written to stderr (not the common-log
        # --log-file), so capture stderr to detect when the pledge is in.
        err_file = tmp_path / "chat.err"
        args = cpu_runner._base_args() + ["--chat", "--verbose", "--nologo"]
        with open(err_file, "wb") as err:
            proc = subprocess.Popen(
                args, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=err,
            )
            try:
                deadline = time.time() + timeouts.server_ready
                ready = False
                while time.time() < deadline:
                    if proc.poll() is not None:
                        pytest.fail("chat exited before reaching the sandbox step")
                    if "sandbox:" in err_file.read_text(errors="replace"):
                        ready = True
                        break
                    time.sleep(0.2)
                assert ready, "chat did not reach the sandbox step"

                pid = _resolve_server_pid(proc.pid)
                _assert_sandbox_installed(pid)
                counts = _thread_filter_counts(pid)
                if counts:
                    assert set(counts.values()) == {max(counts.values())}, (
                        f"chat threads have mismatched SECCOMP filter counts "
                        f"{counts}; a thread (e.g. the log worker) escaped"
                    )
            finally:
                _stop_hard(proc)

        assert "sandbox: active" in err_file.read_text(errors="replace")


@pytest.mark.sandbox
@pytest.mark.combined
@pytest.mark.cpu
class TestCombinedModeSandbox:
    @requires_linux
    def test_combined_mode_skips_sandbox(self, cpu_runner, server_port, timeouts, tmp_path):
        """Combined mode hosts a TUI HTTP client in-process, which needs
        connect(); the accept-only sandbox is skipped and logged.

        --verbose is required: without it main.cpp injects --log-verbosity 1
        (ERROR-only threshold), suppressing the WARN-level skip notice even
        in the log file.  The log file (not a pipe) is used for output so
        --verbose does not cause a pipe-buffer overflow.
        """
        log_file = str(tmp_path / "combined.log")
        proc = cpu_runner.start_combined(
            port=server_port, log_file=log_file, extra_args=["--verbose"],
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Combined-mode server did not become ready"
            _assert_no_added_filter(_resolve_server_pid(proc.pid))
        finally:
            _stop_hard(proc)

        log = LlamafileRunner.read_log_file(log_file)
        assert "disabled in combined mode" in log
