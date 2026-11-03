"""Built-in server tools that start a child process (POST /tools).

exec_shell_command and get_info are the two built-in tools that spawn a
process. On Windows they used to fail every time ("failed to spawn process",
issue #1066): the cosmopolitan build never defines _WIN32, so it took the POSIX
branch and asked for `sh` / `uname`, which a stock Windows install does not
have. Once that was fixed, exit codes from native Windows programs were also
misreported (`exit 3` came back as 1, `exit 128` as 0). The requests below are
written to pass on every host OS: `echo` and `exit` exist in both `sh -c` and
`cmd.exe /c`.

The server runs with --unsecure because the default CPU-server sandbox on
Linux and OpenBSD blocks process creation (see docs/built-in-tools.md).
"""

import pytest
import requests

from utils.llamafile import LlamafileRunner

MARKER = "llamafile-tools-ok"


def _call_tool(port: int, tool: str, params: dict, timeout: float) -> dict:
    response = requests.post(
        f"http://127.0.0.1:{port}/tools",
        json={"tool": tool, "params": params},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.server
@pytest.mark.cpu
class TestBuiltinProcessTools:
    def test_exec_shell_command_and_get_info_spawn(self, executable, model, server_port, timeouts):
        runner = LlamafileRunner(executable=executable, model=model, gpu="disable", no_warmup=True)
        proc = runner.start_server(
            port=server_port,
            extra_args=["--unsecure", "--tools", "exec_shell_command,get_info"],
        )
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "Server did not become ready"

            shell = _call_tool(
                server_port, "exec_shell_command", {"command": f"echo {MARKER}"},
                timeouts.http_request,
            )
            text = shell.get("plain_text_response", "")
            assert "failed to spawn process" not in text, shell
            assert MARKER in text, shell
            assert "[exit code: 0]" in text, shell

            # a failing command must report its own exit code, not 0 or 1
            failed = _call_tool(
                server_port, "exec_shell_command", {"command": "exit 3"},
                timeouts.http_request,
            )
            assert "[exit code: 3]" in failed.get("plain_text_response", ""), failed

            info = _call_tool(server_port, "get_info", {}, timeouts.http_request)
            assert info.get("os") not in (None, "", "unknown"), info
        finally:
            proc.terminate()
            proc.wait()
