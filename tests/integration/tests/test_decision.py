"""Decision models: POST /v1/systemone, --cli JSON mode, and the chat gates.

These tests run when --model is a decision model, detected from its GGUF
metadata as llama.cpp does (see utils/gguf.py); with any other model they
skip, and with a decision-only model the rest of the suite skips. Small
models to try: ggml-org/tinylaya-for-testing-gguf (laya, ~100 MB) and
ggml-org/tinyopenjev-for-testing-gguf (openjev, with an mmproj for the
image test):

    uv run pytest tests/test_decision.py --executable ... \
        --model tinyopenjev-for-testing-Q8_0.gguf \
        --mmproj mmproj-tinyopenjev-for-testing-Q8_0.gguf
"""

import json
import subprocess

import pytest
import requests

from utils.llamafile import LlamafileRunner, read_until_idle, stop_tui, strip_ansi

pytestmark = pytest.mark.decision

STATE = "Customer message: I was charged twice for my order last week and nobody has replied."
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"],
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}
REQUEST = {"state": STATE, "questions": QUESTIONS}
NO_CHAT = "This model is a decision model without text generation, use /v1/systemone"


def systemone(port, body, timeout):
    return requests.post(f"http://127.0.0.1:{port}/v1/systemone", json=body, timeout=timeout)


def run_cli(runner, extra_args=None, stdin=None, timeout=120):
    """--cli without the -p/--nothink that LlamafileRunner.run_cli adds"""
    args = runner._base_args() + ["--cli"] + (extra_args or [])
    return subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=None if stdin is not None else subprocess.DEVNULL,
    )


def check_answers(body):
    """the shape of the answers to QUESTIONS"""
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0
    answers = body["answers"]
    assert list(answers) == ["route", "urgency", "angry"]
    route = answers["route"]
    assert list(route["probabilities"]) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    urgency = answers["urgency"]
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert 0.0 <= answers["angry"]["noul"] <= 1.0


def mmproj_args(mmproj):
    return ["--mmproj", mmproj] if mmproj else []


@pytest.fixture
def decision_only(model_kind):
    """for the chat gates: a model that also chats (Granite Switch) passes them"""
    if model_kind.text:
        pytest.skip(f"{model_kind.decision_type} also generates text")


@pytest.fixture
def server(llamafile, server_port, timeouts, mmproj):
    proc = llamafile.start_server(port=server_port, extra_args=mmproj_args(mmproj))
    try:
        assert LlamafileRunner.wait_for_server(
            server_port, timeout=timeouts.server_ready, proc=proc
        ), "server did not become ready"
        yield server_port
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.mark.server
class TestDecisionServer:
    def test_systemone_answers(self, server, timeouts):
        res = systemone(server, REQUEST, timeouts.http_request)
        assert res.status_code == 200, res.text
        check_answers(res.json())

    def test_models_lists_decisions(self, server):
        models = requests.get(f"http://127.0.0.1:{server}/v1/models", timeout=10).json()
        assert "decisions" in models["data"][0]["architecture"]["output_modalities"]

    def test_invalid_request(self, server, timeouts):
        res = systemone(server, {"state": STATE}, timeouts.http_request)
        assert res.status_code == 400
        assert "error" in res.json()

    def test_chat_completion_refused(self, decision_only, server, timeouts):
        res = requests.post(
            f"http://127.0.0.1:{server}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            timeout=timeouts.http_request,
        )
        assert res.status_code == 501
        assert res.json()["error"]["message"] == NO_CHAT


@pytest.mark.cli
class TestDecisionCli:
    def test_stdin_matches_server(self, llamafile, server, mmproj, timeouts):
        """the CLI gives the same answers as a fresh server"""
        expected = systemone(server, REQUEST, timeouts.http_request).json()
        result = run_cli(llamafile, mmproj_args(mmproj), stdin=json.dumps(REQUEST), timeout=timeouts.cli)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected

    def test_prompt_flag(self, llamafile, timeouts):
        result = run_cli(llamafile, ["-p", json.dumps(REQUEST)], timeout=timeouts.cli)
        assert result.returncode == 0, result.stderr
        check_answers(json.loads(result.stdout))

    def test_escapes_in_request(self, llamafile, tmp_path, timeouts):
        """JSON escapes survive -p and -f, which llama.cpp's escape processing
        would otherwise rewrite: same answer as on stdin"""
        request = json.dumps({
            "state": 'He wrote "refund me"\non the form, saved as C:\\Users\\a.txt',
            "questions": {"angry": QUESTIONS["angry"]},
        })
        assert '\\"' in request and "\\n" in request and "\\\\" in request
        path = tmp_path / "request.json"
        path.write_text(request)
        answers = []
        for args, stdin in ([], request), (["-p", request], None), (["-f", str(path)], None):
            result = run_cli(llamafile, args, stdin=stdin, timeout=timeouts.cli)
            assert result.returncode == 0, f"{args[:1] or 'stdin'}: {result.stderr}"
            answers.append(json.loads(result.stdout))
        assert answers[0] == answers[1] == answers[2]

    def test_invalid_request(self, llamafile, timeouts):
        result = run_cli(llamafile, stdin=json.dumps({"state": STATE}), timeout=timeouts.cli)
        assert result.returncode == 1
        assert result.stdout == ""
        assert "error:" in result.stderr

    def test_text_prompt_gets_a_hint(self, decision_only, llamafile, timeouts):
        result = run_cli(llamafile, ["-p", "is the customer angry?"], timeout=timeouts.cli)
        assert result.returncode != 0
        assert "/v1/systemone" in result.stderr

    @pytest.mark.multimodal
    def test_image_flag(self, llamafile, mmproj, test_image, timeouts):
        if not mmproj:
            pytest.skip("No projector. Use --mmproj or LLAMAFILE_MMPROJ")
        result = run_cli(
            llamafile,
            mmproj_args(mmproj) + ["--image", str(test_image)],
            stdin=json.dumps(REQUEST),
            timeout=timeouts.cli,
        )
        assert result.returncode == 0, result.stderr
        with_image = json.loads(result.stdout)
        check_answers(with_image)
        without = run_cli(llamafile, stdin=json.dumps(REQUEST), timeout=timeouts.cli)
        assert with_image["usage"]["input_tokens"] > json.loads(without.stdout)["usage"]["input_tokens"]


class TestDecisionChatGates:
    @pytest.mark.tui
    def test_chat_refuses_decision_model(self, decision_only, llamafile, timeouts):
        result = subprocess.run(
            llamafile._base_args() + ["--chat", "--nologo"],
            input="hi\n",
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeouts.tui,
        )
        assert result.returncode == 1
        assert NO_CHAT in result.stderr

    @pytest.mark.combined
    def test_combined_mode_serves_decisions(self, decision_only, llamafile, server_port, timeouts):
        """the TUI says the model can't chat and shows the server's error,
        and the server answers /v1/systemone"""
        proc = llamafile.start_combined(port=server_port)
        try:
            assert LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready, proc=proc
            ), "server did not become ready"
            startup = strip_ansi(read_until_idle(proc.stdout, idle_timeout=1.0, max_timeout=10.0))
            assert "decision model" in startup and "/v1/systemone" in startup

            res = systemone(server_port, REQUEST, timeouts.http_request)
            assert res.status_code == 200, res.text
            check_answers(res.json())

            # the TUI prints errors on stderr
            _ = read_until_idle(proc.stderr, idle_timeout=0.5, max_timeout=5.0)
            proc.stdin.write("hello\n")
            proc.stdin.flush()
            errors = strip_ansi(read_until_idle(proc.stderr, idle_timeout=2.0, max_timeout=timeouts.cli))
            assert f"error: {NO_CHAT}" in errors
        finally:
            stop_tui(proc)
