"""Tool calling integration tests."""

import json

import pytest

from utils.llamafile import LlamafileRunner


# Example tool definition for testing
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
}


@pytest.mark.tool_calling
@pytest.mark.server
class TestToolCalling:
    """Tool calling tests using server mode."""

    def test_tool_call_basic(self, llamafile, server_port, timeouts):
        """Test that model can make a tool call."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[
                    {"role": "user", "content": "What is 15 * 23? Use the calculator."}
                ],
                tools=[CALCULATOR_TOOL],
                tool_choice="auto",
                timeout=timeouts.http_request,
            )

            # Check if model made a tool call
            message = response["choices"][0]["message"]

            # Model should either call the tool or give a direct answer
            has_tool_call = "tool_calls" in message and len(message["tool_calls"]) > 0
            has_content = message.get("content") and "345" in message["content"]

            assert has_tool_call or has_content, (
                f"Expected tool call or correct answer. Got: {message}"
            )

        finally:
            proc.terminate()
            proc.wait()

    def test_tool_call_correct_function(self, llamafile, server_port, timeouts):
        """Test that model calls the correct tool."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[
                    {"role": "user", "content": "What's the weather in Tokyo?"}
                ],
                tools=[CALCULATOR_TOOL, WEATHER_TOOL],
                tool_choice="auto",
                timeout=timeouts.http_request,
            )

            message = response["choices"][0]["message"]

            if "tool_calls" in message and len(message["tool_calls"]) > 0:
                tool_call = message["tool_calls"][0]
                # Should call weather, not calculator
                assert tool_call["function"]["name"] == "get_weather"

        finally:
            proc.terminate()
            proc.wait()

    def test_tool_call_with_arguments(self, llamafile, server_port, timeouts):
        """Test that tool calls include correct arguments."""
        proc = llamafile.start_server(port=server_port)

        try:
            ready = LlamafileRunner.wait_for_server(
                server_port, timeout=timeouts.server_ready
            )
            assert ready

            response = LlamafileRunner.chat_completion(
                port=server_port,
                messages=[
                    {"role": "user", "content": "Calculate 100 divided by 4"}
                ],
                tools=[CALCULATOR_TOOL],
                tool_choice="required",
                timeout=timeouts.http_request,
            )

            message = response["choices"][0]["message"]

            assert "tool_calls" in message
            tool_call = message["tool_calls"][0]
            assert tool_call["function"]["name"] == "calculate"

            # Arguments should contain the expression
            args = json.loads(tool_call["function"]["arguments"])
            assert "expression" in args

        finally:
            proc.terminate()
            proc.wait()
