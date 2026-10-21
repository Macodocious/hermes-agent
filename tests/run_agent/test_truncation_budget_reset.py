"""The continuation budget resets on tool success.

Pins the contract from session 20261019_174729_96ecec6f: four truncated replies,
interleaved with successful tool calls, ended a turn with
"Response remained truncated after 4 continuation attempts".
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


def _cut_response():
    message = SimpleNamespace(
        content="", tool_calls=None, reasoning="The final response. " * 40
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")],
        model="test/model",
        usage=None,
        id="resp-cut",
    )
    return response


def _tool_call_response():
    call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="todo", arguments="{}"),
    )
    message = SimpleNamespace(content="", tool_calls=[call], reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
        id="resp-tool",
    )


def _done_response():
    message = SimpleNamespace(content="done", tool_calls=None, reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
        id="resp-done",
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            session_id="truncation-budget-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance.valid_tool_names = {"todo"}
    return instance


def test_successful_tool_call_resets_the_continuation_budget(agent):
    responses = (
        [_cut_response() for _ in range(5)]
        + [_tool_call_response()]
        + [_cut_response() for _ in range(3)]
        + [_done_response()]
    )
    agent._interruptible_api_call = lambda _kwargs: responses.pop(0)

    def _execute(_assistant_message, messages, _task_id, _api_call_count):
        messages.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})

    agent._execute_tool_calls = _execute

    result = agent.run_conversation("finish the report")

    assert result["final_response"] == "done"
    assert result.get("error") is None
