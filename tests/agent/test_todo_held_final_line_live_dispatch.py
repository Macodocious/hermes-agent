"""Live dispatch: the final completion line waits for the turn's end.

A real ``AIAgent`` is driven through ``execute_tool_calls_sequential`` with a
delivery callback attached. The final close's line is held while the turn
continues; the turn-end release ships it, on the normal and interrupted paths.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent import task_manager


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("todo", "switch_context")
]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=TOOL_DEFINITIONS),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.tasks.persist_todo_store", lambda agent: None
    )
    monkeypatch.setattr("agent.task_manager._persist", lambda agent: None)
    monkeypatch.setattr(
        "agent.task_manager._lifecycle_config", lambda: {"enabled": True}
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "agent.tool_executor.maybe_persist_tool_result",
        lambda **kwargs: kwargs["content"],
    )
    agent = _make_agent()
    agent._todo_store.write([
        {"id": "1", "content": "one", "status": "pending"},
        {"id": "2", "content": "two", "status": "pending"},
    ])
    agent._todo_store.drain_notices()
    return agent


def _send(agent, request_id: str, *calls) -> list[str]:
    from agent.tool_executor import execute_tool_calls_sequential

    agent._current_api_request_id = request_id
    message = SimpleNamespace(content="", tool_calls=[
        _call(name, args, f"{request_id}:{index}")
        for index, (name, args) in enumerate(calls, start=1)
    ])
    messages: list = []
    execute_tool_calls_sequential(agent, message, messages, "")
    return [m["content"] for m in messages]


def _work():
    return ("switch_context", {"context": "coding"})


def _complete(item_id: str):
    return ("todo", {"action": "complete", "item_id": item_id})


def _run_to_the_final_close(agent, sent):
    agent.todo_notice_callback = sent.append
    _send(agent, "m1", ("todo", {"action": "begin", "item_id": "1"}))
    _send(agent, "m2", _work())
    _send(agent, "m3", _complete("1"))
    _send(agent, "m4", _work())
    _send(agent, "m5", _complete("2"))
    _send(agent, "m6", _work())  # the turn continues after the final close


def test_the_final_line_waits_for_the_turn_end(agent) -> None:
    sent = []
    _run_to_the_final_close(agent, sent)

    assert sent == [
        "▶️ Working on: one",
        "✅ Task completed: one",
        "▶️ Working on: two",
    ]

    task_manager.release_held_notices(agent)

    assert sent[-1] == "✅ Task completed: two"


def test_an_interrupted_turn_still_ships_the_held_line(agent) -> None:
    sent = []
    _run_to_the_final_close(agent, sent)
    agent._interrupt_requested = True

    task_manager.release_held_notices(agent)

    assert sent[-1] == "✅ Task completed: two"
