"""Live dispatch: the start line waits for the first real call.

A real ``AIAgent`` is driven through ``execute_tool_calls_sequential``. The
head-row list write makes the row current; the first non-todo call records the
start line; a row that never sees a call gets none.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import MagicMock, patch

import pytest


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
    """Dispatch one assistant message holding the given tool calls."""
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


def test_head_row_write_records_no_start_until_the_first_call(agent) -> None:
    sent = []
    agent.todo_notice_callback = sent.append
    _send(agent, "m1", (
        "todo",
        {
            "todos": [{"id": "1", "content": "one", "status": "in_progress"}],
            "merge": True,
        },
    ))

    assert agent._todo_store.read()[0]["status"] == "in_progress"
    assert sent == []

    _send(agent, "m2", _work())

    assert sent == ["▶️ Working on: one"]
    assert agent._todo_store.pending_notices() == []


def test_a_row_that_never_sees_a_call_gets_no_line(agent) -> None:
    _send(agent, "m1", ("todo", {"action": "begin", "item_id": "1"}))
    assert agent._todo_store.pending_notices() == []

    _send(agent, "m2", ("todo", {"action": "complete", "item_id": "1"}))

    # The close is refused: no work under the row, so the row stays current
    # and no start line was ever recorded.
    assert agent._todo_store.read()[0]["status"] == "in_progress"
    assert agent._todo_store.pending_notices() == []
