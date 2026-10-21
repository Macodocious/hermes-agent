"""Live dispatch: the two close checks through the real sequential executor.

A real ``AIAgent`` is driven through
``agent.tool_executor.execute_tool_calls_sequential`` with assistant messages
holding real ``tool_calls``. No store method is called directly: the checks run
where they run in production — inside the store, reached through the tool.

External seams (session persistence, lifecycle config, tool-result
persistence) are pinned by string-path monkeypatch, matching
``test_todo_sequential_dispatch``.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tools.todo_tool import NOTICE_COMPLETED, NOTICE_STARTED


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
    """A real non-todo tool call, made under the current row."""
    return ("switch_context", {"context": "coding"})


def _complete(item_id: str):
    return ("todo", {"action": "complete", "item_id": item_id})


def test_close_without_work_is_refused_through_the_chain(agent) -> None:
    _send(agent, "m1", ("todo", {"action": "begin", "item_id": "1"}), _complete("1"))

    assert agent._todo_store.read()[0]["status"] == "in_progress"


def test_two_closes_in_one_message_are_refused(agent) -> None:
    _send(agent, "m1", ("todo", {"action": "begin", "item_id": "1"}))
    _send(agent, "m2", _work())  # work under row 1

    results = _send(agent, "m3", _complete("1"), _work(), _complete("2"))

    assert "already closed in this message" in results[-1]
    assert agent._todo_store.read()[0]["status"] == "completed"
    assert agent._todo_store.read()[1]["status"] == "in_progress"


def test_one_task_per_message_spreads_the_notices(agent) -> None:
    _send(agent, "m1", ("todo", {"action": "begin", "item_id": "1"}))
    agent._todo_store.drain_notices()  # the begin's own start line
    _send(agent, "m2", _work())
    _send(agent, "m3", _complete("1"))
    assert [(n["kind"], n["id"]) for n in agent._todo_store.drain_notices()] == [
        (NOTICE_COMPLETED, "1"),
        (NOTICE_STARTED, "2"),
    ]

    _send(agent, "m4", _work())
    _send(agent, "m5", _complete("2"))
    assert [(n["kind"], n["id"]) for n in agent._todo_store.drain_notices()] == [
        (NOTICE_COMPLETED, "2")
    ]
    assert agent._todo_store.read()[0]["status"] == "completed"
    assert agent._todo_store.read()[1]["status"] == "completed"
