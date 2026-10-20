"""Fake live sessions: the real agent loop, the real tool executor, the real store.

Each test scripts the model's turns and lets ``run_conversation`` run for real,
so the reminder is observed where the model would observe it — in the tool
result message the loop appends to the conversation.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.task_manager import LIFECYCLE_TOOL_REMINDER
from run_agent import AIAgent
from tools.todo_tool import TodoStore


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response(content: str = "", tool_calls=None) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(
        message=message, finish_reason="tool_calls" if tool_calls else "stop"
    )
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def live_agent(monkeypatch):
    monkeypatch.setattr("hermes_cli.tasks.load_todo", lambda _session_id: None)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("read_file"), _tool_def("todo")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
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
    return agent


def _run(agent, store: TodoStore, responses: list) -> dict:
    agent._todo_store = store
    agent.client.chat.completions.create.side_effect = responses
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("do the work")


def _tool_results(result: dict) -> list:
    return [m for m in result["messages"] if m.get("role") == "tool"]


def _completed_store() -> TodoStore:
    store = TodoStore()
    store.write([{"id": "1", "content": "Old task", "status": "pending"}])
    store.transition("begin", "1")
    store.transition("complete", "1")
    return store


def test_live_turn_with_no_current_task_gets_the_reminder(live_agent, tmp_path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello")

    result = _run(
        live_agent,
        _completed_store(),
        [
            _response(tool_calls=[_tool_call("read_file", {"path": str(target)}, "c1")]),
            _response(content="done"),
        ],
    )

    results = _tool_results(result)
    assert results, result["messages"]
    assert LIFECYCLE_TOOL_REMINDER in results[-1]["content"]


def test_live_turn_with_a_current_task_gets_no_reminder(live_agent, tmp_path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello")
    store = TodoStore()
    store.write([{"id": "1", "content": "Current task", "status": "pending"}])
    store.transition("begin", "1")

    result = _run(
        live_agent,
        store,
        [
            _response(tool_calls=[_tool_call("read_file", {"path": str(target)}, "c1")]),
            _response(content="done"),
        ],
    )

    results = _tool_results(result)
    assert results, result["messages"]
    assert LIFECYCLE_TOOL_REMINDER not in results[-1]["content"]


def test_live_turn_starting_empty_is_seeded_and_reminded(live_agent, tmp_path) -> None:
    """An empty store is seeded from the opening message, so the line fires."""
    target = tmp_path / "note.txt"
    target.write_text("hello")

    result = _run(
        live_agent,
        TodoStore(),
        [
            _response(tool_calls=[_tool_call("read_file", {"path": str(target)}, "c1")]),
            _response(content="done"),
        ],
    )

    results = _tool_results(result)
    assert results, result["messages"]
    assert LIFECYCLE_TOOL_REMINDER in results[-1]["content"]
    assert live_agent._todo_store.read()[0]["status"] == "pending"


def test_live_turn_begins_the_seeded_task_and_stops_the_reminder(
    live_agent, tmp_path
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello")
    store = TodoStore()
    store.seed_from_user_message("do the work")

    result = _run(
        live_agent,
        store,
        [
            _response(tool_calls=[_tool_call("read_file", {"path": str(target)}, "c1")]),
            _response(
                tool_calls=[_tool_call("todo", {"action": "begin", "item_id": "1"}, "c2")]
            ),
            _response(tool_calls=[_tool_call("read_file", {"path": str(target)}, "c3")]),
            _response(content="done"),
        ],
    )

    results = _tool_results(result)
    assert len(results) == 3, results
    assert LIFECYCLE_TOOL_REMINDER in results[0]["content"]
    assert LIFECYCLE_TOOL_REMINDER not in results[2]["content"]
    assert live_agent._todo_store.read()[0]["status"] == "in_progress"
