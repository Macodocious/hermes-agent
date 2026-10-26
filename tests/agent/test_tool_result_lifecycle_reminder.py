"""The tool-result lifecycle reminder: one line per worked call, no refusal.

The turn-end audit speaks only after the turn. This speaks at the call that
did the work, which is where the transition belongs. It refuses nothing and
changes no row — the model still owns the transition.
"""

from types import SimpleNamespace

import pytest

from agent.task_manager import (
    LIFECYCLE_TOOL_REMINDER,
    PACING_CONTENT_MAX_CHARS,
    tool_result_reminder,
)
from tools.todo_tool import TodoStore


def _agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(_todo_store=store, _lifecycle_reminder_issued=False)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr("agent.task_manager._lifecycle_config", lambda: {"enabled": True})


def _store_with(status: str) -> TodoStore:
    store = TodoStore()
    store.write([{"id": "1", "content": "Build the thing", "status": "pending"}])
    if status != "pending":
        store.transition("begin", "1")
    if status == "completed":
        store.mark_work()
        store.transition("complete", "1")
    return store


def test_reminder_fires_once_when_no_task_is_current() -> None:
    agent = _agent(_store_with("completed"))

    assert tool_result_reminder(agent, "read_file") == LIFECYCLE_TOOL_REMINDER
    assert tool_result_reminder(agent, "read_file") == ""


def test_pacing_fires_on_every_worked_call_and_the_count_climbs() -> None:
    store = _store_with("in_progress")
    agent = _agent(store)

    store.mark_work()
    first = tool_result_reminder(agent, "read_file")
    assert first.startswith('[Task lifecycle] Task 1 "Build the thing" is current — ')
    assert "— 1 tool calls under it" in first
    assert "Close it with todo action=complete when its work is done." in first

    store.mark_work()
    second = tool_result_reminder(agent, "read_file")
    assert "— 2 tool calls under it" in second

    store.mark_work()
    assert "— 3 tool calls under it" in tool_result_reminder(agent, "read_file")


def test_pacing_restarts_for_a_task_promoted_in_the_same_turn() -> None:
    store = TodoStore()
    store.write([
        {"id": "1", "content": "First", "status": "pending"},
        {"id": "2", "content": "Second", "status": "pending"},
    ])
    store.transition("begin", "1")
    agent = _agent(store)
    assert "Task 1 " in tool_result_reminder(agent, "read_file")

    store.mark_work()
    store.transition("complete", "1")

    line = tool_result_reminder(agent, "read_file")
    assert "Task 2 " in line
    assert "— 0 tool calls under it" in line


def test_pacing_caps_long_content() -> None:
    store = TodoStore()
    store.write([{"id": "1", "content": "x" * 500, "status": "pending"}])
    store.transition("begin", "1")

    line = tool_result_reminder(_agent(store), "read_file")
    assert '"' + "x" * PACING_CONTENT_MAX_CHARS + "…\"" in line


def test_count_survives_json_round_trip() -> None:
    store = _store_with("in_progress")
    store.mark_work()
    store.mark_work()
    restored = TodoStore.from_json(store.to_json())

    assert restored.work_count() == 2


def test_no_reminder_for_the_lifecycle_tool_itself() -> None:
    agent = _agent(_store_with("completed"))

    assert tool_result_reminder(agent, "todo") == ""


def test_no_reminder_without_a_task_list() -> None:
    agent = _agent(TodoStore())

    assert tool_result_reminder(agent, "read_file") == ""


def test_no_reminder_when_the_lifecycle_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr("agent.task_manager._lifecycle_config", lambda: {"enabled": False})

    assert tool_result_reminder(_agent(_store_with("completed")), "read_file") == ""
    assert tool_result_reminder(_agent(_store_with("in_progress")), "read_file") == ""
