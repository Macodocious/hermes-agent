"""The tool-result lifecycle reminder: one line, once per turn, no refusal.

The turn-end audit speaks only after the turn. This speaks at the call that
did the work, which is where the transition belongs. It refuses nothing and
changes no row — the model still owns the transition.
"""

from types import SimpleNamespace

import pytest

from agent.task_manager import LIFECYCLE_TOOL_REMINDER, tool_result_reminder
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
        store.transition("complete", "1")
    return store


def test_reminder_fires_once_when_no_task_is_current() -> None:
    agent = _agent(_store_with("completed"))

    assert tool_result_reminder(agent, "read_file") == LIFECYCLE_TOOL_REMINDER
    assert tool_result_reminder(agent, "read_file") == ""


def test_no_reminder_while_a_task_is_current() -> None:
    agent = _agent(_store_with("in_progress"))

    assert tool_result_reminder(agent, "read_file") == ""


def test_no_reminder_for_the_lifecycle_tool_itself() -> None:
    agent = _agent(_store_with("completed"))

    assert tool_result_reminder(agent, "todo") == ""


def test_no_reminder_without_a_task_list() -> None:
    agent = _agent(TodoStore())

    assert tool_result_reminder(agent, "read_file") == ""


def test_no_reminder_when_the_lifecycle_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr("agent.task_manager._lifecycle_config", lambda: {"enabled": False})
    agent = _agent(_store_with("completed"))

    assert tool_result_reminder(agent, "read_file") == ""
