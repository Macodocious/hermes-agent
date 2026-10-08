"""E2E: the real todo dispatch path drives the agent-owned lifecycle.

Exercises ``agent_runtime_helpers.invoke_tool`` for the ``todo`` tool end to
end: todo_tool -> persist_todo_store -> on_todo_write (persist + stamp). The
lifecycle no longer arms a goal and no longer observes a verdict — the store
owns the transitions and the agent's own complete action finalizes a task.

External seams are pinned: SessionDB persistence is faked, so the test is
deterministic and needs no real goals provider.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import invoke_tool
from tools.todo_tool import TodoStore


def _make_agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="e2e-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store: TodoStore, item_id: str, content: str) -> None:
    store.write([{"id": item_id, "content": content, "status": "pending"}])


@pytest.fixture
def dispatched(monkeypatch, tmp_path):
    """Pin persistence + the lifecycle config; return (agent, calls)."""
    calls: list = []
    monkeypatch.setattr(
        "hermes_cli.tasks.persist_todo_store", lambda agent: calls.append("persist")
    )
    monkeypatch.setattr("agent.task_manager._persist", lambda agent: None)
    monkeypatch.setattr(
        "agent.task_manager._lifecycle_config", lambda: {"enabled": True}
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    return _make_agent(store), calls


def _invoke(agent, action: str, item_id: str, **extra) -> None:
    invoke_tool(
        agent,
        "todo",
        {"action": action, "item_id": item_id, **extra},
        effective_task_id="",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    )


def test_begin_write_transitions_store_and_stamps(dispatched) -> None:
    agent, calls = dispatched
    _invoke(agent, "begin", "1")

    assert agent._todo_store.read()[0]["status"] == "in_progress"
    assert agent._task_lifecycle_action_issued is True
    assert "persist" in calls


def test_complete_write_finalizes_without_a_verdict(dispatched) -> None:
    """The agent's complete action alone marks the task done."""
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "complete", "1")

    assert agent._todo_store.read()[0]["status"] == "completed"


def test_cancel_write_requires_a_reason(dispatched) -> None:
    """A reasonless cancel is refused; the task stays in_progress."""
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "cancel", "1")

    assert agent._todo_store.read()[0]["status"] == "in_progress"


def test_cancel_write_with_reason_cancels(dispatched) -> None:
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "cancel", "1", reason="not needed")

    assert agent._todo_store.read()[0]["status"] == "cancelled"
