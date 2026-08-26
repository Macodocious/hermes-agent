"""Tests for the turn-end audit wiring in the finalizer (P2).

The finalizer must attach the task-lifecycle pull-back nudge to the turn
result when the audit fires, and stamp it on the agent for the CLI/TUI
loop paths.
"""

from types import SimpleNamespace

from agent import task_manager


def _make_agent(store) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_max_turns=0,
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store, item_id: str, content: str) -> None:
    store.write([{"id": item_id, "content": content, "status": "pending"}])


def test_audit_returns_nudge_for_work_without_open_task() -> None:
    from tools.todo_tool import TodoStore

    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)

    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is not None
    assert "begin" in nudge


def test_audit_returns_none_for_clean_turn() -> None:
    from tools.todo_tool import TodoStore

    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    store.transition("begin", "1")

    nudge = task_manager.audit_turn_end(
        agent, final_response="Done.", interrupted=False, tool_call_count=3
    )
    assert nudge is None


def test_audit_returns_none_when_no_task_list() -> None:
    from tools.todo_tool import TodoStore

    agent = _make_agent(TodoStore())

    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is None
