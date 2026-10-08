"""Tests for the todo lifecycle state machine the agent owns.

Covers the deterministic transition table: begin/pause/resume/complete/
cancel, the single-current-task pivot refusal, the mechanical next-task
activation on complete, and the cancel-reason requirement.

``transition`` returns ``{"ok": True, "item": {...}}`` on success or
``{"ok": False, "error": "..."}`` on refusal — it never raises.
"""

import pytest

from tools.todo_tool import TodoStore


@pytest.fixture
def store() -> TodoStore:
    s = TodoStore()
    s.write(
        [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "pending"},
        ]
    )
    return s


def test_begin_moves_pending_to_in_progress(store: TodoStore) -> None:
    result = store.transition("begin", "1")
    assert result["ok"] is True
    assert result["item"]["status"] == "in_progress"
    assert store.read()[0]["status"] == "in_progress"


def test_begin_refuses_while_another_task_is_in_progress(store: TodoStore) -> None:
    store.transition("begin", "1")
    result = store.transition("begin", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "in_progress", "2": "pending"}


def test_pause_then_resume(store: TodoStore) -> None:
    store.transition("begin", "1")
    paused = store.transition("pause", "1")
    assert paused["ok"] is True
    assert paused["item"]["status"] == "paused"
    resumed = store.transition("resume", "1")
    assert resumed["ok"] is True
    assert resumed["item"]["status"] == "in_progress"


def test_pause_opens_the_slot_for_another_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("pause", "1")
    result = store.transition("begin", "2")
    assert result["ok"] is True
    assert result["item"]["status"] == "in_progress"


def test_resume_refused_while_another_task_is_in_progress(store: TodoStore) -> None:
    store.transition("begin", "2")
    store.transition("pause", "2")
    store.transition("begin", "1")
    before = store.read()
    result = store.transition("resume", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    assert store.read() == before


def test_resume_allowed_after_the_running_task_is_paused(store: TodoStore) -> None:
    store.transition("begin", "2")
    store.transition("pause", "2")
    store.transition("begin", "1")
    store.transition("pause", "1")
    result = store.transition("resume", "2")
    assert result["ok"] is True
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "paused", "2": "in_progress"}


def test_complete_finalizes_the_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    result = store.transition("complete", "1")
    assert result["ok"] is True
    assert store.read()[0]["status"] == "completed"


def test_complete_activates_the_next_pending_task(store: TodoStore) -> None:
    """Completing a task mechanically activates the first pending sibling."""
    store.transition("begin", "1")
    store.transition("complete", "1")
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "completed", "2": "in_progress"}


def test_complete_with_no_pending_sibling_activates_nothing() -> None:
    s = TodoStore()
    s.write([{"id": "1", "content": "Only task", "status": "pending"}])
    s.transition("begin", "1")
    s.transition("complete", "1")
    assert s.read()[0]["status"] == "completed"


def test_complete_refused_from_pending(store: TodoStore) -> None:
    result = store.transition("complete", "1")
    assert result["ok"] is False


def test_cancel_requires_a_reason(store: TodoStore) -> None:
    store.transition("begin", "1")
    result = store.transition("cancel", "1")
    assert result["ok"] is False
    assert "reason" in result["error"]
    assert store.read()[0]["status"] == "in_progress"


def test_cancel_stores_the_reason(store: TodoStore) -> None:
    store.transition("begin", "1")
    result = store.transition("cancel", "1", "no longer needed")
    assert result["ok"] is True
    assert store.read()[0]["status"] == "cancelled"
    assert store.read()[0]["reason"] == "no longer needed"


def test_cancel_of_a_terminal_item_is_refused(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("complete", "1")
    result = store.transition("cancel", "1", "too late")
    assert result["ok"] is False


def test_unknown_action_is_refused(store: TodoStore) -> None:
    result = store.transition("teleport", "1")
    assert result["ok"] is False
    assert "unknown" in result["error"]


def test_removed_judge_actions_are_unknown(store: TodoStore) -> None:
    """close/escalate/block/finalize are gone with the judge — not actions."""
    for action in ("close", "escalate", "block", "finalize"):
        result = store.transition(action, "1")
        assert result["ok"] is False
        assert "unknown" in result["error"]


def test_begin_on_missing_item_is_refused(store: TodoStore) -> None:
    result = store.transition("begin", "nope")
    assert result["ok"] is False


def test_begin_on_completed_item_is_refused(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("complete", "1")
    result = store.transition("begin", "1")
    assert result["ok"] is False


def test_write_demotes_extra_in_progress_to_pending() -> None:
    """The data layer keeps at most one in_progress item."""
    s = TodoStore()
    s.write(
        [
            {"id": "1", "content": "A", "status": "in_progress"},
            {"id": "2", "content": "B", "status": "in_progress"},
        ]
    )
    statuses = {i["id"]: i["status"] for i in s.read()}
    assert statuses == {"1": "in_progress", "2": "pending"}


def test_complete_records_start_and_complete_notices() -> None:
    """The store records a notice per start/complete for the emitter."""
    s = TodoStore()
    s.write([{"id": "1", "content": "A", "status": "pending"}])
    s.transition("begin", "1")
    s.transition("complete", "1")
    assert [n["kind"] for n in s.drain_notices()] == ["started", "completed"]
    assert s.drain_notices() == []
