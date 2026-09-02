"""Tests for the todo lifecycle state machine (P1).

Covers the deterministic transition table: begin/pause/resume/close/
escalate/finalize, the pivot refusal (no second in_progress task), and
the two-key close (closing only finalizes via the judge's done verdict).

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
    # Task 1 stays current; task 2 stays pending.
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


def test_close_enters_closing_not_completed(store: TodoStore) -> None:
    store.transition("begin", "1")
    closed = store.transition("close", "1")
    assert closed["ok"] is True
    assert closed["item"]["status"] == "closing"
    # The judge's done verdict is the second key — close alone does not
    # complete the task.
    assert store.read()[0]["status"] == "closing"


def test_finalize_completes_a_closing_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("close", "1")
    result = store.finalize("1")
    assert result["ok"] is True
    assert store.read()[0]["status"] == "completed"


def test_finalize_refuses_non_closing_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    result = store.finalize("1")
    assert result["ok"] is False


def test_escalate_marks_escalated(store: TodoStore) -> None:
    store.transition("begin", "1")
    escalated = store.transition("escalate", "1")
    assert escalated["ok"] is True
    assert escalated["item"]["status"] == "escalated"


def test_escalate_opens_the_slot_for_another_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("escalate", "1")
    result = store.transition("begin", "2")
    assert result["ok"] is True
    assert result["item"]["status"] == "in_progress"


def test_unknown_action_is_refused(store: TodoStore) -> None:
    result = store.transition("teleport", "1")
    assert result["ok"] is False
    assert "unknown" in result["error"]


def test_begin_on_missing_item_is_refused(store: TodoStore) -> None:
    result = store.transition("begin", "nope")
    assert result["ok"] is False


def test_begin_on_completed_item_is_refused(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.transition("close", "1")
    store.finalize("1")
    result = store.transition("begin", "1")
    assert result["ok"] is False


def test_close_refused_while_another_task_is_closing(store: TodoStore) -> None:
    """Revised fix 4: only one task may be closing at a time.

    The judge finalizes exactly one closing task per done verdict, so a
    second closing task would strand forever. The refusal names the task
    already closing so the model can recover.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    store.transition("begin", "2")
    result = store.transition("close", "2")
    assert result["ok"] is False
    assert "already closing" in result["error"]
    assert "1" in result["error"]
    # Task 2 stays in_progress; task 1 stays closing.
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "closing", "2": "in_progress"}


def test_close_after_finalize_opens_the_slot(store: TodoStore) -> None:
    """Revised fix 4: once the judge finalizes the closing task, the next
    task can close — the invariant is per-closing-task, not permanent."""
    store.transition("begin", "1")
    store.transition("close", "1")
    store.finalize("1")
    store.transition("begin", "2")
    result = store.transition("close", "2")
    assert result["ok"] is True
    assert result["item"]["status"] == "closing"


def test_reclose_from_closing_is_idempotent(store: TodoStore) -> None:
    """Fix 5: close from closing is allowed (idempotent re-close).

    A stale in-memory store can show a task as closing while the DB has
    already finalized it; re-close must not deadlock the agent. The task
    stays closing awaiting the judge.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    result = store.transition("close", "1")
    assert result["ok"] is True
    assert result["item"]["status"] == "closing"
    assert store.read()[0]["status"] == "closing"
