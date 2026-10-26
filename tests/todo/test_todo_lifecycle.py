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


def test_pause_does_not_open_the_slot_for_a_later_task(store: TodoStore) -> None:
    """Pausing the head keeps it open, so a later task cannot start in front.

    Pause is not a way to skip a task. The head stays the head until it is
    completed or cancelled, so a later row is refused exactly as it is while
    the head is in progress.
    """
    store.transition("begin", "1")
    store.transition("pause", "1")
    result = store.transition("begin", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "paused", "2": "pending"}


def test_complete_refused_while_an_earlier_task_is_open() -> None:
    """Only the head may close. A later row closing first is refused.

    The list is a sequence: a row cannot be finished while the row in front of
    it is still open, because that is exactly the jump that produced the
    completion-then-start-before-it notices.
    """
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "in_progress"},
        ]
    )
    before = store.read()
    result = store.transition("complete", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    assert store.read() == before


def test_complete_promotes_the_immediate_successor(store: TodoStore) -> None:
    """Finishing the head starts the row after it, not the first pending row."""
    store.transition("begin", "1")
    store.mark_work()
    assert store.transition("complete", "1")["ok"] is True
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "completed", "2": "in_progress"}


def test_complete_promotes_a_paused_successor() -> None:
    """A paused row after the head becomes current when the head completes.

    Refusing this instead would deadlock: the paused row cannot resume while
    the head is open, and the head could not complete while the row is paused.
    """
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "First task", "status": "in_progress"},
            {"id": "2", "content": "Second task", "status": "paused"},
        ]
    )
    store.mark_work()
    assert store.transition("complete", "1")["ok"] is True
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "completed", "2": "in_progress"}


def test_complete_finalizes_the_task(store: TodoStore) -> None:
    store.transition("begin", "1")
    store.mark_work()
    result = store.transition("complete", "1")
    assert result["ok"] is True
    assert store.read()[0]["status"] == "completed"


def test_complete_promotes_the_next_task(store: TodoStore) -> None:
    """Completing a task begins the next one from the store.

    The promotion makes the next row current; its ``▶️ Working on:`` line is
    recorded at the first call under it, not at this completion.
    """
    store.transition("begin", "1")
    store.mark_work()
    store.transition("complete", "1")
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "completed", "2": "in_progress"}


def test_complete_with_no_pending_sibling_activates_nothing() -> None:
    s = TodoStore()
    s.write([{"id": "1", "content": "Only task", "status": "pending"}])
    s.transition("begin", "1")
    s.mark_work()
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
    store.mark_work()
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
    store.mark_work()
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


def test_complete_records_the_start_and_holds_the_final_close() -> None:
    """The start line waits for work; the final close waits for turn end."""
    s = TodoStore()
    s.write([{"id": "1", "content": "A", "status": "pending"}])
    s.transition("begin", "1")
    assert s.drain_notices() == []
    s.mark_work()
    assert [n["kind"] for n in s.drain_notices()] == ["started"]
    s.transition("complete", "1")
    assert [n["kind"] for n in s.pending_notices()] == ["completed"]
    assert s.pending_notices()[0]["held"] is True
    s.release_held_notices()
    assert [n["kind"] for n in s.drain_notices()] == ["completed"]


def test_cancelling_the_current_task_makes_the_next_one_current() -> None:
    """Cancel is terminal like complete: the sequence keeps moving."""
    s = TodoStore()
    s.write(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "2", "content": "B", "status": "pending"},
        ]
    )
    s.transition("begin", "1")

    s.transition("cancel", "1", "not needed after all")

    statuses = {i["id"]: i["status"] for i in s.read()}
    assert statuses == {"1": "cancelled", "2": "in_progress"}


def test_cancelling_a_later_task_moves_nothing() -> None:
    """Only the current task's cancellation promotes; dropping a later row is local."""
    s = TodoStore()
    s.write(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "2", "content": "B", "status": "pending"},
            {"id": "3", "content": "C", "status": "pending"},
        ]
    )
    s.transition("begin", "1")

    s.transition("cancel", "3", "dropped from the plan")

    statuses = {i["id"]: i["status"] for i in s.read()}
    assert statuses == {"1": "in_progress", "2": "pending", "3": "cancelled"}
