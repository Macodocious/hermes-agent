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
    already closing so the model can recover. The overlap is now
    constructible only by internal mutation — begin refuses it outright
    — mirroring the verdict tests.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    for item in store._items:
        if item["id"] == "2":
            item["status"] = "in_progress"
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


def test_begin_refused_while_another_task_is_closing(store: TodoStore) -> None:
    """Sequential close: begin is refused while a closing task waits on
    the judge — closing occupies the current-task slot like in_progress.

    This was the root-cause hole: close(1) moved task 1 to closing, then
    begin(2) succeeded because the pivot rule only guarded in_progress.
    The overlap stranded the closing task (or forced the PR #66
    auto-finalize chain). Refusal, naming the closing task.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    result = store.transition("begin", "2")
    assert result["ok"] is False
    assert "closing" in result["error"]
    assert "1" in result["error"]
    # Task 1 stays closing; task 2 stays pending — no in_progress item.
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "closing", "2": "pending"}


def test_begin_allowed_after_judge_finalizes(store: TodoStore) -> None:
    """Sequential close: begin is a blocking gate, not a permanent lock.

    Once the judge finalizes the closing task (done verdict second key),
    the slot frees and the next task can begin. This is the legal flow:
    begin(1) -> close(1) -> finalize(1) -> begin(2).
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    store.finalize("1")
    result = store.transition("begin", "2")
    assert result["ok"] is True
    assert result["item"]["status"] == "in_progress"
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "completed", "2": "in_progress"}


def test_begin_refused_while_closing_named_tasks_sequentially(store: TodoStore) -> None:
    """Sequential close: the begin refusal names the closing task and
    holds across repeated attempts until the judge runs.

    Mirrors the production loop: an agent that ignores the refusal and
    retries begin is refused every time, keeping exactly one current
    task in the list.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    for _ in range(3):
        result = store.transition("begin", "2")
        assert result["ok"] is False
        assert "task 1" in result["error"]
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "closing", "2": "pending"}


def test_write_demotes_in_progress_when_another_task_is_closing() -> None:
    """Write-path backstop: the data layer demotes any in_progress item
    to pending when a closing item occupies the current-task slot.

    begin now refuses the overlap through the transition door, so this
    constructs the illegal state directly (internal mutation) to prove
    the invariant also holds on the raw write path — the enforcement is
    mechanical, not agent compliance.
    """
    s = TodoStore()
    s.write(
        [
            {"id": "1", "content": "Closing task", "status": "closing"},
            {"id": "2", "content": "Sneaky current task", "status": "in_progress"},
        ]
    )
    statuses = {i["id"]: i["status"] for i in s.read()}
    assert statuses == {"1": "closing", "2": "pending"}


def test_begin_refusal_names_the_recovery(store: TodoStore) -> None:
    """The begin refusal is actionable: it says how to free the slot.

    A refusal that only names the closing task leaves the caller stuck —
    the same silent-rollback confusion this lifecycle exists to remove.
    The escape hatches are the judge's done verdict or escalate, so the
    error must name at least one of them and must not mutate the list.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    before = store.read()
    result = store.transition("begin", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    assert "finalize" in result["error"] or "escalate" in result["error"]
    assert store.read() == before


def test_resume_refused_while_another_task_is_closing(store: TodoStore) -> None:
    """Sequential close: a paused task cannot resume while a sibling is
    closing.

    Same bug class as the begin refusal: resuming a paused sibling while
    another task is closing constructs the closing + in_progress overlap
    the write path demotes, so the resume would be silently rolled back
    and the following close refused. Refusal names the closing task and
    the recovery, and leaves the list untouched.
    """
    store.transition("begin", "2")
    store.transition("pause", "2")
    store.transition("begin", "1")
    store.transition("close", "1")
    before = store.read()
    result = store.transition("resume", "2")
    assert result["ok"] is False
    assert "task 1" in result["error"]
    assert "finalize" in result["error"] or "escalate" in result["error"]
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "closing", "2": "paused"}
    assert store.read() == before


def test_resume_still_reverses_a_premature_close(store: TodoStore) -> None:
    """The judge's premature-close reversal must keep working.

    ``observe_verdict`` reverses a continue/wait verdict by resuming the
    closing task itself (closing -> in_progress). That is not a second
    concurrent task, so the sequential-close guard — which excludes the
    item itself — must not block it.
    """
    store.transition("begin", "1")
    store.transition("close", "1")
    result = store.transition("resume", "1")
    assert result["ok"] is True
    assert result["item"]["status"] == "in_progress"
    statuses = {i["id"]: i["status"] for i in store.read()}
    assert statuses == {"1": "in_progress", "2": "pending"}
