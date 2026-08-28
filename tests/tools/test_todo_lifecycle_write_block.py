"""Tests for the lifecycle write block (P7).

The legacy write path (``todos=[...]`` with a lifecycle status) was a
complete bypass of the task lifecycle: an ``in_progress`` write silenced
the turn-end audit, never armed the GoalEngine (arming keys on
``action``), and never reached the judge or the post-close review. The
tool now refuses any write that would CHANGE a lifecycle status, forcing
the model through ``action=begin/pause/resume/close/escalate``.

The store itself is NOT the boundary — hydration, seeding, and internal
code (task_manager) write lifecycle statuses directly. Only the
model-facing tool entry point enforces the block.
"""

import json

import pytest

from tools.todo_tool import TodoStore, todo_tool


def _write(store: TodoStore, todos, merge: bool = False) -> dict:
    return json.loads(todo_tool(todos=todos, merge=merge, store=store))


class TestLifecycleStatusWritesRefused:
    def test_in_progress_on_new_item_is_refused(self):
        store = TodoStore()
        result = _write(store, [{"id": "1", "content": "Task", "status": "in_progress"}])
        assert "error" in result
        assert "action=begin" in result["error"]

    def test_in_progress_on_existing_pending_item_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "in_progress"}], merge=True)
        assert "error" in result
        assert "action=begin" in result["error"]
        # The write must not be applied.
        assert store.read()[0]["status"] == "pending"

    def test_paused_write_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "paused"}], merge=True)
        assert "error" in result
        assert "action=pause" in result["error"]

    def test_closing_write_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "closing"}], merge=True)
        assert "error" in result
        assert "action=close" in result["error"]

    def test_escalated_write_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "escalated"}], merge=True)
        assert "error" in result
        assert "action=escalate" in result["error"]

    def test_completed_on_agent_item_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "completed"}], merge=True)
        assert "error" in result
        assert "action=close" in result["error"]

    def test_refused_write_leaves_store_unchanged(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        _write(store, [{"id": "1", "status": "in_progress"}], merge=True)
        items = store.read()
        assert len(items) == 1
        assert items[0]["status"] == "pending"


class TestLifecycleStatusWritesAllowed:
    def test_noop_echo_of_current_in_progress_is_allowed(self):
        """Replace-mode list maintenance echoes the current state; a no-op
        echo is never a transition and must keep working."""
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        store.transition("begin", "1")
        result = _write(store, [{"id": "1", "content": "Task", "status": "in_progress"}])
        assert "error" not in result
        assert result["summary"]["in_progress"] == 1

    def test_completed_on_user_sourced_item_is_allowed(self):
        """P4: user-sourced items stay markable completed directly."""
        store = TodoStore()
        store.write([{"id": "1", "content": "User task", "status": "pending", "source": "user"}])
        result = _write(store, [{"id": "1", "status": "completed"}], merge=True)
        assert "error" not in result
        assert result["summary"]["completed"] == 1

    def test_cancelled_on_agent_item_is_allowed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "cancelled"}], merge=True)
        assert "error" not in result
        assert result["summary"]["cancelled"] == 1

    def test_pending_write_is_allowed(self):
        store = TodoStore()
        result = _write(store, [{"id": "1", "content": "Task", "status": "pending"}])
        assert "error" not in result
        assert result["summary"]["pending"] == 1


class TestStoreBoundaryNotEnforced:
    """The store is not the boundary: hydration, seeding, and internal
    code (task_manager) write lifecycle statuses directly."""

    def test_store_write_accepts_in_progress(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "in_progress"}])
        assert store.read()[0]["status"] == "in_progress"

    def test_store_write_accepts_completed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "completed"}])
        assert store.read()[0]["status"] == "completed"


class TestSchemaTeachesTheRule:
    """The tool schema is the instruction surface (design contract:
    behavioral guidance lives in the schema description)."""

    def test_schema_mentions_action_driven_lifecycle(self):
        from tools.todo_tool import TODO_SCHEMA
        desc = TODO_SCHEMA["description"]
        assert "driven by the action parameter" in desc
        assert "action=close" in desc
