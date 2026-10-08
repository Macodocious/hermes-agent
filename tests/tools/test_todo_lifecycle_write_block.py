"""Tests for the lifecycle write block.

The model drives task state through the ``action`` parameter; a direct
lifecycle status write through the todos list bypasses the transition
table, so the tool refuses any todos-list write that would CHANGE a
lifecycle status, naming the action to use instead. ``completed`` and
``cancelled`` are no longer redirected to a judge-owned close action — the
agent owns them — but they are still driven by their own actions.
"""

import json

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
        assert store.read()[0]["status"] == "pending"

    def test_paused_write_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "paused"}], merge=True)
        assert "error" in result
        assert "action=pause" in result["error"]

    def test_completed_on_agent_item_redirects_to_the_agent_action(self):
        """The redirect names the agent-owned complete action, not a judge close."""
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = _write(store, [{"id": "1", "status": "completed"}], merge=True)
        assert "error" in result
        assert "action=complete" in result["error"]
        assert "action=close" not in result["error"]

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

    def test_pending_write_is_allowed(self):
        store = TodoStore()
        result = _write(store, [{"id": "1", "content": "Task", "status": "pending"}])
        assert "error" not in result
        assert result["summary"]["pending"] == 1


class TestStoreBoundaryNotEnforced:
    """The store is not the boundary: hydration, seeding, and internal
    code write lifecycle statuses directly."""

    def test_store_write_accepts_in_progress(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "in_progress"}])
        assert store.read()[0]["status"] == "in_progress"

    def test_store_write_accepts_completed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "completed"}])
        assert store.read()[0]["status"] == "completed"


class TestSchemaTeachesTheRule:
    """The tool schema is the instruction surface."""

    def test_schema_action_enum_is_agent_owned(self):
        from tools.todo_tool import TODO_SCHEMA

        enum = TODO_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert set(enum) == {"begin", "complete", "cancel", "pause", "resume"}

    def test_schema_declares_reason_parameter(self):
        from tools.todo_tool import TODO_SCHEMA

        props = TODO_SCHEMA["parameters"]["properties"]
        assert props["reason"]["type"] == "string"


class TestCancelActionToolEntry:
    """The tool entry requires a reason for cancel and passes it through."""

    def test_cancel_without_reason_is_refused(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        store.transition("begin", "1")
        result = json.loads(todo_tool(action="cancel", item_id="1", store=store))
        assert "error" in result
        assert "reason" in result["error"]

    def test_cancel_with_reason_succeeds(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        store.transition("begin", "1")
        result = json.loads(
            todo_tool(action="cancel", item_id="1", reason="not needed", store=store)
        )
        assert "error" not in result
        item = next(i for i in result["todos"] if i["id"] == "1")
        assert item["status"] == "cancelled"
        assert item["reason"] == "not needed"
