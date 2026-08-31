"""Tests for the task-stopped notification: executor helper + CLI spinner.

Covers ``_detect_todo_task_stop`` (the pre-call transition hook wired into
both ``tool.started`` emitter sites) and the CLI spinner override that
renders "Task cancelled/escalated" instead of the generic todo label.
"""

from types import SimpleNamespace

from agent.tool_executor import _detect_todo_task_stop
from tools.todo_tool import TodoStore


def _agent_with_store(items=None):
    store = TodoStore()
    if items:
        store.write(items)
    return SimpleNamespace(_todo_store=store), store


class TestDetectTodoTaskStop:
    """Helper-level transition detection (gateway- and CLI-facing)."""

    def test_non_todo_tool_returns_none(self):
        agent, _ = _agent_with_store()
        assert _detect_todo_task_stop(agent, "terminal", {"command": "ls"}) is None

    def test_read_call_returns_none(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        assert _detect_todo_task_stop(agent, "todo", {}) is None

    def test_cancel_transition_detected(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "cancelled"}]}
        )
        assert stopped is not None
        assert stopped["id"] == "1"
        assert stopped["content"] == "task one"
        assert stopped["status"] == "cancelled"

    def test_escalate_transition_detected(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "escalated"}]}
        )
        assert stopped is not None
        assert stopped["status"] == "escalated"

    def test_closing_to_cancelled_detected(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "closing"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "cancelled"}]}
        )
        assert stopped is not None
        assert stopped["status"] == "cancelled"

    def test_pending_to_cancelled_returns_none(self):
        # A task that was never active cannot "stop".
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "cancelled"}]}
        )
        assert stopped is None

    def test_reassert_returns_none(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "cancelled"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "cancelled"}]}
        )
        assert stopped is None

    def test_agent_without_store_returns_none(self):
        agent = SimpleNamespace()
        stopped = _detect_todo_task_stop(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "cancelled"}]}
        )
        assert stopped is None

    def test_malformed_args_returns_none(self):
        agent, _ = _agent_with_store()
        assert _detect_todo_task_stop(agent, "todo", {"todos": "not json"}) is None
        assert _detect_todo_task_stop(agent, "todo", None) is None


class TestCliSpinnerTaskStop:
    """CLI _on_tool_progress renders the stopped label for stopped tasks."""

    def _make_cli(self):
        import importlib
        mod = importlib.import_module("cli")
        return mod.HermesCLI(verbose=False), mod

    def test_cancelled_label_replaces_generic(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": "ship the feature", "status": "cancelled"}]},
            stopped_task={"id": "1", "content": "ship the feature", "status": "cancelled"},
        )
        assert cli._spinner_text == "📋 Task cancelled: ship the feature"

    def test_escalated_label_replaces_generic(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": "ship the feature", "status": "escalated"}]},
            stopped_task={"id": "1", "content": "ship the feature", "status": "escalated"},
        )
        assert cli._spinner_text == "📋 Task escalated: ship the feature"

    def test_generic_label_when_no_stopped_task(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Reading the task list", {},
        )
        assert cli._spinner_text == "📋 Reading the task list"

    def test_stopped_label_truncates_long_content(self, monkeypatch):
        import agent.display
        monkeypatch.setattr(agent.display, "get_tool_preview_max_len", lambda: 40)
        cli, _ = self._make_cli()
        long_content = "x" * 200
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": long_content, "status": "cancelled"}]},
            stopped_task={"id": "1", "content": long_content, "status": "cancelled"},
        )
        assert cli._spinner_text.startswith("📋 Task cancelled: ")
        assert cli._spinner_text.endswith("...")
        assert len(cli._spinner_text) < 60
