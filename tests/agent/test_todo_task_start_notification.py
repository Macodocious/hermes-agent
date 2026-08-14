"""Tests for the task-started notification: executor helper + CLI spinner.

Covers ``_detect_todo_task_start`` (the pre-call transition hook wired into
both ``tool.started`` emitter sites) and the CLI spinner override that
renders "Working on <task>" instead of the generic todo label.
"""

from types import SimpleNamespace

from agent.tool_executor import _detect_todo_task_start
from tools.todo_tool import TodoStore


def _agent_with_store(items=None):
    store = TodoStore()
    if items:
        store.write(items)
    return SimpleNamespace(_todo_store=store), store


class TestDetectTodoTaskStart:
    """Helper-level transition detection (gateway- and CLI-facing)."""

    def test_non_todo_tool_returns_none(self):
        agent, _ = _agent_with_store()
        assert _detect_todo_task_start(agent, "terminal", {"command": "ls"}) is None

    def test_read_call_returns_none(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        assert _detect_todo_task_start(agent, "todo", {}) is None

    def test_transition_detected(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "pending"},
        ])
        started = _detect_todo_task_start(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "in_progress"}]}
        )
        assert started is not None
        assert started["id"] == "1"
        assert started["content"] == "task one"

    def test_reassert_returns_none(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "task one", "status": "in_progress"},
        ])
        started = _detect_todo_task_start(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "in_progress"}]}
        )
        assert started is None

    def test_agent_without_store_returns_none(self):
        agent = SimpleNamespace()
        started = _detect_todo_task_start(
            agent, "todo", {"todos": [{"id": "1", "content": "task one", "status": "in_progress"}]}
        )
        assert started is None

    def test_malformed_args_returns_none(self):
        agent, _ = _agent_with_store()
        assert _detect_todo_task_start(agent, "todo", {"todos": "not json"}) is None
        assert _detect_todo_task_start(agent, "todo", None) is None


class TestCliSpinnerTaskStart:
    """CLI _on_tool_progress renders the working-on label for started tasks."""

    def _make_cli(self):
        import importlib
        mod = importlib.import_module("cli")
        return mod.HermesCLI(verbose=False), mod

    def test_working_on_label_replaces_generic(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": "ship the feature", "status": "in_progress"}]},
            started_task={"id": "1", "content": "ship the feature", "status": "in_progress"},
        )
        assert cli._spinner_text == "📋 Working on ship the feature"

    def test_generic_label_when_no_started_task(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Reading the task list", {},
        )
        assert cli._spinner_text == "📋 Reading the task list"

    def test_working_on_truncates_long_content(self, monkeypatch):
        import agent.display
        monkeypatch.setattr(agent.display, "get_tool_preview_max_len", lambda: 40)
        cli, _ = self._make_cli()
        long_content = "x" * 200
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": long_content, "status": "in_progress"}]},
            started_task={"id": "1", "content": long_content, "status": "in_progress"},
        )
        assert cli._spinner_text.startswith("📋 Working on ")
        assert cli._spinner_text.endswith("...")
        assert len(cli._spinner_text) < 60
