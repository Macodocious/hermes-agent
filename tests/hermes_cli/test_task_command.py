"""Tests for the CLI `/task` command (current task + full task list)."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from tools.todo_tool import TodoStore


class _Stub(CLICommandsMixin):
    def __init__(self, agent=None, session_id="sess-1"):
        self.agent = agent
        self.session_id = session_id


def _make_store() -> TodoStore:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "Working thing", "status": "in_progress"},
            {"id": "3", "content": "Next thing", "status": "pending"},
        ]
    )
    store.capture_request("Captured ask")
    return store


def _run(stub):
    lines = []
    with patch("cli._cprint", side_effect=lambda text: lines.append(text)):
        stub._handle_task_command()
    return lines


def test_task_command_shows_current_task_and_full_list():
    agent = SimpleNamespace(_todo_store=_make_store())
    lines = _run(_Stub(agent=agent))

    assert any("Working on:" not in line for line in lines)
    assert any("- [x] 1. Done thing" in line for line in lines)
    assert any("- [>] 2. Working thing" in line for line in lines)
    assert not any("← CURRENT TASK" in line for line in lines)
    assert any("- [ ] 3. Next thing" in line for line in lines)
    assert any("Captured requests:" in line for line in lines)
    assert any("[captured] c1. Captured ask" in line for line in lines)


def test_task_command_falls_back_to_persisted_store(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.tasks.load_todo",
        lambda session_id: _make_store() if session_id == "sess-1" else None,
    )
    lines = _run(_Stub(agent=None))

    assert any("Working on:" not in line for line in lines)
    assert any("- [>] 2. Working thing" in line for line in lines)
    assert not any("← CURRENT TASK" in line for line in lines)


def test_task_command_empty_store_reports_empty():
    agent = SimpleNamespace(_todo_store=TodoStore())
    lines = _run(_Stub(agent=agent))

    assert any("The task list for this session is empty." in line for line in lines)


def test_task_command_no_in_progress_reports_none():
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "Next thing", "status": "pending"},
        ]
    )
    lines = _run(_Stub(agent=SimpleNamespace(_todo_store=store)))

    assert any("Working on:" not in line for line in lines)
    assert any("- [x] 1. Done thing" in line for line in lines)
    assert any("- [ ] 2. Next thing" in line for line in lines)


def test_task_command_no_store_and_no_row_reports_absent(monkeypatch):
    monkeypatch.setattr("hermes_cli.tasks.load_todo", lambda session_id: None)
    lines = _run(_Stub(agent=None))

    assert any("No task list for this session yet" in line for line in lines)
