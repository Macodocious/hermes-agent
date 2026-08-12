"""Tests for hermes_cli/tasks.py — persistent task state (P1/P5).

Mirrors the hermes_cli/goals.py test fixture pattern: isolated HERMES_HOME
so SessionDB.state_meta writes never touch the real database.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # hermes_state.DEFAULT_DB_PATH is frozen at module import time, and other
    # test modules (e.g. test_turn_context.py) import hermes_state at module
    # level — so by the time this fixture runs, the constant may already point
    # at the real state.db. Pin it explicitly so SessionDB() resolves to the
    # isolated home regardless of import order.
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    # Bust the tasks-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import tasks

    tasks._DB_CACHE.clear()
    yield home
    tasks._DB_CACHE.clear()


class _StubAgent:
    """Minimal stand-in carrying the attributes persist_todo_store reads."""

    def __init__(self, session_id: str = ""):
        from tools.todo_tool import TodoStore

        self.session_id: Optional[str] = session_id or None
        self._todo_store: Optional[Any] = TodoStore()
        self._persist_disabled = False


class TestSaveLoadTodo:
    def test_save_and_load_round_trip(self, hermes_home):
        from hermes_cli.tasks import load_todo, save_todo
        from tools.todo_tool import TodoStore

        store = TodoStore()
        store.write([{"id": "1", "content": "Ship it", "status": "in_progress"}])
        save_todo("sess-1", store)

        loaded = load_todo("sess-1")
        assert loaded is not None
        assert loaded.read() == store.read()

    def test_load_missing_session_returns_none(self, hermes_home):
        from hermes_cli.tasks import load_todo

        assert load_todo("nope") is None

    def test_load_empty_session_id_returns_none(self, hermes_home):
        from hermes_cli.tasks import load_todo

        assert load_todo("") is None

    def test_corrupt_row_returns_none(self, hermes_home):
        from hermes_cli import tasks
        from hermes_cli.tasks import _get_session_db

        db = _get_session_db()
        assert db is not None
        db.set_meta("todo:sess-bad", "{{{ not json")
        assert tasks.load_todo("sess-bad") is None

    def test_save_is_noop_without_db(self, hermes_home, monkeypatch):
        from hermes_cli import tasks

        monkeypatch.setattr(tasks, "_get_session_db", lambda: None)
        # Must not raise.
        tasks.save_todo("sess-1", object())


class TestClearTodo:
    def test_clear_removes_row(self, hermes_home):
        from hermes_cli.tasks import clear_todo, load_todo, save_todo
        from tools.todo_tool import TodoStore

        save_todo("sess-1", TodoStore())
        assert clear_todo("sess-1") is True
        assert load_todo("sess-1") is None

    def test_clear_missing_row_returns_false(self, hermes_home):
        from hermes_cli.tasks import clear_todo

        assert clear_todo("missing") is False

    def test_clear_empty_session_id_returns_false(self, hermes_home):
        from hermes_cli.tasks import clear_todo

        assert clear_todo("") is False


class TestMigrateTodoToSession:
    def test_migrates_active_store_to_child(self, hermes_home):
        from hermes_cli.tasks import load_todo, migrate_todo_to_session, save_todo
        from tools.todo_tool import TodoStore

        store = TodoStore()
        store.write([{"id": "1", "content": "Ship the feature", "status": "in_progress"}])
        save_todo("parent-sid", store)

        assert migrate_todo_to_session("parent-sid", "child-sid", reason="compression") is True
        child = load_todo("child-sid")
        assert child is not None and child.read()[0]["content"] == "Ship the feature"
        # Parent row removed so only the child is active.
        assert load_todo("parent-sid") is None

    def test_no_store_to_migrate_returns_false(self, hermes_home):
        from hermes_cli.tasks import migrate_todo_to_session

        assert migrate_todo_to_session("empty-parent", "child2") is False

    def test_does_not_clobber_existing_child_store(self, hermes_home):
        from hermes_cli.tasks import load_todo, migrate_todo_to_session, save_todo
        from tools.todo_tool import TodoStore

        store = TodoStore()
        store.write([{"id": "1", "content": "parent", "status": "pending"}])
        save_todo("p3", store)
        child = TodoStore()
        child.write([{"id": "9", "content": "child has own", "status": "pending"}])
        save_todo("c3", child)

        assert migrate_todo_to_session("p3", "c3") is False
        child = load_todo("c3")
        assert child is not None
        assert child.read()[0]["content"] == "child has own"

    def test_same_id_is_noop(self, hermes_home):
        from hermes_cli.tasks import migrate_todo_to_session, save_todo
        from tools.todo_tool import TodoStore

        save_todo("same", TodoStore())
        assert migrate_todo_to_session("same", "same") is False

    def test_empty_store_migration_drops_stale_row(self, hermes_home):
        from hermes_cli.tasks import load_todo, migrate_todo_to_session, save_todo
        from tools.todo_tool import TodoStore

        save_todo("empty-parent", TodoStore())
        assert migrate_todo_to_session("empty-parent", "child4") is False
        assert load_todo("empty-parent") is None

    def test_migrates_captures_too(self, hermes_home):
        from hermes_cli.tasks import load_todo, migrate_todo_to_session, save_todo
        from tools.todo_tool import TodoStore

        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        store.capture_request("pending request")
        save_todo("p5", store)

        assert migrate_todo_to_session("p5", "c5") is True
        child = load_todo("c5")
        assert child is not None
        assert len(child.pending_captures()) == 1


class TestPersistTodoStoreHelper:
    def test_writes_through_for_agent(self, hermes_home):
        from hermes_cli.tasks import load_todo, persist_todo_store

        agent = _StubAgent(session_id="sess-a")
        assert agent._todo_store is not None
        agent._todo_store.write([{"id": "1", "content": "task", "status": "pending"}])

        persist_todo_store(agent)
        loaded = load_todo("sess-a")
        assert loaded is not None
        assert loaded.read()[0]["content"] == "task"

    def test_respects_persist_disabled(self, hermes_home):
        from hermes_cli.tasks import load_todo, persist_todo_store

        agent = _StubAgent(session_id="sess-fork")
        agent._persist_disabled = True
        assert agent._todo_store is not None
        agent._todo_store.write([{"id": "1", "content": "task", "status": "pending"}])

        persist_todo_store(agent)
        assert load_todo("sess-fork") is None

    def test_skips_agents_without_session(self, hermes_home):
        from hermes_cli.tasks import persist_todo_store

        agent = _StubAgent(session_id=None)
        persist_todo_store(agent)  # must not raise

    def test_skips_agents_without_store(self, hermes_home):
        from hermes_cli.tasks import persist_todo_store

        agent = _StubAgent(session_id="sess-x")
        agent._todo_store = None
        persist_todo_store(agent)  # must not raise
