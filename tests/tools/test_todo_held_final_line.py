"""A close that leaves no active task emits no completion line.

A close that promotes a successor records its completion line immediately. A
close with no successor is the turn's final completion: the user's rule is
that once the task is completed it is final, and with no active tasks nothing
is emitted.
"""

from agent import task_manager
from tools.todo_tool import NOTICE_COMPLETED, TodoStore


def _store(*contents):
    store = TodoStore()
    store.write([
        {"id": str(index), "content": text, "status": "pending"}
        for index, text in enumerate(contents, start=1)
    ])
    store.drain_notices()
    return store


def _agent(store, callback):
    agent = type("_Agent", (), {})()
    agent._todo_store = store
    agent.todo_notice_callback = callback
    return agent


class TestFinalClose:
    def test_a_close_with_a_successor_records_its_line(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        notice = next(
            n for n in store.pending_notices() if n["kind"] == NOTICE_COMPLETED
        )
        assert notice["held"] is False

    def test_a_close_with_no_successor_records_nothing(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        assert [
            n for n in store.pending_notices() if n["kind"] == NOTICE_COMPLETED
        ] == []

    def test_a_cancellation_is_recorded(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("cancel", "1", "not needed")["ok"]
        notice = next(
            n for n in store.pending_notices() if n["kind"] == "cancelled"
        )
        assert notice["held"] is False


class TestDrain:
    def test_the_successor_close_ships_and_the_final_close_is_silent(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        assert store.mark_work()
        assert store.transition("complete", "2")["ok"]
        sent = []

        task_manager._emit_pending_notices(_agent(store, sent.append))

        assert sent == [
            "✅ Task completed: one",
            "▶️ Working on: two",
        ]
        assert store.pending_notices() == []


class TestRelease:
    def test_release_sends_nothing_after_a_final_close(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        sent = []

        task_manager.release_held_notices(_agent(store, sent.append))

        assert sent == []
        assert store.pending_notices() == []
