"""The final completion line is held to the turn's end.

A close that promotes a successor records its completion line immediately. A
close with no successor is the turn's final completion: its line is held in
the queue, the drain skips it while the lines behind it ship, and the turn
finalizer releases it.
"""

from agent import task_manager
from tools.todo_tool import NOTICE_COMPLETED, NOTICE_STARTED, TodoStore


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


class TestHoldDecision:
    def test_a_close_with_a_successor_is_not_held(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        notice = next(
            n for n in store.pending_notices() if n["kind"] == NOTICE_COMPLETED
        )
        assert notice["held"] is False

    def test_a_close_with_no_successor_is_held(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        notice = next(
            n for n in store.pending_notices() if n["kind"] == NOTICE_COMPLETED
        )
        assert notice["held"] is True

    def test_a_cancellation_is_never_held(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("cancel", "1", "not needed")["ok"]
        notice = next(
            n for n in store.pending_notices() if n["kind"] == "cancelled"
        )
        assert notice["held"] is False


class TestDrainSkipsTheHeldLine:
    def test_the_lines_behind_the_held_line_ship(self):
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
        assert [n["kind"] for n in store.pending_notices()] == [NOTICE_COMPLETED]

    def test_a_delivery_failure_leaves_the_held_line_and_the_rest(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        assert store.mark_work()
        assert store.transition("complete", "2")["ok"]
        sent = []

        def refuse(line):
            sent.append(line)
            return False

        task_manager._emit_pending_notices(_agent(store, refuse))

        assert len(sent) == 1
        assert [n["kind"] for n in store.pending_notices()] == [
            NOTICE_COMPLETED,
            NOTICE_STARTED,
            NOTICE_COMPLETED,
        ]


class TestRelease:
    def test_release_ships_the_held_line(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        sent = []

        task_manager.release_held_notices(_agent(store, sent.append))

        assert sent == ["✅ Task completed: one"]
        assert store.pending_notices() == []

    def test_release_ships_every_held_line_in_order(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        store.write(
            [{"id": "2", "content": "two", "status": "pending"}], merge=True
        )
        assert store.transition("begin", "2")["ok"]
        store.mark_work()
        assert store.transition("complete", "2")["ok"]
        assert len([n for n in store.pending_notices() if n["held"]]) == 2
        sent = []

        task_manager.release_held_notices(_agent(store, sent.append))

        assert sent == [
            "✅ Task completed: one",
            "▶️ Working on: two",
            "✅ Task completed: two",
        ]
