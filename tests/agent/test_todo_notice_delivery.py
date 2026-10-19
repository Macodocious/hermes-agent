"""The notice queue has one consumer, and it awaits each send.

The store records notices in commit order. These tests pin the delivery half:
one line at a time, the next issued only after the previous resolved, and a
failed send leaving the queue intact from that line onward.
"""

import logging

from agent import task_manager
from tools.todo_tool import TodoStore, format_notice


class _Send:
    """A scheduled send: records when it resolves, or fails on resolve."""

    def __init__(self, events, line, fail=False):
        self._events = events
        self._line = line
        self._fail = fail

    def result(self):
        if self._fail:
            raise RuntimeError("platform rejected the message")
        self._events.append(("resolved", self._line))


def _agent(store, callback):
    """A stand-in agent. The callback is set on the instance, not the class:
    a class attribute would bind as a method and receive ``self``."""
    agent = type("_Agent", (), {})()
    agent._todo_store = store
    agent.todo_notice_callback = callback
    return agent


def _two_task_store():
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "pending"},
        ]
    )
    return store


def test_completion_resolves_before_the_promotion_is_issued():
    """One complete produces two lines; the second waits for the first."""
    store = _two_task_store()
    store.transition("begin", "1")
    store.drain_notices()
    events = []

    def callback(line):
        events.append(("issued", line))
        return _Send(events, line)

    store.transition("complete", "1")
    task_manager._emit_pending_notices(_agent(store, callback))

    assert [kind for kind, _ in events] == ["issued", "resolved", "issued", "resolved"]
    assert "omplet" in events[0][1]
    assert "Working on" in events[2][1]
    assert store.pending_notices() == []


def test_a_failed_send_stops_the_drain_and_keeps_the_queue():
    """The failed line and the one behind it stay queued; nothing is skipped."""
    store = _two_task_store()
    store.transition("begin", "1")
    store.drain_notices()
    issued = []

    def callback(line):
        issued.append(line)
        return _Send([], line, fail=True)

    store.transition("complete", "1")
    task_manager._emit_pending_notices(_agent(store, callback))

    assert len(issued) == 1
    assert len(store.pending_notices()) == 2


def test_no_consumer_is_reported_and_nothing_is_dropped(caplog):
    """A notice with nowhere to go is a fault, not a queue to ship later."""
    store = _two_task_store()
    store.transition("begin", "1")

    with caplog.at_level(logging.WARNING, logger="agent.task_manager"):
        task_manager._emit_pending_notices(_agent(store, None))

    assert "no consumer" in caplog.text
    assert len(store.pending_notices()) == 1


def test_a_synchronous_callback_delivers_and_clears():
    """The CLI prints before it returns, so True is a real delivery."""
    store = _two_task_store()
    store.transition("begin", "1")
    lines = []

    task_manager._emit_pending_notices(
        _agent(store, lambda line: lines.append(line) or True)
    )

    assert len(lines) == 1
    assert store.pending_notices() == []


def test_a_callback_returning_none_counts_as_delivered():
    """A plain sink (list.append) has no failure channel, so it is taken."""
    store = _two_task_store()
    store.transition("begin", "1")
    lines = []

    task_manager._emit_pending_notices(_agent(store, lines.append))

    assert len(lines) == 1
    assert store.pending_notices() == []


def test_a_refused_line_stays_queued():
    """False means the line was not handed over, so it is not spent."""
    store = _two_task_store()
    store.transition("begin", "1")

    task_manager._emit_pending_notices(_agent(store, lambda line: False))

    assert len(store.pending_notices()) == 1


def test_the_label_is_frozen_at_commit():
    """A later rewrite of the list cannot change what a notice describes."""
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "First", "status": "completed", "plan": "p.md"},
            {"id": "2", "content": "Second", "status": "in_progress", "plan": "p.md"},
            {"id": "3", "content": "Third", "status": "pending", "plan": "p.md"},
        ]
    )
    store.drain_notices()
    store.transition("complete", "2")
    notice = next(n for n in store.pending_notices() if n["kind"] == "completed")
    rendered = format_notice(notice, store)

    # Drop a row and renumber: the completed row is now first of two.
    store.write(
        [
            {"id": "9", "content": "Second", "status": "completed", "plan": "p.md"},
            {"id": "8", "content": "Third", "status": "in_progress", "plan": "p.md"},
        ]
    )

    assert notice["position"] == 2
    assert notice["total"] == 3
    assert format_notice(notice, store) == rendered
    assert "Task 2 of 3" in rendered
