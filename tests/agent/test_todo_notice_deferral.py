"""A refused lifecycle notice is deferred, never destroyed.

The delivery callback refuses a line when the run that produced it has been
superseded. The emitter must then keep that notice — and everything behind it
— queued for the turn-end drain, instead of clearing the queue behind the
refusal.
"""

from agent.task_manager import _emit_pending_notices
from tools.todo_tool import TodoStore


def _store_with(items):
    store = TodoStore()
    store.write(items)
    return store


def _store_with_two_notices():
    """One start and one completion, in queue order."""
    store = _store_with([
        {"id": "1", "content": "one", "status": "pending"},
        {"id": "2", "content": "two", "status": "pending"},
    ])
    store.transition("begin", "1")
    store.mark_work()
    store.transition("complete", "1")
    return store


class _Agent:
    def __init__(self, store, callback):
        self._todo_store = store
        self.todo_notice_callback = callback


class TestRefusedNoticeIsDeferred:
    def test_a_refused_line_stays_queued(self):
        store = _store_with_two_notices()

        _emit_pending_notices(_Agent(store, lambda line: False))

        assert [n["kind"] for n in store.pending_notices()] == [
            "started",
            "completed",
        ]

    def test_a_refusal_stops_the_emit_and_keeps_the_rest(self):
        store = _store_with_two_notices()
        sent = []

        def refuse_the_second(line):
            sent.append(line)
            return len(sent) < 2

        _emit_pending_notices(_Agent(store, refuse_the_second))

        assert len(sent) == 2
        # The first line shipped, so only it is drained; the second waits.
        assert [n["kind"] for n in store.pending_notices()] == ["completed"]

    def test_a_callback_returning_nothing_counts_as_accepted(self):
        store = _store_with_two_notices()
        sent = []

        _emit_pending_notices(_Agent(store, sent.append))

        assert len(sent) == 2
        assert store.pending_notices() == []

    def test_a_raising_callback_keeps_the_undelivered_rest(self):
        store = _store_with_two_notices()
        sent = []

        def raise_on_the_second(line):
            sent.append(line)
            if len(sent) == 2:
                raise RuntimeError("delivery blew up")

        _emit_pending_notices(_Agent(store, raise_on_the_second))

        assert len(sent) == 2
        assert [n["kind"] for n in store.pending_notices()] == ["completed"]


class TestCountedDrain:
    def test_a_counted_drain_clears_only_the_head(self):
        store = _store_with_two_notices()

        drained = store.drain_notices(1)

        assert [n["kind"] for n in drained] == ["started"]
        assert [n["kind"] for n in store.pending_notices()] == ["completed"]

    def test_a_counted_drain_past_the_end_clears_everything(self):
        store = _store_with_two_notices()

        assert len(store.drain_notices(5)) == 2
        assert store.pending_notices() == []

    def test_an_uncounted_drain_still_clears_everything(self):
        store = _store_with_two_notices()

        assert len(store.drain_notices()) == 2
        assert store.pending_notices() == []
