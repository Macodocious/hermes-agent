"""The start line is recorded by the transition that opens work.

Every entry into ``in_progress`` — begin or resume — records the start line
inside that call. A non-todo call no longer records it: the line that claims
work started arrives with the call that started it, not after the work. A
close promotes nothing, so the next task starts only by the agent's begin.
"""

from tools.todo_tool import NOTICE_STARTED, TodoStore, todo_tool


def _store(*contents):
    store = TodoStore()
    store.write([
        {"id": str(index), "content": text, "status": "pending"}
        for index, text in enumerate(contents, start=1)
    ])
    store.drain_notices()
    return store


def _starts(store):
    return [n for n in store.pending_notices() if n["kind"] == NOTICE_STARTED]


class TestTransitionStart:
    def test_begin_records_the_start_line_in_the_call(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert [(n["kind"], n["id"]) for n in _starts(store)] == [
            (NOTICE_STARTED, "1")
        ]

    def test_resume_records_a_second_start_line(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("pause", "1")["ok"]
        assert store.transition("resume", "1")["ok"]
        assert len(_starts(store)) == 2

    def test_mark_work_records_nothing(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.drain_notices()
        assert store.mark_work() is False
        assert _starts(store) == []


class TestListWriteStart:
    def test_head_row_write_makes_it_current_without_a_start_line(self):
        store = _store("one", "two")
        result = todo_tool(
            todos=[{"id": "1", "content": "one", "status": "in_progress"}],
            merge=True,
            store=store,
        )
        assert '"in_progress"' in result
        assert store.read()[0]["status"] == "in_progress"
        assert _starts(store) == []

    def test_a_later_row_cannot_be_started_by_a_write(self):
        store = _store("one", "two")
        result = todo_tool(
            todos=[{"id": "2", "content": "two", "status": "in_progress"}],
            merge=True,
            store=store,
        )
        assert "cannot write status 'in_progress'" in result
        assert store.read()[1]["status"] == "pending"

    def test_a_write_cannot_start_a_row_while_another_is_current(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        result = todo_tool(
            todos=[{"id": "2", "content": "two", "status": "in_progress"}],
            merge=True,
            store=store,
        )
        assert "cannot write status 'in_progress'" in result


class TestCloseDoesNotStart:
    def test_completing_leaves_the_next_row_pending(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert store.transition("complete", "1")["ok"]
        assert store.read()[1]["status"] == "pending"
        store.drain_notices()
        assert _starts(store) == []

    def test_cancelling_leaves_the_next_row_pending(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("cancel", "1", "not needed")["ok"]
        assert store.read()[1]["status"] == "pending"
