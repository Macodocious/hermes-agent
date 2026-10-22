"""The start line is recorded at the first real call under a row.

A row becoming current is not the work starting. Every path that makes a row
current — a list write that starts the head row, begin, resume, and a
promotion — arms the start line; the first non-todo call under the row records
it. A row that becomes current but never sees a call gets no start line.
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

    def test_first_call_records_the_start_line(self):
        store = _store("one", "two")
        todo_tool(
            todos=[{"id": "1", "content": "one", "status": "in_progress"}],
            merge=True,
            store=store,
        )
        assert store.mark_work() is True
        assert [(n["kind"], n["id"]) for n in _starts(store)] == [
            (NOTICE_STARTED, "1")
        ]

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


class TestEveryStartPathWaits:
    def test_begin_alone_records_nothing(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert _starts(store) == []

    def test_resume_alone_records_nothing(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("pause", "1")["ok"]
        store.drain_notices()
        assert store.transition("resume", "1")["ok"]
        assert _starts(store) == []

    def test_promotion_alone_records_nothing(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        assert _starts(store) == []

    def test_a_row_that_never_sees_a_call_gets_no_line(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        # Row 2 is current but no call has run under it.
        assert store.read()[1]["status"] == "in_progress"
        assert _starts(store) == []


class TestOneLinePerTenure:
    def test_a_second_call_records_nothing(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        assert store.mark_work() is True
        assert store.mark_work() is False
        assert len(_starts(store)) == 1

    def test_a_new_tenure_records_a_new_line(self):
        store = _store("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        assert store.mark_work() is True
        assert [(n["kind"], n["id"]) for n in _starts(store)] == [
            (NOTICE_STARTED, "2")
        ]

    def test_the_start_mark_survives_a_reload(self):
        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        restored = TodoStore.from_json(store.to_json())
        assert restored.mark_work() is False
        assert _starts(restored) == []

    def test_the_start_mark_crosses_the_merge(self):
        from hermes_cli.tasks import _merge_todo_stores

        store = _store("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        merged = _merge_todo_stores(store, TodoStore.from_json(store.to_json()))
        assert merged.mark_work() is False
        assert _starts(merged) == []
