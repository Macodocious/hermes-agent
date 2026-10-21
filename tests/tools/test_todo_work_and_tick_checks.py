"""Tests for the two close checks: work under the row, one close per message.

A ``complete`` is the only signal the store has that a task is finished, so a
close the lifecycle did not see happen is refused:

- the row must carry a work mark, set by the executor when a call other than
  ``todo`` runs while the row is current;
- one assistant message may close one task, so a list cannot be ticked off in
  a single batch.
"""

import json

from tools.todo_tool import TodoStore, todo_tool


def _store_with(*contents: str) -> TodoStore:
    store = TodoStore()
    store.write([
        {"id": str(index), "content": text, "status": "pending"}
        for index, text in enumerate(contents, start=1)
    ])
    return store


def _close(store: TodoStore, item_id: str, request_id: str = "turn:api:1") -> dict:
    return store.transition("complete", item_id, "", request_id=request_id)


class TestWorkUnderTheRow:
    def test_close_without_work_is_refused(self):
        store = _store_with("one")
        assert store.transition("begin", "1")["ok"]
        result = _close(store, "1")
        assert result["ok"] is False
        assert "no work under it" in result["error"]
        assert store.read()[0]["status"] == "in_progress"

    def test_close_after_work_passes(self):
        store = _store_with("one")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1")["ok"] is True
        assert store.read()[0]["status"] == "completed"

    def test_work_before_begin_is_not_credited(self):
        store = _store_with("one")
        store.mark_work()  # no row is current, so nothing is recorded
        assert store.transition("begin", "1")["ok"]
        assert _close(store, "1")["ok"] is False

    def test_promoted_row_needs_its_own_work(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1")["ok"]
        # Task 2 is current now; work under task 1 does not carry over.
        assert _close(store, "2", request_id="turn:api:2")["ok"] is False
        store.mark_work()
        assert _close(store, "2", request_id="turn:api:2")["ok"]


class TestOneClosePerMessage:
    def test_second_close_in_same_message_is_refused(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1", request_id="m1")["ok"]
        store.mark_work()
        result = _close(store, "2", request_id="m1")
        assert result["ok"] is False
        assert "already closed in this message" in result["error"]
        assert store.read()[1]["status"] == "in_progress"

    def test_close_in_later_message_passes(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1", request_id="m1")["ok"]
        store.mark_work()
        assert _close(store, "2", request_id="m2")["ok"]

    def test_cancel_and_pause_need_no_work(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        assert store.transition("pause", "1")["ok"]
        assert store.transition("cancel", "2", "not needed")["ok"]


class TestPersistence:
    def test_work_mark_and_last_close_survive_reload(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1", request_id="m1")["ok"]
        store.mark_work()  # work under row 2, not yet through a close
        restored = TodoStore.from_json(store.to_json())
        # The last close's message id survived: a second close in it is refused.
        assert _close(restored, "2", request_id="m1")["ok"] is False
        # The work mark survived too: the close passes in a new message.
        assert _close(restored, "2", request_id="m2")["ok"]


class TestMergeCarry:
    """The persisted store replaces the in-memory one after every call.

    That merge reconciles items; the close state has to cross it too, or a
    second close in the same message is forgiven right after the first.
    """

    def test_merge_keeps_the_close_state(self):
        from hermes_cli.tasks import _merge_todo_stores

        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        assert _close(store, "1", request_id="m1")["ok"]
        store.mark_work()  # work under row 2, recorded before the merge

        merged = _merge_todo_stores(store, TodoStore.from_json(store.to_json()))

        result = _close(merged, "2", request_id="m1")
        assert result["ok"] is False
        assert "already closed in this message" in result["error"]


class TestToolEntry:
    def test_tool_entry_refuses_second_close_in_one_message(self):
        store = _store_with("one", "two")
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        first = json.loads(
            todo_tool(action="complete", item_id="1", store=store, request_id="m1")
        )
        assert first["summary"]["completed"] == 1
        store.mark_work()
        refused = todo_tool(action="complete", item_id="2", store=store, request_id="m1")
        assert "already closed in this message" in refused
