"""Unit tests for the store's commit-derived lifecycle notices.

Every mutation commits through one lock-guarded boundary that diffs the
ordered list before and after the change, so the notice queue records what
actually happened rather than what the caller asked for. Ordering is a
property of that serialization — nothing sorts the queue afterwards.
"""

import threading

from tools.todo_tool import NOTICE_COMPLETED, NOTICE_STARTED, TodoStore


def _store(*statuses):
    """A store holding one row per status, with the write's notices drained."""
    store = TodoStore()
    store.write([
        {"id": str(index), "content": f"task {index}", "status": status}
        for index, status in enumerate(statuses, start=1)
    ])
    store.drain_notices()
    return store


class TestCommitOrdering:
    def test_completion_precedes_the_start_it_unlocks(self):
        store = _store("pending", "pending", "pending")
        store.transition("begin", "1")
        store.mark_work()
        store.drain_notices()

        store.transition("complete", "1")

        assert [(n["kind"], n["id"]) for n in store.drain_notices()] == [
            (NOTICE_COMPLETED, "1"),
            (NOTICE_STARTED, "2"),
        ]

    def test_write_path_records_a_start(self):
        store = TodoStore()

        store.write([{"id": "1", "content": "task 1", "status": "in_progress"}])

        assert [(n["kind"], n["id"]) for n in store.drain_notices()] == [
            (NOTICE_STARTED, "1"),
        ]

    def test_replace_mode_renumber_does_not_announce_a_start(self):
        store = _store("pending", "in_progress")

        # Same two rows, reordered: the in_progress row's id moves 2 -> 1.
        store.write([
            {"id": "1", "content": "task 2", "status": "in_progress"},
            {"id": "2", "content": "task 1", "status": "pending"},
        ])

        assert store.drain_notices() == []

    def test_idempotent_begin_records_one_notice(self):
        store = _store("pending")

        store.transition("begin", "1")
        store.transition("begin", "1")

        assert len(store.drain_notices()) == 1

    def test_refused_transition_records_nothing(self):
        store = _store("pending", "pending")
        store.transition("begin", "1")
        store.drain_notices()

        result = store.transition("begin", "2")

        assert result["ok"] is False
        assert store.drain_notices() == []


class TestConcurrentMutation:
    def test_two_threads_beginning_at_once_serialize(self):
        store = _store("pending", "pending")
        barrier = threading.Barrier(2)
        results = []

        def begin(item_id):
            barrier.wait()
            results.append(store.transition("begin", item_id))

        threads = [
            threading.Thread(target=begin, args=(item_id,))
            for item_id in ("1", "2")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sum(1 for result in results if result["ok"]) == 1
        assert len(store.drain_notices()) == 1
        assert [i["status"] for i in store.read()].count("in_progress") == 1


class TestDetectorsRemoved:
    def test_request_inspecting_detectors_are_gone(self):
        store = TodoStore()

        assert not hasattr(store, "detect_task_start")
        assert not hasattr(store, "detect_task_stop")
