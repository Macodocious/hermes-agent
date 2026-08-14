"""Tests for the single-current-task and sequential-id invariants.

The store enforces two invariants at the data layer:

- Single current task: at most one item may be ``in_progress``. The kept
  item is the task the write started (the same transition
  ``detect_task_start`` detects); otherwise the first ``in_progress`` in
  list order (list order is priority). Any others demote to ``pending``.
  Two ``[>]`` rows are impossible.
- Sequential ids: replace-mode writes and the load path renumber to
  ``1..N`` in list order; merge-mode writes keep existing ids stable and
  assign new items the next integer. Model-chosen ids (``t5``,
  ``call-1``, ...) never leak into the rendered list.
"""

from tools.todo_tool import TodoStore


class TestSingleInProgress:
    def test_two_in_progress_demotes_second(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "first", "status": "in_progress"},
            {"id": "2", "content": "second", "status": "in_progress"},
        ])
        items = store.read()
        assert [i["status"] for i in items] == ["in_progress", "pending"]

    def test_first_in_progress_wins_by_list_order(self):
        store = TodoStore()
        store.write([
            {"id": "a", "content": "alpha", "status": "in_progress"},
            {"id": "b", "content": "beta", "status": "in_progress"},
            {"id": "c", "content": "gamma", "status": "in_progress"},
        ])
        items = store.read()
        assert [i["status"] for i in items] == ["in_progress", "pending", "pending"]

    def test_merge_write_that_starts_new_task_demotes_old(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "old task", "status": "in_progress"},
            {"id": "2", "content": "waiting", "status": "pending"},
        ])
        # Model switches to the other task via merge.
        store.write([{"id": "2", "status": "in_progress"}], merge=True)
        items = store.read()
        assert [i["status"] for i in items] == ["pending", "in_progress"]

    def test_merge_new_item_started_current_demotes_old(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "old task", "status": "in_progress"},
            {"id": "2", "content": "waiting", "status": "pending"},
        ])
        # Model starts a brand-new task via merge.
        store.write(
            [{"id": "zzz", "content": "new task", "status": "in_progress"}],
            merge=True,
        )
        items = store.read()
        assert [i["status"] for i in items] == ["pending", "pending", "in_progress"]
        assert items[2]["content"] == "new task"

    def test_single_in_progress_unchanged(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "current", "status": "in_progress"},
            {"id": "2", "content": "next", "status": "pending"},
        ])
        items = store.read()
        assert [i["status"] for i in items] == ["in_progress", "pending"]

    def test_completed_and_cancelled_untouched(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "done", "status": "completed"},
            {"id": "2", "content": "cancelled", "status": "cancelled"},
            {"id": "3", "content": "current", "status": "in_progress"},
        ])
        items = store.read()
        assert [i["status"] for i in items] == ["completed", "cancelled", "in_progress"]

    def test_polluted_store_heals_on_load(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "first", "status": "in_progress"},
            {"id": "2", "content": "second", "status": "in_progress"},
        ])
        restored = TodoStore.from_json(store.to_json())
        items = restored.read()
        assert [i["status"] for i in items] == ["in_progress", "pending"]


class TestSequentialIds:
    def test_replace_renumbers_in_list_order(self):
        store = TodoStore()
        store.write([
            {"id": "t5", "content": "task five", "status": "in_progress"},
            {"id": "call-1", "content": "other", "status": "pending"},
        ])
        items = store.read()
        assert [i["id"] for i in items] == ["1", "2"]

    def test_merge_keeps_existing_ids_stable(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "task one", "status": "in_progress"},
            {"id": "2", "content": "task two", "status": "pending"},
        ])
        store.write([{"id": "2", "status": "completed"}], merge=True)
        items = store.read()
        assert [i["id"] for i in items] == ["1", "2"]
        assert items[1]["status"] == "completed"

    def test_merge_assigns_sequential_id_to_new_item(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "first", "status": "in_progress"},
            {"id": "2", "content": "second", "status": "pending"},
        ])
        store.write([{"id": "zzz", "content": "third", "status": "pending"}], merge=True)
        items = store.read()
        assert [i["id"] for i in items] == ["1", "2", "3"]
        assert items[2]["content"] == "third"

    def test_renumbered_ids_round_trip(self):
        store = TodoStore()
        store.write([
            {"id": "t5", "content": "task five", "status": "in_progress"},
            {"id": "call-1", "content": "other", "status": "pending"},
        ])
        restored = TodoStore.from_json(store.to_json())
        assert [i["id"] for i in restored.read()] == ["1", "2"]

    def test_load_renumbers_polluted_persisted_ids(self):
        store = TodoStore.from_json(
            '{"items": ['
            '{"id": "t5", "content": "five", "status": "in_progress"}, '
            '{"id": "call-1", "content": "ex", "status": "pending"}'
            "]}"
        )
        assert [i["id"] for i in store.read()] == ["1", "2"]

    def test_sequential_ids_after_replace_with_arbitrary_ids(self):
        store = TodoStore()
        store.write([
            {"id": "9", "content": "nine", "status": "pending"},
            {"id": "x", "content": "ex", "status": "in_progress"},
            {"id": "t5", "content": "five", "status": "pending"},
        ])
        items = store.read()
        assert [i["id"] for i in items] == ["1", "2", "3"]
        assert [i["content"] for i in items] == ["nine", "ex", "five"]
