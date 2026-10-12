"""The store is the single producer of a task's start line.

The task card and the CLI spinner used to compose ``Working on <task>`` for a
todo call while the store recorded its own notice for the same transition —
two producers, one event. Neither surface composes a lifecycle line now: the
store's notice is the only one, rendered by ``tools.todo_tool.format_notice``.
"""

from tools.todo_tool import NOTICE_STARTED, TodoStore, format_notice


def _store_with(items):
    store = TodoStore()
    store.write(items)
    return store


class TestStoreIsTheStartProducer:
    def test_begin_records_one_start_notice(self):
        store = _store_with([{"id": "1", "content": "task one", "status": "pending"}])
        assert store.transition("begin", "1")["ok"] is True
        notices = [n for n in store.pending_notices() if n["kind"] == NOTICE_STARTED]
        assert len(notices) == 1
        assert notices[0]["content"] == "task one"

    def test_start_line_names_the_ordered_position(self):
        store = _store_with([
            {"id": "1", "content": "done one", "status": "completed"},
            {"id": "2", "content": "done two", "status": "completed"},
            {"id": "3", "content": "ship the feature", "status": "pending"},
        ])
        assert store.transition("begin", "3")["ok"] is True
        notice = [n for n in store.pending_notices() if n["kind"] == NOTICE_STARTED][-1]
        assert format_notice(notice, store) == (
            "▶️ Working on task 3 of 3: ship the feature"
        )

    def test_start_line_falls_back_without_a_position(self):
        # A notice whose row has left the store still renders, bare.
        store = _store_with([{"id": "1", "content": "task one", "status": "pending"}])
        stray = {"kind": NOTICE_STARTED, "id": "gone", "content": "task one"}
        assert format_notice(stray, store) == "▶️ Working on: task one"


class TestSpinnerComposesNoLifecycleLine:
    """A todo call shows its own preview, never a lifecycle line."""

    def _make_cli(self):
        import importlib
        mod = importlib.import_module("cli")
        return mod.HermesCLI(verbose=False), mod

    def test_started_task_kwarg_does_not_change_the_label(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Reading the task list", {},
            started_task={"id": "1", "content": "ship the feature", "status": "in_progress"},
        )
        assert cli._spinner_text == "📋 Reading the task list"
