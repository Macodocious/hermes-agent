"""The store is the single producer of a task's cancellation line.

The task card and the CLI spinner used to compose ``Task N of M cancelled``
for the same transition the store recorded a notice for. Neither surface
composes a lifecycle line now. The store's notice carries the cancelled task's
ordered position and the reason it was cancelled.
"""

from tools.todo_tool import NOTICE_CANCELLED, TodoStore, format_notice


def _store_with(items):
    store = TodoStore()
    store.write(items)
    return store


class TestStoreIsTheCancelProducer:
    def test_cancel_records_one_notice_with_position_and_reason(self):
        store = _store_with([
            {"id": "1", "content": "done one", "status": "completed"},
            {"id": "2", "content": "done two", "status": "completed"},
            {"id": "3", "content": "ship the feature", "status": "in_progress"},
        ])
        assert store.transition("cancel", "3", "superseded")["ok"] is True
        notices = [n for n in store.pending_notices() if n["kind"] == NOTICE_CANCELLED]
        assert len(notices) == 1
        assert format_notice(notices[0], store) == (
            "🚫 Cancelled task 3 of 3: ship the feature\n  reason: superseded"
        )

    def test_cancel_without_a_reason_is_refused_and_records_nothing(self):
        store = _store_with([{"id": "1", "content": "task one", "status": "in_progress"}])
        assert store.transition("cancel", "1")["ok"] is False
        assert [n for n in store.pending_notices() if n["kind"] == NOTICE_CANCELLED] == []

    def test_cancel_line_falls_back_without_a_position(self):
        store = _store_with([{"id": "1", "content": "task one", "status": "in_progress"}])
        stale = {
            "kind": NOTICE_CANCELLED,
            "id": "gone",
            "content": "task one",
            "reason": "superseded",
        }
        assert format_notice(stale, store) == (
            "🚫 Cancelled: task one\n  reason: superseded"
        )


class TestSpinnerComposesNoLifecycleLine:
    """A todo call shows its own preview, never a lifecycle line."""

    def _make_cli(self):
        import importlib
        mod = importlib.import_module("cli")
        return mod.HermesCLI(verbose=False), mod

    def test_stopped_task_kwarg_does_not_change_the_label(self):
        cli, _ = self._make_cli()
        cli._on_tool_progress(
            "tool.started", "todo", "Updating 1 task",
            {"todos": [{"id": "1", "content": "ship the feature", "status": "cancelled"}]},
            stopped_task={"id": "1", "content": "ship the feature", "status": "cancelled"},
        )
        assert cli._spinner_text == "📋 Updating 1 task"
