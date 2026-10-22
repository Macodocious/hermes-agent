"""The held line's label is frozen at the close, and the cap spares it.

A held line is rendered at the turn's end, after the close. The plan decision
and the count it prints are frozen at the close, so a later list write, a
merge, or a reload cannot rewrite the line the turn promised. The queue cap
drops the oldest line that is not held.
"""

import logging

from tools.todo_tool import (
    MAX_TODO_NOTICES,
    NOTICE_COMPLETED,
    TodoStore,
    format_notice,
)


def _plan_store():
    store = TodoStore()
    store.write([
        {"id": "1", "content": "a", "status": "pending", "plan": "p.md"},
        {"id": "2", "content": "b", "status": "pending", "plan": "p.md"},
    ])
    store.drain_notices()
    return store


def _close_final(store):
    assert store.transition("begin", "1")["ok"]
    store.mark_work()
    store.drain_notices()
    assert store.transition("complete", "1")["ok"]
    assert store.mark_work()
    assert store.transition("complete", "2")["ok"]
    return next(
        n
        for n in store.pending_notices()
        if n["kind"] == NOTICE_COMPLETED and n["held"]
    )


class TestFrozenLabel:
    def test_the_plan_decision_and_count_are_frozen(self):
        store = _plan_store()
        notice = _close_final(store)

        assert notice["plan_complete"] is True
        assert notice["plan_total"] == 2
        assert format_notice(notice, store) == (
            "🏁 Plan complete: all 2 tasks done (p.md)"
        )

    def test_a_later_write_cannot_rewrite_the_line(self):
        store = _plan_store()
        notice = _close_final(store)
        rendered = format_notice(notice, store)

        store.write(
            [{"id": "3", "content": "c", "status": "pending", "plan": "p.md"}],
            merge=True,
        )

        assert format_notice(notice, store) == rendered

    def test_a_round_trip_renders_the_same_line(self):
        store = _plan_store()
        notice = _close_final(store)
        rendered = format_notice(notice, store)

        restored = TodoStore.from_json(store.to_json())
        restored_notice = next(
            n
            for n in restored.pending_notices()
            if n["kind"] == NOTICE_COMPLETED and n["held"]
        )

        assert format_notice(restored_notice, restored) == rendered

    def test_a_merge_renders_the_same_line(self):
        from hermes_cli.tasks import _merge_todo_stores

        store = _plan_store()
        notice = _close_final(store)
        rendered = format_notice(notice, store)

        merged = _merge_todo_stores(store, TodoStore.from_json(store.to_json()))
        merged_notice = next(
            n
            for n in merged.pending_notices()
            if n["kind"] == NOTICE_COMPLETED and n["held"]
        )

        assert format_notice(merged_notice, merged) == rendered

    def test_the_start_mark_crosses_the_merge(self):
        from hermes_cli.tasks import _merge_todo_stores

        store = _plan_store()
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()

        merged = _merge_todo_stores(store, TodoStore.from_json(store.to_json()))

        assert merged.mark_work() is False


class TestCapSparesTheHeldLine:
    def test_the_held_line_survives_a_full_queue(self, caplog):
        store = TodoStore()
        store.write([{"id": "1", "content": "one", "status": "pending"}])
        store.drain_notices()
        assert store.transition("begin", "1")["ok"]
        store.mark_work()
        store.drain_notices()
        assert store.transition("complete", "1")["ok"]
        held = next(n for n in store.pending_notices() if n["held"])

        with caplog.at_level(logging.WARNING, logger="tools.todo_tool"):
            for index in range(MAX_TODO_NOTICES):
                store._record_notice(
                    "started",
                    {"id": str(index + 2), "content": f"t{index}"},
                )

        assert len(store.pending_notices()) == MAX_TODO_NOTICES
        assert held in store.pending_notices()
        assert "dropped 1 lifecycle notice(s)" in caplog.text

    def test_when_every_line_is_held_the_newest_is_dropped(self):
        store = TodoStore()
        for index in range(MAX_TODO_NOTICES):
            store._record_notice(
                "completed",
                {"id": str(index), "content": f"t{index}"},
                held=True,
            )

        store._record_notice(
            "completed", {"id": "new", "content": "new"}, held=True
        )

        assert len(store.pending_notices()) == MAX_TODO_NOTICES
        assert all(n["id"] != "new" for n in store.pending_notices())
