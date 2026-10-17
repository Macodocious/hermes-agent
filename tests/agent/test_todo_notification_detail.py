"""Tests for ordered-list identification in task-lifecycle notifications.

Covers the pieces that make a lifecycle notice name the task the way a human
reads it, and that announce plan completion:

- ``task_position_label`` — the shared "<position> of <total>" label.
- ``task_position`` / ``task_plan_ref`` / ``plan_task_total`` /
  ``plan_is_complete`` — the ordered-list and plan-aggregate helpers the
  completion line and the plan-completion signal consume.
- the post-write emitter — one message per notice, at the transition.
"""

import pytest

from agent.display import task_position_label
from agent.task_manager import (
    plan_is_complete,
    plan_task_total,
    task_plan_ref,
    task_position,
)
from tools.todo_tool import TodoStore


def _store(items):
    store = TodoStore()
    store.write(items)
    return store


class TestTaskPosition:
    def test_position_is_the_list_index_not_the_id(self):
        store = _store([
            {"id": "1", "content": "a", "status": "completed"},
            {"id": "2", "content": "b", "status": "completed"},
            {"id": "3", "content": "c", "status": "closing"},
        ])
        assert task_position(store, "3") == (3, 3)

    def test_absent_row_reports_zero_position_but_real_total(self):
        store = _store([
            {"id": "1", "content": "a", "status": "pending"},
            {"id": "2", "content": "b", "status": "pending"},
        ])
        assert task_position(store, "99") == (0, 2)


class TestTaskPlanRef:
    def test_reads_the_rows_plan_reference(self):
        store = _store([
            {"id": "1", "content": "a", "status": "completed", "plan": "my-plan"},
        ])
        assert task_plan_ref(store, "1") == "my-plan"

    def test_absent_row_and_absent_reference_are_the_same_answer(self):
        store = _store([{"id": "1", "content": "a", "status": "pending"}])
        assert task_plan_ref(store, "1") == ""
        assert task_plan_ref(store, "99") == ""


class TestPlanTaskTotal:
    def test_counts_only_the_rows_carrying_the_plan(self):
        store = _store([
            {"id": "1", "content": "a", "status": "completed", "plan": "p"},
            {"id": "2", "content": "b", "status": "completed", "plan": "p"},
            {"id": "3", "content": "c", "status": "pending", "plan": "other"},
        ])
        assert plan_task_total(store, "p") == 2
        assert plan_task_total(store, "other") == 1

    def test_empty_reference_counts_nothing(self):
        store = _store([{"id": "1", "content": "a", "status": "pending"}])
        assert plan_task_total(store, "") == 0


class TestPlanIsComplete:
    def test_all_rows_terminal_is_complete(self):
        store = _store([
            {"id": "1", "content": "a", "status": "completed", "plan": "p"},
            {"id": "2", "content": "b", "status": "cancelled", "plan": "p"},
        ])
        assert plan_is_complete(store, "p") is True

    def test_one_open_row_keeps_the_plan_open(self):
        store = _store([
            {"id": "1", "content": "a", "status": "completed", "plan": "p"},
            {"id": "2", "content": "b", "status": "pending", "plan": "p"},
        ])
        assert plan_is_complete(store, "p") is False

    def test_no_reference_and_no_matching_row_are_not_complete(self):
        store = _store([{"id": "1", "content": "a", "status": "completed"}])
        assert plan_is_complete(store, "") is False
        assert plan_is_complete(store, "unknown") is False


class TestTaskPositionLabel:
    def test_builds_the_shared_label(self):
        assert task_position_label({"position": 2, "total": 5}) == "2 of 5"

    def test_absent_position_renders_no_label(self):
        assert task_position_label({"content": "a"}) == ""
        assert task_position_label(None) == ""

    @pytest.mark.parametrize("position,total", [(3, 0), (0, 3)])
    def test_incomplete_pair_renders_no_label(self, position, total):
        assert task_position_label({"position": position, "total": total}) == ""


class TestPlanCompleteNotice:
    """The plan-completion line belongs to the plan's final task, once.

    ``format_notice`` consults the store when it renders, so a batch of
    completions used to render the plan line for every ref-carrying row — one
    per completing task, the first of them mid-plan.
    """

    @staticmethod
    def _drained_lines(store):
        from tools.todo_tool import format_notice

        return [
            line
            for line in (format_notice(n, store) for n in store.drain_notices())
            if line
        ]

    def test_a_batch_of_completions_yields_one_plan_line(self):
        store = _store([
            {"id": "1", "content": "seed", "status": "pending"},
            {"id": "2", "content": "a", "status": "pending", "plan": "p"},
            {"id": "3", "content": "b", "status": "pending", "plan": "p"},
        ])
        for item_id in ("1", "2", "3"):
            store.transition("begin", item_id)
            store.transition("complete", item_id)

        lines = self._drained_lines(store)

        assert [line for line in lines if line.startswith("🏁")] == [
            "🏁 Plan complete: all 2 tasks done (p)"
        ]
        # The plan's non-final task reports itself, numbered within the plan
        # so its total matches the plan line's count.
        assert "✅ Task 1 of 2 complete: a" in lines
        # The un-tagged seed row still reports its own position.
        assert "✅ Task 1 of 3 complete: seed" in lines

    def test_plan_line_waits_for_the_final_task(self):
        store = _store([
            {"id": "1", "content": "a", "status": "pending", "plan": "p"},
            {"id": "2", "content": "b", "status": "pending", "plan": "p"},
        ])
        store.transition("begin", "1")
        store.transition("complete", "1")

        lines = self._drained_lines(store)

        assert not [line for line in lines if line.startswith("🏁")]
        assert "✅ Task 1 of 2 complete: a" in lines


class TestPostWriteNoticeEmission:
    """The post-write hook ships a transition's notice at the transition."""

    class _Agent:
        def __init__(self, store, callback):
            self._todo_store = store
            self.todo_notice_callback = callback

    def test_emits_and_drains_the_pending_notices(self):
        from agent.task_manager import _emit_pending_notices

        store = _store([{"id": "1", "content": "ship it", "status": "pending"}])
        store.transition("begin", "1")
        sent = []

        _emit_pending_notices(self._Agent(store, sent.append))

        assert sent == ["▶️ Working on: ship it"]
        assert store.pending_notices() == []

    def test_no_callback_leaves_the_notices_for_the_turn_drain(self):
        from agent.task_manager import _emit_pending_notices

        store = _store([{"id": "1", "content": "ship it", "status": "pending"}])
        store.transition("begin", "1")

        _emit_pending_notices(self._Agent(store, None))

        assert [notice["kind"] for notice in store.pending_notices()] == ["started"]

    def test_start_line_numbers_only_the_plan_rows(self):
        from tools.todo_tool import format_notice

        store = _store([
            {"id": "1", "content": "the opening request", "status": "pending"},
            {"id": "2", "content": "a", "status": "pending", "plan": "p"},
        ])
        store.transition("begin", "1")
        store.transition("complete", "1")  # promotes row 2 and records its start

        lines = [
            line
            for line in (format_notice(n, store) for n in store.drain_notices())
            if line
        ]

        # A row outside a plan is not in a plan's sequence: it stays plain.
        assert "▶️ Working on: the opening request" in lines
        # A plan row is numbered within its plan, so its total matches the
        # count the plan line prints.
        assert "▶️ Working on task 1 of 1: a" in lines

    def test_a_raising_callback_keeps_the_notices(self):
        from agent.task_manager import _emit_pending_notices

        store = _store([{"id": "1", "content": "ship it", "status": "pending"}])
        store.transition("begin", "1")

        def _boom(_message):
            raise RuntimeError("no delivery rail")

        _emit_pending_notices(self._Agent(store, _boom))

        assert [notice["kind"] for notice in store.pending_notices()] == ["started"]

    def test_each_notice_is_its_own_message(self):
        """One notice, one message — never one joined block.

        Two transitions in one write used to arrive welded together, a start
        line under the completion that produced it, because the emitter joined
        the lines with a newline before the callback saw them.
        """
        from agent.task_manager import _emit_pending_notices

        store = _store([{"id": "1", "content": "ship it", "status": "pending"}])
        store.transition("begin", "1")
        store.transition("complete", "1")
        sent = []

        _emit_pending_notices(self._Agent(store, sent.append))

        assert sent == [
            "▶️ Working on: ship it",
            "✅ Task 1 of 1 complete: ship it",
        ]
