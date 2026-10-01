"""Tests for ordered-list identification in task-lifecycle notifications.

Covers the three new pieces that make a lifecycle notice name the task the
way a human reads it, and that announce plan completion:

- ``_enrich_todo_transition`` — attaches ``position``/``total``/``plan`` to a
  detected transition before the UI renders it.
- ``task_position_prefix`` — the shared "<position> of <total>: " prefix the
  gateway bubbles and the CLI spinner both build.
- ``task_position`` / ``task_plan_ref`` / ``plan_task_total`` /
  ``plan_is_complete`` — the ordered-list and plan-aggregate helpers the
  completion line and the plan-completion signal consume.
"""

from types import SimpleNamespace

import pytest

from agent.display import task_position_label
from agent.task_manager import (
    plan_is_complete,
    plan_task_total,
    task_plan_ref,
    task_position,
)
from agent.tool_executor import _detect_todo_task_start, _detect_todo_task_stop
from tools.todo_tool import TodoStore


def _store(items):
    store = TodoStore()
    store.write(items)
    return store


def _agent_with_store(items):
    store = _store(items)
    return SimpleNamespace(_todo_store=store), store


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


class TestEnrichedTransitions:
    def test_start_carries_position_total_and_plan(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "a", "status": "completed", "plan": "p"},
            {"id": "2", "content": "b", "status": "pending", "plan": "p"},
        ])
        started = _detect_todo_task_start(
            agent, "todo",
            {"todos": [{"id": "2", "content": "b", "status": "in_progress"}]},
        )
        assert started is not None
        assert started["position"] == 2
        assert started["total"] == 2
        assert started["plan"] == "p"

    def test_stop_carries_position_total_and_plan(self):
        agent, _ = _agent_with_store([
            {"id": "1", "content": "a", "status": "in_progress", "plan": "p"},
            {"id": "2", "content": "b", "status": "pending", "plan": "p"},
        ])
        stopped = _detect_todo_task_stop(
            agent, "todo",
            {"todos": [{"id": "1", "content": "a", "status": "cancelled"}]},
        )
        assert stopped is not None
        assert stopped["position"] == 1
        assert stopped["total"] == 2
        assert stopped["plan"] == "p"
        assert stopped["status"] == "cancelled"


class TestTaskPositionLabel:
    def test_builds_the_shared_label(self):
        assert task_position_label({"position": 2, "total": 5}) == "2 of 5"

    def test_absent_position_renders_no_label(self):
        assert task_position_label({"content": "a"}) == ""
        assert task_position_label(None) == ""

    @pytest.mark.parametrize("position,total", [(3, 0), (0, 3)])
    def test_incomplete_pair_renders_no_label(self, position, total):
        assert task_position_label({"position": position, "total": total}) == ""
