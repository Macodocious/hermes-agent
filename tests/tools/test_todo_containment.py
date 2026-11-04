"""Tests for the containment gate: a pending head row must be begun before work.

The gate is a store method, so the tests drive it directly: any pending head
row refuses a non-todo call until it is begun, except a row whose plan is
still being authored.
"""

import pytest

from tools.todo_tool import TodoStore, _EXECUTED_MARK_PREFIX


def _plan_doc(tmp_path, *, executed: bool) -> str:
    """A minimal plan document, executed or still being written."""
    doc = tmp_path / "plan.md"
    body = "# Plan\n\n| Task |\n| --- |\n| First |\n"
    if executed:
        body += "\n" + _EXECUTED_MARK_PREFIX + "─" * 48 + "\n" + "a" * 64 + "\n"
    doc.write_text(body, encoding="utf-8")
    return str(doc)


@pytest.fixture
def store(tmp_path) -> TodoStore:
    s = TodoStore()
    s.write(
        [
            {
                "id": "1",
                "content": "First task",
                "status": "pending",
                "plan": _plan_doc(tmp_path, executed=True),
            }
        ]
    )
    return s


def test_pending_plan_task_refuses_work(store: TodoStore) -> None:
    refusal = store.containment_refusal()
    assert "has not begun" in refusal
    assert "action='begin'" in refusal


def test_begun_task_allows_work(store: TodoStore) -> None:
    store.transition("begin", "1")
    assert store.containment_refusal() == ""


def test_unauthorized_plan_does_not_confine(tmp_path) -> None:
    s = TodoStore()
    s.write(
        [
            {
                "id": "1",
                "content": "First task",
                "status": "pending",
                "plan": _plan_doc(tmp_path, executed=False),
            }
        ]
    )
    assert s.containment_refusal() == ""


def test_row_without_a_plan_refuses_work() -> None:
    s = TodoStore()
    s.write([{"id": "1", "content": "First task", "status": "pending"}])
    refusal = s.containment_refusal()
    assert "has not begun" in refusal
    assert s.transition("begin", "1") is not None
    assert s.containment_refusal() == ""
