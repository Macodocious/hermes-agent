"""Convergence proof: every task in a multi-item plan reaches a terminal state.

With the agent owning completion, a three-item plan converges by beginning
and completing each item in turn — the store activates the next pending item
mechanically on each completion. No judge, no verdict, no second key.

The test drives the real store; it calls no LLM.
"""

from agent import task_manager
from tools.todo_tool import TodoStore

PLAN_REF = "/plans/three-item-plan/plan.md"
PLAN_ITEMS = (
    ("1", "First item"),
    ("2", "Second item"),
    ("3", "Third item"),
)
TERMINAL_STATUSES = frozenset({"completed", "cancelled"})


def _seed_plan(store: TodoStore) -> None:
    store.write(
        [
            {"id": item_id, "content": content, "status": "pending", "plan": PLAN_REF}
            for item_id, content in PLAN_ITEMS
        ]
    )


def _statuses(store: TodoStore) -> dict:
    return {item["id"]: item["status"] for item in store.read()}


def test_three_item_plan_reaches_all_terminal() -> None:
    """A three-item plan converges: all three items complete.

    Begin the first item; each ``complete`` mechanically activates the next
    pending sibling, so the chain walks the whole plan.
    """
    store = TodoStore()
    _seed_plan(store)

    assert store.transition("begin", "1")["ok"] is True
    for item_id, _content in PLAN_ITEMS:
        assert store.transition("complete", item_id)["ok"] is True

    statuses = _statuses(store)
    assert all(status in TERMINAL_STATUSES for status in statuses.values()), statuses
    assert statuses == {"1": "completed", "2": "completed", "3": "completed"}


def test_next_sibling_activates_on_complete() -> None:
    store = TodoStore()
    _seed_plan(store)
    store.transition("begin", "1")
    store.transition("complete", "1")
    assert _statuses(store)["2"] == "in_progress"


def test_plan_is_complete_reports_when_all_terminal() -> None:
    store = TodoStore()
    _seed_plan(store)
    assert task_manager.plan_is_complete(store, PLAN_REF) is False
    store.transition("begin", "1")
    for item_id, _content in PLAN_ITEMS:
        store.transition("complete", item_id)
    assert task_manager.plan_is_complete(store, PLAN_REF) is True
