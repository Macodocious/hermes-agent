"""Convergence proof: every task in a multi-item plan reaches a terminal state.

The invariant this module proves:

    Every task reaches a terminal state (``completed``, ``cancelled`` or
    ``escalated``) in a bounded number of turns, independent of how many
    sibling tasks the plan has.

The defect it guards against is the D1 deadlock: the judge was handed a
multi-item plan document, so every item's DONE verdict was blocked by rows
that were none of its business, and a healthy task was reopened whenever a
sibling was still open. Nothing converged.

The test drives a three-item plan through the two-key close for each item —
seed, begin, close, judge done, next item — using the real store and the
real verdict entry points. It calls no LLM and does not depend on the
judge's prose. It fails on the unmodified core and passes after the fix.
"""

from types import SimpleNamespace

import pytest

from agent import task_manager
from tools.todo_tool import TodoStore

PLAN_REF = "/plans/three-item-plan/plan.md"
PLAN_ITEMS = (
    ("1", "First item"),
    ("2", "Second item"),
    ("3", "Third item"),
)
TERMINAL_STATUSES = frozenset({"completed", "cancelled", "escalated"})


@pytest.fixture(autouse=True)
def _lifecycle_on(monkeypatch, tmp_path) -> None:
    """Pin the lifecycle config and keep the probe off the host tree."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)


def _seed_plan(store: TodoStore) -> None:
    store.write(
        [
            {"id": item_id, "content": content, "status": "pending", "plan": PLAN_REF}
            for item_id, content in PLAN_ITEMS
        ]
    )


def _agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="convergence-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _statuses(store: TodoStore) -> dict:
    return {item["id"]: item["status"] for item in store.read()}


def _close_and_finalize(store: TodoStore, agent: SimpleNamespace, item_id: str) -> None:
    """Walk one item through the two-key close: begin, close, judge done."""
    assert store.transition("begin", item_id)["ok"] is True
    assert store.transition("close", item_id)["ok"] is True
    decision = {"verdict": "done", "reason": "complete", "bound_task_id": item_id}
    task_manager.observe_verdict(agent, decision)
    assert decision.get("lifecycle_finalized_id") == item_id


def test_three_item_plan_reaches_all_terminal(monkeypatch, tmp_path) -> None:
    """A three-item plan converges: all three items terminate."""
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    store = TodoStore()
    _seed_plan(store)
    agent = _agent(store)

    for item_id, _content in PLAN_ITEMS:
        _close_and_finalize(store, agent, item_id)

    statuses = _statuses(store)
    assert all(status in TERMINAL_STATUSES for status in statuses.values()), statuses
    assert statuses == {"1": "completed", "2": "completed", "3": "completed"}


def test_open_siblings_never_append_rework_rows(monkeypatch) -> None:
    """The plan's open siblings are not the closing task's business."""
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    store = TodoStore()
    _seed_plan(store)
    agent = _agent(store)

    store.transition("begin", "1")
    store.transition("close", "1")
    task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "complete", "bound_task_id": "1"}
    )

    assert store.read()[0]["status"] == "completed"
    assert not [i for i in store.read() if i.get("source") == "review"]


def test_plan_still_converges_through_a_rejected_close(monkeypatch) -> None:
    """A rejection leg: one close fails review once, the plan still finishes.

    Item 2's close is rejected: the task returns to in_progress with no
    rework row and no nudge (the async review and its auto-rework append
    were removed). The plan continues past the failure and every original
    item still terminates.
    """
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    store = TodoStore()
    _seed_plan(store)
    agent = _agent(store)

    _close_and_finalize(store, agent, "1")

    # Item 2 is closed, then rejected by the judge, then closed again.
    store.transition("begin", "2")
    store.transition("close", "2")
    rework_nudge = task_manager.observe_verdict(
        agent, {"verdict": "continue", "reason": "spec not met", "bound_task_id": "2"}
    )
    assert rework_nudge is None
    assert not [i for i in store.read() if i.get("review_of") == "2"]
    assert store.read()[1]["status"] == "in_progress"

    # Close and finalize the original item.
    assert store.transition("close", "2")["ok"] is True
    task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "rework complete", "bound_task_id": "2"}
    )
    assert _statuses(store)["2"] == "completed"

    _close_and_finalize(store, agent, "3")

    statuses = _statuses(store)
    original = {i: statuses[i] for i in ("1", "2", "3")}
    assert all(status in TERMINAL_STATUSES for status in original.values()), original
