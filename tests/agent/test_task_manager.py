"""Tests for the task_manager lifecycle owner (P2).

Covers GoalEngine arming on begin, clearing when the current task leaves
in_progress, the two-key close (verdict observation), and the turn-end
audit (work with no open task must not end cleanly).
"""

from types import SimpleNamespace

import pytest

from agent import task_manager
from tools.todo_tool import TodoStore


def _make_agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store: TodoStore, item_id: str, content: str) -> None:
    store.write([{"id": item_id, "content": content, "status": "pending"}])


def _armed_goal_text(store: TodoStore, item_id: str) -> str:
    """The goal text on_todo_write is expected to arm for an item.

    Derived from the production builder so these tests assert the *wiring*
    (the hook arms the loop with the bound item's goal text) rather than
    freezing the text's wording, which is the goal-builder's own concern.
    """
    item = next(i for i in store.read() if i["id"] == item_id)
    return task_manager._goal_text_for_item(item)


def _spec_prefix(content: str) -> str:
    """The goal text's leading clause for an item's own content."""
    return f"Complete the task per its specification: {content}"


def _calls_set_entry(content: str) -> str:
    """A FakeMgr ``set:`` call entry for an item's content clause.

    FakeMgrs record ``f"set:{text}"``; this builds the matching prefix so an
    assertion can check that the hook armed the loop with the item's goal
    text without freezing the rest of the wording.
    """
    return f"set:{_spec_prefix(content)}"


@pytest.fixture(autouse=True)
def _lifecycle_on(monkeypatch, tmp_path) -> None:
    """Pin the lifecycle config so tests are independent of the host config."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    # The post-close probe must never touch the host probes dir in tests.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)


# ── on_todo_write: GoalEngine arming ──────────────────────────────────


def test_on_todo_write_arms_goal_on_begin(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})

    assert f"set:{_armed_goal_text(store, '1')}" in calls
    assert "persist" in calls


def test_on_todo_write_holds_authorization_on_plan_begin(monkeypatch) -> None:
    """A plan-carrying task stamps the execution-authorization hold on begin."""
    store = TodoStore()
    store.write(
        [
            {
                "id": "1",
                "content": "Build the thing",
                "status": "pending",
                "plan": "/tmp/some/plan.md",
            }
        ]
    )
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def hold_authorization(self) -> None:
            calls.append("hold_authorization")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})

    assert f"set:{_armed_goal_text(store, '1')}" in calls
    assert "hold_authorization" in calls


def test_on_todo_write_does_not_hold_without_plan(monkeypatch) -> None:
    """A plain begin (no plan ref) must not stamp the authorization hold."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def hold_authorization(self) -> None:
            calls.append("hold_authorization")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})

    assert f"set:{_armed_goal_text(store, '1')}" in calls
    assert "hold_authorization" not in calls


def test_on_todo_write_clears_goal_when_no_task_open(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append("set")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    store.transition("pause", "1")
    task_manager.on_todo_write(agent, {"action": "pause", "item_id": "1"})

    assert "clear" in calls
    assert calls.count("set") == 1


def test_on_todo_write_stamps_action_flag(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: None)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    assert agent._task_lifecycle_action_issued is True


def test_on_todo_write_stays_armed_while_close_in_flight(monkeypatch) -> None:
    """A close in flight must NOT clear the goal — the judge's done
    verdict is the second key of the two-key close. Clearing here would
    strand the task in closing forever (regression: PR review)."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append("set")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    store.transition("close", "1")
    task_manager.on_todo_write(agent, {"action": "close", "item_id": "1"})

    assert "clear" not in calls
    assert calls.count("set") == 2


def test_on_todo_write_does_not_rearm_identical_active_goal(monkeypatch) -> None:
    """A routine todo read-back must not reset the judge's turn budget.

    set() builds a fresh goal state with turns_used=0, so re-arming an
    identical active goal on every todo write keeps the loop stuck at
    (1/max_turns) forever — the budget can never fire. When the goal is
    already active with the same text, on_todo_write must leave it alone
    (regression: PR #51 'Continuing toward goal (1/100)' loop)."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")
            self._state = None

        @property
        def state(self):
            return self._state

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")
            # set() mirrors GoalManager: a fresh active goal with the text.
            self._state = SimpleNamespace(status="active", goal=text)

        def clear(self) -> None:
            calls.append("clear")

    manager = FakeMgr()
    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: manager)
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    # First write arms the loop (manager starts with no state).
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    # Subsequent read-back writes must not re-arm.
    task_manager.on_todo_write(agent, {})

    assert len([c for c in calls if c.startswith("set:")]) == 1
    assert "persist" in calls


def test_on_todo_write_rearms_when_active_goal_text_changes(monkeypatch) -> None:
    """Editing an in_progress task's content must sync the goal text."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    armed_text = task_manager._goal_text_for_item(
        {"id": "1", "content": "Build the thing now", "status": "in_progress"}
    )
    stale_text = task_manager._goal_text_for_item(
        {"id": "1", "content": "Old content", "status": "in_progress"}
    )
    assert stale_text != armed_text

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        @property
        def state(self):
            return SimpleNamespace(status="active", goal=stale_text)

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    # Change the item content so the goal text differs from the manager's.
    store.write([{"id": "1", "content": "Build the thing now", "status": "in_progress"}])
    task_manager.on_todo_write(agent, {})

    assert f"set:{armed_text}" in calls
    assert _spec_prefix("Build the thing now") in armed_text


# ── config toggle: tasks.lifecycle.enabled=false disables the lifecycle ─


def test_disabled_lifecycle_short_circuits_hooks(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": False})

    # No goal arming, no action stamp, no persistence.
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    assert agent._task_lifecycle_action_issued is False

    # No audit nudge.
    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is None

    # No verdict observation.
    assert task_manager.observe_verdict(agent, {"verdict": "done"}) is None
    assert task_manager.observe_verdict_for_session("test-session", {"verdict": "done"}) is None


# ── observe_verdict: the two-key close ────────────────────────────────


def test_verdict_records_finalized_id_on_canonical_close(monkeypatch) -> None:
    """The verdict records which task it actually finalized.

    The gateway emits ``✅ Task completed`` off this id, so a done verdict
    and an actual completion can never be confused.
    """
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    decision = {"verdict": "done"}
    task_manager.observe_verdict(agent, decision)

    assert decision.get("lifecycle_finalized_id") == "1"


def test_verdict_records_no_finalized_id_for_a_continue(monkeypatch) -> None:
    """A plan-level continue that reopens the task finalizes nothing."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    decision = {"verdict": "continue", "reason": "not yet"}
    task_manager.observe_verdict(agent, decision)

    assert decision.get("lifecycle_finalized_id") is None
    assert store.read()[0]["status"] == "in_progress"


def test_verdict_records_finalized_id_on_auto_finalize(monkeypatch) -> None:
    """A done verdict on an open task closes and finalizes it for real."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    decision = {"verdict": "done"}
    task_manager.observe_verdict(agent, decision)

    assert decision.get("lifecycle_finalized_id") == "1"
    assert store.read()[0]["status"] == "completed"


def test_verdict_records_no_finalized_id_when_blocked(monkeypatch) -> None:
    """A blocked done verdict is a park: nothing finalizes, no id claims it."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    decision = {"verdict": "done", "blocked": True}
    task_manager.observe_verdict(agent, decision)

    assert decision.get("lifecycle_finalized_id") is None
    assert store.read()[0]["status"] == "closing"


def test_verdict_done_finalizes_closing_task(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert store.read()[0]["status"] == "completed"


def test_verdict_done_plan_sibling_pulls_next(monkeypatch) -> None:
    """R3 plan-level completion: finalizing a plan-carrying item while a
    sibling of the same plan remains must return a plan-continuation
    nudge naming the next sibling — the plan, not the todo list, is the
    unit of approved work (kills the 'items 4-7 still pending' gap)."""
    store = TodoStore()
    plan_ref = "/tmp/some/plan.md"
    store.write(
        [
            {"id": "1", "content": "Build the thing", "status": "pending", "plan": plan_ref},
            {"id": "2", "content": "Ship the thing", "status": "pending", "plan": plan_ref},
        ]
    )
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert store.read()[0]["status"] == "completed"
    # The sibling is NOT silently begun — the nudge pulls the agent back.
    assert store.read()[1]["status"] == "pending"
    assert nudge is not None
    assert "action=begin" in nudge
    assert "item_id=2" in nudge
    assert "Ship the thing" in nudge


def test_verdict_done_plan_sibling_pulls_next_auto_finalize(monkeypatch) -> None:
    """R3 on the auto-finalize path: a done verdict on an OPEN plan-carrying
    item finalizes it and returns the plan-continuation nudge instead of
    the plain finalize nudge."""
    store = TodoStore()
    plan_ref = "/tmp/some/plan.md"
    store.write(
        [
            {"id": "1", "content": "Build the thing", "status": "pending", "plan": plan_ref},
            {"id": "2", "content": "Ship the thing", "status": "pending", "plan": plan_ref},
        ]
    )
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done", "reason": "judge says done"})

    assert store.read()[0]["status"] == "completed"
    assert store.read()[1]["status"] == "pending"
    assert nudge is not None
    assert "action=begin" in nudge
    assert "item_id=2" in nudge


def test_verdict_done_last_plan_item_no_nudge(monkeypatch) -> None:
    """R3 edge: finalizing the LAST plan-carrying item (no siblings left)
    returns no plan nudge — the plan is complete."""
    store = TodoStore()
    plan_ref = "/tmp/some/plan.md"
    store.write(
        [{"id": "1", "content": "Build the thing", "status": "pending", "plan": plan_ref}]
    )
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert store.read()[0]["status"] == "completed"


def test_verdict_done_finalizes_closing_task_only(monkeypatch) -> None:
    """A done verdict finalizes exactly the closing task — no chain.

    The begin pivot and the write-path invariant make a concurrent
    in_progress task impossible through the tool, so the judge's done
    verdict is the second key for the closing task alone (the PR #66
    overlap chain is removed). This constructs the old overlap directly
    (internal mutation) to prove finalization touches nothing else.
    """
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Build the thing", "status": "pending"},
            {"id": "2", "content": "Ship the thing", "status": "pending"},
        ]
    )
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    # Old overlap: an in_progress task coexisting with the closing one
    # (constructible only by internal mutation now; both public doors
    # refuse it).
    for item in store._items:
        if item["id"] == "2":
            item["status"] = "in_progress"
    nudge = task_manager.observe_verdict(agent, {"verdict": "done", "reason": "both done"})

    assert nudge is None
    # The closing task finalized; the overlapped in_progress task was NOT
    # auto-finalized — nothing happens behind the model's back. The
    # write-path backstop demotes it on the next write, and the model
    # begins it explicitly after the judge clears.
    assert [i["status"] for i in store.read()] == ["completed", "in_progress"]


def test_verdict_done_with_only_closing_task_keeps_single_finalize(monkeypatch) -> None:
    """No in_progress task: the done verdict finalizes only the closing
    task and returns no nudge (existing behavior preserved)."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert store.read()[0]["status"] == "completed"


def test_verdict_continue_with_closing_and_in_progress_keeps_both_open(monkeypatch) -> None:
    """A continue verdict must not finalize anything: the premature close
    returns to in_progress, the write-path invariant demotes the
    concurrent in_progress item to pending, and a rework task is
    appended with the review-failure reason.

    The overlap is now constructible only by internal mutation — the
    begin pivot refuses it — mirroring the verdict-done test. The
    rework append rides the write path, so the invariant collapses the
    mutated overlap to a single current task.
    """
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Build the thing", "status": "pending"},
            {"id": "2", "content": "Ship the thing", "status": "pending"},
        ]
    )
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    for item in store._items:
        if item["id"] == "2":
            item["status"] = "in_progress"
    nudge = task_manager.observe_verdict(
        agent, {"verdict": "continue", "reason": "the fix was reverted"}
    )

    assert nudge is not None
    assert "the fix was reverted" in nudge
    assert "rework task" in nudge
    assert [i["status"] for i in store.read()] == ["in_progress", "pending", "pending"]
    rework = next(i for i in store.read() if i.get("review_of") == "1")
    assert rework["source"] == "review"
    assert "the fix was reverted" in rework["content"]


def test_verdict_continue_returns_premature_close_to_in_progress(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(
        agent, {"verdict": "continue", "reason": "spec not met"}
    )

    assert nudge is not None
    assert "spec not met" in nudge
    assert store.read()[0]["status"] == "in_progress"
    rework = next(i for i in store.read() if i.get("review_of") == "1")
    assert rework["status"] == "pending"
    assert rework["source"] == "review"


def test_verdict_wait_on_closing_task_parks_without_rework(monkeypatch) -> None:
    """A wait verdict is a park, not a rejection: the task returns to
    in_progress and no rework task is spawned (the loop resumes
    automatically when the async thing clears)."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(
        agent, {"verdict": "wait", "reason": "waiting on the build"}
    )

    assert nudge is None
    assert store.read()[0]["status"] == "in_progress"
    assert not any(i.get("review_of") == "1" for i in store.read())


def test_verdict_done_on_open_task_finalizes_with_nudge(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done", "reason": "looks done"})

    assert nudge is not None
    assert "close" in nudge
    assert store.read()[0]["status"] == "completed"


def test_verdict_continue_on_open_task_is_noop(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "continue"})

    assert nudge is None
    assert store.read()[0]["status"] == "in_progress"


# ── audit_turn_end: work with no open task must not end cleanly ───────


def test_audit_clean_when_task_in_progress() -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    store.transition("begin", "1")

    nudge = task_manager.audit_turn_end(
        agent, final_response="Done.", interrupted=False, tool_call_count=3
    )
    assert nudge is None


def test_audit_pulls_back_after_work_with_no_open_task() -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)

    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is not None
    assert "begin" in nudge


def test_audit_skips_terse_conversational_reply() -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)

    nudge = task_manager.audit_turn_end(
        agent, final_response="Sure.", interrupted=False, tool_call_count=0
    )
    assert nudge is None


def test_audit_skips_when_lifecycle_action_issued() -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    agent._task_lifecycle_action_issued = True

    nudge = task_manager.audit_turn_end(
        agent, final_response="Pausing here.", interrupted=False, tool_call_count=2
    )
    assert nudge is None


def test_audit_skips_interrupted_turns() -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)

    nudge = task_manager.audit_turn_end(
        agent, final_response="partial", interrupted=True, tool_call_count=2
    )
    assert nudge is None


def test_audit_skips_when_no_task_list() -> None:
    agent = _make_agent(TodoStore())

    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is None


# ── on_todo_write: block parks the loop ───────────────────────────────


def test_on_todo_write_block_parks_goal_with_reason(monkeypatch) -> None:
    """A declared block is the state change: the hook parks the goal."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def park(self, reason: str) -> None:
            calls.append(f"park:{reason}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(
        agent, {"action": "block", "item_id": "1", "reason": "need your call"}
    )

    assert "park:need your call" in calls
    assert "persist" in calls
    # Block must not re-arm (that would reset the turn budget) or clear.
    assert not any(c.startswith("set:") for c in calls)
    assert "clear" not in calls


def test_on_todo_write_block_leaves_task_in_progress(monkeypatch) -> None:
    """The task is still the current work — block does not shelve it."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)

    class FakeMgr:
        def park(self, reason: str) -> None:
            pass

        def set(self, text: str, **kwargs) -> None:
            raise AssertionError("block must not arm the goal")

        def clear(self) -> None:
            raise AssertionError("block must not clear the goal")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    task_manager.on_todo_write(
        agent, {"action": "block", "item_id": "1", "reason": "waiting on you"}
    )

    assert store.read()[0]["status"] == "in_progress"


def test_on_todo_write_routine_write_does_not_release_park(monkeypatch) -> None:
    """A parked goal is the user's to release: a later routine write must
    not re-arm it (which would rebuild state with the barrier down)."""
    store = TodoStore()
    _seed(store, "1", "Edited content while parked")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            # Parked, and the goal text no longer matches the item.
            self.state = SimpleNamespace(
                status="active",
                goal="Complete the task per its specification: old content",
                awaiting_user_input=True,
            )

        def set(self, text: str, **kwargs) -> None:
            calls.append(f"set:{text}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": None})

    assert not any(c.startswith("set:") for c in calls)
    assert "clear" not in calls


# ── the goal binds the judge to the task's own row, not the plan ──────


def test_goal_text_binds_to_task_row_not_plan_file(tmp_path) -> None:
    """A plan-carrying item's goal must not inline the plan file.

    Inlining the plan handed the judge a multi-item document whose siblings
    stay open until the plan's last item closes, so every item's DONE verdict
    was blocked by rows that were none of its business (the D1 deadlock).
    """
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# Plan\n- item A\n- item B\n- item C\n", encoding="utf-8")

    text = task_manager._goal_text_for_item(
        {"id": "1", "content": "Item A", "status": "in_progress", "plan": str(plan_file)}
    )

    assert _spec_prefix("Item A") in text
    assert str(plan_file) not in text
    assert "item B" not in text


def test_goal_text_names_the_bound_task() -> None:
    """The goal text states which task the verdict is about."""
    text = task_manager._goal_text_for_item(
        {"id": "7", "content": "Item A", "status": "in_progress"}
    )
    assert "todo item 7" in text


def test_goal_text_without_id_has_no_bound_task_clause() -> None:
    """An id-less item still yields a usable goal text."""
    text = task_manager._goal_text_for_item({"content": "Item A", "status": "pending"})
    assert _spec_prefix("Item A") in text
    assert "Bound task" not in text


# ── plan_is_complete: the mechanical aggregate over the bound plan ────


def test_plan_is_complete_false_while_any_task_is_open() -> None:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "A", "status": "completed", "plan": "P"},
            {"id": "2", "content": "B", "status": "pending", "plan": "P"},
        ]
    )
    assert task_manager.plan_is_complete(store, "P") is False


def test_plan_is_complete_true_when_every_task_is_terminal() -> None:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "A", "status": "completed", "plan": "P"},
            {"id": "2", "content": "B", "status": "cancelled", "plan": "P"},
        ]
    )
    assert task_manager.plan_is_complete(store, "P") is True


def test_plan_is_complete_ignores_rows_of_other_plans() -> None:
    """Only rows carrying the same plan ref are the plan's business."""
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "A", "status": "completed", "plan": "P"},
            {"id": "2", "content": "B", "status": "pending", "plan": "OTHER"},
        ]
    )
    assert task_manager.plan_is_complete(store, "P") is True


def test_plan_is_complete_false_for_unknown_or_empty_ref() -> None:
    store = TodoStore()
    store.write([{"id": "1", "content": "A", "status": "completed", "plan": "P"}])
    assert task_manager.plan_is_complete(store, "NOPE") is False
    assert task_manager.plan_is_complete(store, "") is False


# ── _advances_open_plan: membership is the TOOLSET, not the tool name ──


def test_advances_open_plan_resolves_toolset_not_tool_name() -> None:
    """A plan tool stays plan-advancing after a rename.

    Membership is the toolset the registry reports, so the write_plan tool
    surface can be renamed under the plugin without silently dropping out of
    the audit predicate. Previously a hardcoded name set plus a `plan_` prefix
    stopped matching every plan tool the moment the surface was renamed.
    """
    from tools.registry import registry

    registry.register(
        name="write_spec",
        toolset="write_plan",
        schema={"name": "write_spec", "parameters": {}},
        handler=lambda args, **kw: None,
    )
    assert task_manager._advances_open_plan(["write_spec"]) is True


def test_advances_open_plan_ignores_unrelated_tools() -> None:
    """Work with no plan-advancing tool is still drift."""
    assert task_manager._advances_open_plan(["read_file", "patch"]) is False
    assert task_manager._advances_open_plan([]) is False
    assert task_manager._advances_open_plan(None) is False


def test_advances_open_plan_accepts_todo_transitions() -> None:
    """The lifecycle lever itself always counts as advancing the plan."""
    assert task_manager._advances_open_plan(["todo"]) is True
