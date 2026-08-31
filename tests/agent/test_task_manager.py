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


@pytest.fixture(autouse=True)
def _lifecycle_on(monkeypatch) -> None:
    """Pin the lifecycle config so tests are independent of the host config."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    # The post-close review must not fire real LLM calls in unit tests.
    monkeypatch.setattr(task_manager, "_review_config", lambda: {"enabled": False})


# ── on_todo_write: GoalEngine arming ──────────────────────────────────


def test_on_todo_write_arms_goal_on_begin(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str) -> None:
            calls.append(f"set:{text}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    store.transition("begin", "1")
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})

    assert "set:Complete the task: Build the thing" in calls
    assert "persist" in calls


def test_on_todo_write_clears_goal_when_no_task_open(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list[str] = []

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        def set(self, text: str) -> None:
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

        def set(self, text: str) -> None:
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

        def set(self, text: str) -> None:
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

    class FakeMgr:
        def __init__(self, **kwargs):
            calls.append("init")

        @property
        def state(self):
            return SimpleNamespace(status="active", goal="Complete the task: Old content")

        def set(self, text: str) -> None:
            calls.append(f"set:{text}")

        def clear(self) -> None:
            calls.append("clear")

    monkeypatch.setattr(task_manager, "_load_goal_manager", lambda a: FakeMgr())
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    # Change the item content so the goal text differs from the manager's.
    store.write([{"id": "1", "content": "Build the thing now", "status": "in_progress"}])
    task_manager.on_todo_write(agent, {})

    assert "set:Complete the task: Build the thing now" in calls


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


def test_verdict_continue_returns_premature_close_to_in_progress(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    store.transition("begin", "1")
    store.transition("close", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "continue"})

    assert nudge is None
    assert store.read()[0]["status"] == "in_progress"


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
