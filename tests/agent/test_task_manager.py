"""Tests for the task_manager lifecycle owner.

The agent owns the task transitions (applied inside the todo store), so this
module's job is narrow: ``on_todo_write`` stamps the turn and persists the
store, and ``audit_turn_end`` pulls the agent back when a turn did work with
no open task. The judge-era goal arming and verdict handling are gone.

Covers ``on_todo_write``, the deterministic helpers (plan_is_complete,
task_position, task_plan_ref, plan_task_total), the turn-end audit, the
config toggle, and that the judge machinery has been removed from the module.
"""

import json
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
def _lifecycle_on(monkeypatch, tmp_path) -> None:
    """Pin the lifecycle config so tests are independent of the host config."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    # The post-close probe must never touch the host probes dir in tests.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)


# ── on_todo_write: agent-owned — stamp + persist, never arm a goal ────


def test_on_todo_write_stamps_action_flag(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    assert agent._task_lifecycle_action_issued is True


def test_on_todo_write_persists(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    calls: list = []
    monkeypatch.setattr(task_manager, "_persist", lambda a: calls.append("persist"))

    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    assert "persist" in calls


def test_on_todo_write_ignores_a_read_back(monkeypatch) -> None:
    """A routine read-back (no action) does not stamp the turn."""
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    task_manager.on_todo_write(agent, {})
    assert agent._task_lifecycle_action_issued is False


def test_module_has_no_judge_machinery() -> None:
    """The goal arming and verdict handling are removed from the module."""
    assert not hasattr(task_manager, "_load_goal_manager")
    assert not hasattr(task_manager, "observe_verdict")
    assert not hasattr(task_manager, "observe_verdict_for_session")
    assert not hasattr(task_manager, "_apply_verdict")
    assert not hasattr(task_manager, "_goal_text_for_item")


# ── config toggle: tasks.lifecycle.enabled=false disables the lifecycle ─


def test_disabled_lifecycle_short_circuits_hooks(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": False})

    # No action stamp.
    task_manager.on_todo_write(agent, {"action": "begin", "item_id": "1"})
    assert agent._task_lifecycle_action_issued is False

    # No audit nudge.
    nudge = task_manager.audit_turn_end(
        agent, final_response="I did the work.", interrupted=False, tool_call_count=2
    )
    assert nudge is None


# ── is_completion: retained predicate for out-of-module callers ────────


def test_is_completion_true_for_a_done_unblocked_verdict() -> None:
    assert task_manager.is_completion({"verdict": "done"}) is True


def test_is_completion_false_for_continue_or_blocked() -> None:
    assert task_manager.is_completion({"verdict": "continue"}) is False
    assert task_manager.is_completion({"verdict": "done", "blocked": True}) is False


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


# ── task_position / task_plan_ref / plan_task_total ───────────────────


def test_task_position_reports_list_index_and_total() -> None:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "A", "status": "pending"},
            {"id": "2", "content": "B", "status": "pending"},
        ]
    )
    assert task_manager.task_position(store, "2") == (2, 2)
    assert task_manager.task_position(store, "nope") == (0, 2)


def test_task_plan_ref_reads_the_row_ref() -> None:
    store = TodoStore()
    store.write([{"id": "1", "content": "A", "status": "pending", "plan": "P"}])
    assert task_manager.task_plan_ref(store, "1") == "P"
    assert task_manager.task_plan_ref(store, "nope") == ""


def test_plan_task_total_counts_the_plans_rows() -> None:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "A", "status": "pending", "plan": "P"},
            {"id": "2", "content": "B", "status": "pending", "plan": "P"},
            {"id": "3", "content": "C", "status": "pending", "plan": "Q"},
        ]
    )
    assert task_manager.plan_task_total(store, "P") == 2
    assert task_manager.plan_task_total(store, "") == 0


# ── _advances_open_plan: membership is the TOOLSET, not the tool name ──


def test_advances_open_plan_resolves_toolset_not_tool_name() -> None:
    """A plan tool stays plan-advancing after a rename.

    Membership is the toolset the registry reports, so the write_plan tool
    surface can be renamed under the plugin without silently dropping out of
    the audit predicate.
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
    """Work with no plan-advancing tool is still drift.

    Uses a tool whose toolset is never in the advance set so the assertion
    is independent of which other suites have populated the registry.
    """
    assert task_manager._advances_open_plan(["web_search"]) is False
    assert task_manager._advances_open_plan([]) is False
    assert task_manager._advances_open_plan(None) is False


def test_advances_open_plan_accepts_todo_transitions() -> None:
    """The lifecycle lever itself always counts as advancing the plan."""
    assert task_manager._advances_open_plan(["todo"]) is True


# ── plan completion mark: core writes completed_at when the plan is done ──


def _plan_dir(tmp_path, slug: str = "P"):
    """A plan directory with a live state.json; returns its plan.md ref."""
    plan_dir = tmp_path / f"20260101_000000-{slug}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / task_manager.PLAN_STATE_FILENAME).write_text(
        json.dumps({"plan": {"slug": slug, "title": slug, "status": "completed"}, "spec": {}}),
        encoding="utf-8",
    )
    return plan_dir, plan_dir / "plan.md"


def _store_for(plan_ref, statuses):
    store = TodoStore()
    store.write(
        [
            {"id": str(i + 1), "content": f"T{i}", "status": s, "plan": str(plan_ref)}
            for i, s in enumerate(statuses)
        ]
    )
    return store


def _read_plan(tmp_path, slug: str = "P") -> dict:
    return json.loads(
        (tmp_path / f"20260101_000000-{slug}" / task_manager.PLAN_STATE_FILENAME).read_text()
    )["plan"]


def test_completion_mark_written_when_plan_is_complete(tmp_path) -> None:
    _, plan_ref = _plan_dir(tmp_path)
    task_manager._write_plan_completion_mark(_store_for(plan_ref, ["completed", "completed"]))
    assert _read_plan(tmp_path).get("completed_at")


def test_completion_mark_not_written_for_all_cancelled(tmp_path) -> None:
    _, plan_ref = _plan_dir(tmp_path)
    task_manager._write_plan_completion_mark(_store_for(plan_ref, ["cancelled", "cancelled"]))
    assert "completed_at" not in _read_plan(tmp_path)


def test_completion_mark_open_plan_not_marked(tmp_path) -> None:
    _, plan_ref = _plan_dir(tmp_path)
    task_manager._write_plan_completion_mark(_store_for(plan_ref, ["completed", "pending"]))
    assert "completed_at" not in _read_plan(tmp_path)


def test_completion_mark_not_rewritten(tmp_path) -> None:
    _, plan_ref = _plan_dir(tmp_path)
    store = _store_for(plan_ref, ["completed"])
    task_manager._write_plan_completion_mark(store)
    first = _read_plan(tmp_path)["completed_at"]
    task_manager._write_plan_completion_mark(store)
    assert _read_plan(tmp_path)["completed_at"] == first


def test_completion_mark_fail_closed_on_missing_state(tmp_path) -> None:
    """An unresolvable plan ref reports and writes nothing — it never raises."""
    store = _store_for(tmp_path / "nope" / "plan.md", ["completed"])
    task_manager._write_plan_completion_mark(store)  # must not raise


def test_on_todo_write_writes_completion_mark(tmp_path, monkeypatch) -> None:
    _, plan_ref = _plan_dir(tmp_path)
    store = _store_for(plan_ref, ["completed"])
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    task_manager.on_todo_write(agent, {"action": "complete", "item_id": "1"})
    assert _read_plan(tmp_path).get("completed_at")
