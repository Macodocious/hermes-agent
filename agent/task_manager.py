"""Task lifecycle manager — deterministic owner of the todo task state machine.

The task lifecycle makes the todo list a physical, code-enforced task state
machine the agent owns end to end: the agent works on exactly one task at a
time, every transition goes through the ``todo`` tool's lifecycle actions, and
the agent's own ``complete`` action finalizes the task — there is no judge and
no second key. This module owns the deterministic side of that contract:

- ``on_todo_write`` — hooks the todo dispatch point so a lifecycle write
  persists the store and stamps the turn as having issued a transition.
- ``audit_turn_end`` — hooks the turn finalizer so a turn that did work
  without an open task cannot end cleanly: the loop pulls the agent back with
  a continuation nudge.

The module no longer arms the goal loop and no longer consumes a judge
verdict; task completion is the agent's ``complete`` action, and the store
activates the next pending task mechanically. Everything here is
deterministic code — no LLM calls, no model discretion.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The one continuation nudge the lifecycle still emits: the "you did work with
# no open task" pull-back. Task completion no longer needs a nudge (the
# agent's own ``complete`` action is the whole story), so the judge-era
# finalize and plan-next nudges are gone.
LIFECYCLE_AUDIT_NUDGE = (
    "[You did work this turn, but no task is open]\n"
    "Every task must be started with todo action=begin before work, and "
    "ended with action=complete (or action=cancel with a reason) when you "
    "stop. Begin the task you were working on, or explain why no task "
    "applies."
)

# Task-status grouping for the one status-query owner (_items_with_status).
# Single-sourced here so the lifecycle's notion of "current" cannot drift
# between the helpers that ask the same question.
CURRENT_TASK_STATUSES: tuple = ("in_progress",)


def _lifecycle_config() -> Dict[str, Any]:
    """The tasks.lifecycle config block (best-effort, never raises)."""
    try:
        from hermes_cli.config import load_config as _load_config

        cfg = _load_config() or {}
        tasks_cfg = cfg.get("tasks") if isinstance(cfg, dict) else None
        if isinstance(tasks_cfg, dict):
            block = tasks_cfg.get("lifecycle")
            if isinstance(block, dict):
                return block
    except Exception:  # pragma: no cover - defensive
        pass
    return {}


def _lifecycle_enabled() -> bool:
    """Whether the task lifecycle is active (config tasks.lifecycle.enabled)."""
    return bool(_lifecycle_config().get("enabled", True))


def _persist(agent: Any) -> None:
    """Write-through the agent's todo store (best-effort, never raises)."""
    try:
        from hermes_cli.tasks import persist_todo_store

        persist_todo_store(agent)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: persist failed: %s", exc)


def _items_with_status(store: Any, statuses: tuple) -> list:
    """The rows whose status is in ``statuses``, in list order (priority).

    The one owner of the status query, so the helpers that ask "which row is
    current" cannot disagree about the same fact.
    """
    if store is None:
        return []
    return [item for item in store.read() if item["status"] in statuses]


def _current_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single in_progress task, or None."""
    items = _items_with_status(getattr(agent, "_todo_store", None), CURRENT_TASK_STATUSES)
    return items[0] if items else None


# =============================================================================
# Completion predicate (no longer drives the lifecycle)
# =============================================================================
# The task lifecycle finalizes on the agent's ``complete`` action, not on a
# judge verdict, so nothing here reads a verdict to change task state.
# ``is_completion`` is retained because two callers outside the lifecycle's own
# module still define completion through it — the gateway rejection gate and
# ``hermes_cli/goals`` — and a single definition keeps the verdict/blocked
# rule from drifting between them.

# The one verdict string that constitutes completion.
COMPLETION_VERDICT = "done"


def is_completion(
    decision: Dict[str, Any], task_status: Optional[str] = None
) -> bool:
    """True iff a judge decision is a completion.

    The verdict must be ``done`` and ``blocked`` must be false. ``task_status``
    is accepted for call compatibility; the lifecycle no longer gates
    completion on a row status, so it is not consulted.

    Mechanically pure: no LLM, no I/O, no store access. The caller supplies
    whatever context it holds so the same predicate serves the live-agent and
    gateway paths alike.
    """
    if str(decision.get("verdict") or "").strip() != COMPLETION_VERDICT:
        return False
    return not decision.get("blocked")


# Task rows that will never advance again without new input. A task in any
# other status (pending / in_progress / paused) is still live and keeps its
# plan open.
TERMINAL_TASK_STATUSES = frozenset({"completed", "cancelled"})


def plan_is_complete(store: Any, plan_ref: str) -> bool:
    """True iff every task carrying ``plan_ref`` has reached a terminal state.

    The plan is the unit of approved work; the lifecycle itself is
    task-by-task. This is the *mechanical* definition of "the plan is
    complete" — a set predicate over the store rows that carry the plan
    ref. It is deliberately not a judge verdict and not a second goal: plan
    completion is derived from what the store actually holds.

    A row with no plan ref belongs to no plan; a plan ref that matches no
    row is not complete. Both return False rather than raising, so callers
    can treat this as a pure predicate.
    """
    ref = str(plan_ref or "").strip()
    if not ref:
        return False
    rows = [
        item
        for item in store.read()
        if str(item.get("plan") or "").strip() == ref
    ]
    if not rows:
        return False
    return all(
        str(item.get("status") or "").strip() in TERMINAL_TASK_STATUSES
        for item in rows
    )


def task_position(store: Any, item_id: Any) -> tuple:
    """Return the 1-based ``(position, total)`` of a task in the ordered list.

    Position is the row's index in list order (list order is priority), not
    the row id: a replace-mode write renumbers ids to 1..N, so the id is not
    a position a human reader can follow. Returns ``(0, total)`` when the row
    is absent, so a caller renders a line without a position rather than
    failing — the completion line must still ship on a degraded store.
    """
    rows = store.read()
    target = str(item_id or "").strip()
    for position, item in enumerate(rows, start=1):
        if str(item.get("id") or "") == target:
            return position, len(rows)
    return 0, len(rows)


def task_plan_ref(store: Any, item_id: Any) -> str:
    """Return the plan reference carried by a task row, or ''. Absent row
    and absent reference are the same answer: this task belongs to no plan."""
    target = str(item_id or "").strip()
    if not target:
        return ""
    for item in store.read():
        if str(item.get("id") or "") == target:
            return str(item.get("plan") or "").strip()
    return ""


def plan_task_total(store: Any, plan_ref: str) -> int:
    """Count the task rows carrying ``plan_ref`` — the plan's size, used by
    the user-visible plan-completion line to report how many tasks finished."""
    ref = str(plan_ref or "").strip()
    if not ref:
        return 0
    return sum(
        1
        for item in store.read()
        if str(item.get("plan") or "").strip() == ref
    )


def on_todo_write(agent: Any, args: Dict[str, Any]) -> None:
    """Post-write lifecycle hook for the todo dispatch point.

    The agent owns the transitions — they are applied inside the todo store —
    so this hook only records that a lifecycle action was issued this turn
    (the turn-end audit reads the flag to know the turn ended with a
    legitimate transition rather than a silent stop) and persists the store.

    It never arms or clears the goal loop: the lifecycle no longer runs under
    the judge. When ``tasks.lifecycle.enabled`` is false the hook is a no-op.
    """
    if not _lifecycle_enabled():
        return
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    if args.get("action") is not None:
        agent._task_lifecycle_action_issued = True
    _persist(agent)


def audit_turn_end(
    agent: Any,
    *,
    final_response: Optional[str],
    interrupted: bool,
    tool_call_count: int = 0,
    tool_names: Optional[list] = None,
) -> Optional[str]:
    """Turn-end audit: work without an open task must not end cleanly.

    Returns a continuation nudge (to be enqueued as a user-role message)
    when the turn did substantive work — a real response, tool calls, or
    file mutations — while no task is ``in_progress`` and no lifecycle
    action transitioned a task. Returns None when the turn is clean.

    The audit is the mechanical boundary: the model cannot be prevented
    from acting without the tool, but it cannot get away with it — the
    turn does not end cleanly and the loop pulls it back. When
    ``tasks.lifecycle.enabled`` is false the audit is a no-op.
    """
    if not _lifecycle_enabled():
        return None
    if interrupted:
        return None
    if getattr(agent, "_task_lifecycle_action_issued", False):
        # The turn ended with a legitimate transition (begin/complete/
        # cancel/pause/resume) — the lifecycle is in control, not a silent
        # stop.
        return None
    # R1: an open task is no longer a blanket exemption. The plan is the
    # work contract — when the open task carries a plan, a substantive
    # turn whose tool calls do not advance that plan is drift and becomes
    # nudgeable (kills the D1 hole: audit silent while any task is open).
    # Without a plan ref there is no contract yet to drift from, so the
    # single-item exemption stands (simple tasks run unchanged).
    current = _current_item(agent)
    if current is not None:
        if str(current.get("plan") or "").strip() and not _advances_open_plan(
            tool_names or []
        ):
            return LIFECYCLE_AUDIT_NUDGE
        return None
    store = getattr(agent, "_todo_store", None)
    if store is None or not store.has_items():
        # No task list at all — the lifecycle is not in play this session.
        return None
    if not (final_response or "").strip():
        return None
    # Work evidence: any tool call this turn, or a substantive response.
    if tool_call_count == 0 and len((final_response or "").strip()) < 40:
        # A terse reply with no tool use is a conversational turn, not
        # task work — leave it alone.
        return None
    return LIFECYCLE_AUDIT_NUDGE


# Toolsets that count as advancing the open task's plan: the lifecycle lever
# (todo transitions) and plan authoring. A substantive turn whose tool calls
# contain none of these — while an open task carries a plan — is work outside
# the plan and gets nudged.
#
# Membership is resolved by TOOLSET through the registry, not by tool name.
# The plan-authoring tools are owned by the write_plan plugin and get renamed
# as that surface evolves; the earlier name-set + `plan_` prefix could not
# track a rename and silently stopped matching every plan tool at once, which
# fired the out-of-plan nudge on genuine plan work. A toolset is the stable
# contract.
_ADVANCE_TOOLSETS = frozenset({"todo", "write_plan", "file", "terminal"})


def _advances_open_plan(tool_names: Optional[list]) -> bool:
    """True when the turn's tool calls show work on the open task's plan."""
    from tools.registry import registry

    return any(
        registry.get_toolset_for_tool(name) in _ADVANCE_TOOLSETS
        for name in tool_names or []
    )
