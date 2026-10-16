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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plan completion mark
# ---------------------------------------------------------------------------
# When every task carrying a plan ref is terminal AND at least one is
# ``completed``, core records the instant in the plan's state.json. The mark is
# what code-review selects on. The plan store itself is owned by the write-plan
# plugin, so only its path and the two keys of its document are named here.
PLAN_STORE_DIR = Path("/root/.hermes/cache/plans")
PLAN_STATE_FILENAME = "state.json"
PLAN_HALF_KEY = "plan"
COMPLETED_AT_KEY = "completed_at"
PLAN_COMPLETED_STATUS = "completed"

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


def plan_final_row_id(store: Any, plan_ref: str) -> str:
    """The id of the last row (list order) carrying ``plan_ref``, or ''.

    The plan-completion line belongs to the plan's final task, not to every
    task that closes. ``format_notice`` renders it for this row alone, so a
    batch of completions yields one line, at the plan's end, instead of one
    per completing task.
    """
    ref = str(plan_ref or "").strip()
    if not ref:
        return ""
    rows = [
        item
        for item in store.read()
        if str(item.get("plan") or "").strip() == ref
    ]
    return str(rows[-1].get("id") or "") if rows else ""


def task_position(store: Any, item_id: Any) -> tuple:
    """Return the 1-based ``(position, total)`` of a task in the ordered list.

    Position is the row's index in list order (list order is priority), not
    the row id: a replace-mode write renumbers ids to 1..N, so the id is not
    a position a human reader can follow. Returns ``(0, total)`` when the row
    is absent, so a caller renders a line without a position rather than
    failing — the completion line must still ship on a degraded store.

    A row carrying a plan ref is numbered within its plan's rows, so the
    ``N of M`` a notice prints and the ``all M tasks done`` the plan line
    prints are the same M. Numbering across every row let a plan row whose
    ref was missing read as ``task 4 of 4`` while the plan line said ``all 3``.
    """
    rows = store.read()
    target = str(item_id or "").strip()
    ref = ""
    for item in rows:
        if str(item.get("id") or "") == target:
            ref = str(item.get("plan") or "").strip()
            break
    if ref:
        rows = [i for i in rows if str(i.get("plan") or "").strip() == ref]
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


def _now_iso() -> str:
    """The current instant as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _has_completed_task(store: Any, plan_ref: str) -> bool:
    """True iff at least one task carrying ``plan_ref`` is ``completed``.

    The guard that keeps an all-cancelled plan from being marked complete: the
    terminal set counts ``cancelled``, but a plan nobody implemented must not be
    reported as implemented.
    """
    ref = str(plan_ref or "").strip()
    return any(
        str(item.get("plan") or "").strip() == ref
        and str(item.get("status") or "").strip() == PLAN_COMPLETED_STATUS
        for item in store.read()
    )


def _plan_state_path(plan_ref: str) -> Optional[Path]:
    """The plan's state.json, resolved from a todo row's plan ref.

    A task's plan ref is the plan document path (``<plan_dir>/plan.md``); the
    state file sits beside it. A ref that resolves to no plan directory or no
    state file returns None, so the caller reports and writes nothing.
    """
    ref = str(plan_ref or "").strip()
    if not ref:
        return None
    candidate = Path(ref)
    plan_dir = candidate.parent if candidate.name else candidate
    state_path = plan_dir / PLAN_STATE_FILENAME
    return state_path if state_path.is_file() else None


def _stamp_plan_completed_at(plan_ref: str) -> None:
    """Write ``completed_at`` into the plan's state.json, once.

    Fail-closed and idempotent: an unresolvable plan ref, an unreadable or
    malformed state, an absent plan half, or an already-marked plan all leave
    the file untouched and report. Never raises.
    """
    state_path = _plan_state_path(plan_ref)
    if state_path is None:
        logger.warning("task_manager: plan ref %r resolves to no state file", plan_ref)
        return
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("task_manager: plan state unreadable at %s: %s", state_path, exc)
        return
    if not isinstance(document, dict):
        return
    plan_state = document.get(PLAN_HALF_KEY)
    if not isinstance(plan_state, dict):
        logger.warning("task_manager: no %r half at %s", PLAN_HALF_KEY, state_path)
        return
    if plan_state.get(COMPLETED_AT_KEY):
        return
    plan_state[COMPLETED_AT_KEY] = _now_iso()
    try:
        state_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("task_manager: could not write plan state at %s: %s", state_path, exc)
        return
    logger.info("task_manager: marked plan complete: %s", state_path.parent.name)


def _write_plan_completion_mark(store: Any) -> None:
    """Mark every complete plan complete. Fail-closed; never raises.

    Runs after every todo transition. For each plan ref the store carries: when
    ``plan_is_complete`` holds and at least one task is ``completed``, record the
    instant in that plan's state.json. Core writes no other plan field and never
    rewrites plan.md.
    """
    try:
        refs = {str(item.get("plan") or "").strip() for item in store.read()}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: plan mark read failed: %s", exc)
        return
    for ref in refs:
        if not ref:
            continue
        try:
            if not plan_is_complete(store, ref):
                continue
            if not _has_completed_task(store, ref):
                continue
            _stamp_plan_completed_at(ref)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("task_manager: plan completion mark for %r failed: %s", ref, exc)


def _emit_pending_notices(agent: Any) -> None:
    """Hand the store's pending lifecycle notices to the delivery callback.

    The turn-end drain is too late for a start line: a task begun and finished
    inside one turn would report its start next to its completion. This runs
    right after the write that produced the transition, so the line ships at
    the transition itself.

    Only the notices the callback actually took are drained. A callback that
    raises, or that reports a refusal by returning ``False``, leaves that
    notice and everything behind it queued for the turn-end drain — the
    callback's own staleness guard must not be able to destroy a transition
    the store has already recorded.
    """
    callback = getattr(agent, "todo_notice_callback", None)
    if callback is None:
        return
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    delivered = 0
    try:
        from tools.todo_tool import format_notice

        # One notice, one message. Joining them welds a start line under the
        # completion that produced it, and delivers a whole turn's backlog as
        # one block instead of at the transitions themselves.
        for notice in store.pending_notices():
            line = format_notice(notice, store)
            # A notice that renders to nothing has nothing to deliver, so it
            # is spent rather than left to block the queue behind it.
            if line and callback(line) is False:
                break
            delivered += 1
    except Exception as exc:
        logger.debug("task_manager: lifecycle notice emission failed: %s", exc)
    if delivered:
        store.drain_notices(delivered)


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
    _write_plan_completion_mark(store)
    _emit_pending_notices(agent)
    _persist(agent)


def on_tool_call(agent: Any, tool_name: str) -> None:
    """Begin the current task the moment work on it starts.

    The store already knows which task is current the moment the previous one
    ends (the ``complete`` promotion). This is the same fact at the other end
    of the turn: work is what begins a task, so the start is taken from it
    rather than waiting on an agent ``begin`` that arrives after the work is
    done — which is how a turn that defers every transition to its end emits
    the whole lifecycle as one burst, with the start line beside the
    completion instead of at the task's real begin.

    One task executes at a time: while a task is ``in_progress`` this is a
    no-op, and the store refuses a second ``begin`` regardless. The transition
    is the store's own door, so the notice, the ordering and the persistence
    are the ones every other transition already goes through. When
    ``tasks.lifecycle.enabled`` is false the hook is a no-op.
    """
    if not _lifecycle_enabled():
        return
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    from tools.registry import registry

    if registry.get_toolset_for_tool(tool_name) not in _WORK_TOOLSETS:
        return
    if _current_item(agent) is not None:
        return
    pending = _items_with_status(store, ("pending",))
    if not pending:
        return
    store.transition("begin", pending[0]["id"])
    _write_plan_completion_mark(store)
    _emit_pending_notices(agent)
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
        # stop. The exemption is narrow: it covers the turn that made the
        # transition, not every turn that merely touched the todo tool. A
        # turn that issued a transition and then did unrelated work still
        # has to leave the list current, so the flag is cleared as it is
        # read — the next turn must earn its own exemption.
        agent._task_lifecycle_action_issued = False
        return None
    # R1: an open task is no longer a blanket exemption. The plan is the
    # work contract — when the open task carries a plan, a substantive
    # turn whose tool calls do not advance that plan is drift and becomes
    # nudgeable (kills the D1 hole: audit silent while any task is open).
    # Without a plan ref there is no contract yet to drift from, so the
    # single-item exemption stands (simple tasks run unchanged).
    #
    # R2: the open task must be the one the turn advanced. A turn that
    # works while a *different* task is open — or that leaves the open task
    # untouched — is the end-of-plan burst in miniature: the list is not
    # current, so the turn does not end cleanly.
    current = _current_item(agent)
    if current is not None:
        if not _advances_open_plan(tool_names or []):
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

# Toolsets whose calls are work on the machine, so a call in one of them is the
# moment the current task begins. ``todo`` and ``write_plan`` are excluded: the
# transitions themselves and the plan authoring that precedes them are
# bookkeeping, not work on a task. Resolved by toolset through the registry for
# the same reason _ADVANCE_TOOLSETS is — a toolset is the stable contract.
_WORK_TOOLSETS = frozenset({"file", "terminal"})


def _advances_open_plan(tool_names: Optional[list]) -> bool:
    """True when the turn's tool calls show work on the open task's plan."""
    from tools.registry import registry

    return any(
        registry.get_toolset_for_tool(name) in _ADVANCE_TOOLSETS
        for name in tool_names or []
    )
