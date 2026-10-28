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
import threading
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
    "Start the task with todo action=begin before the work, and close it with "
    "action=complete (or action=cancel with a reason) when the work is done. "
    "Completing or cancelling the current task makes the next one current by "
    "itself — there is nothing to ask."
)

# The completion hand-off, delivered once per completed task. Same instruction
# the plan's approval sends to start the work: a completion that leaves the
# plan with more work must not end the turn. A plan that is over, and the
# completion that closed its last task, end the turn exactly as before.
LIFECYCLE_CONTINUATION_NUDGE = (
    "[Task lifecycle] Task complete; the plan has more work.\n"
    "Begin the next task immediately — call a tool relevant to its first step "
    "now."
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


# Appended to a tool result when the call did work with no task in progress.
# The turn-end audit speaks only after the turn; this speaks at the call that
# did the work, which is where the transition belongs. It refuses nothing and
# changes no row — the model still owns the transition.
LIFECYCLE_TOOL_REMINDER = (
    "[Task lifecycle] No task is in progress. Start the task you are working on "
    "with todo action=begin before the next change, and close it with "
    "action=complete when its work is done."
)

# The active-task counterpart: the empty case above can only speak when no row
# is current, so a task that stayed current through a long stretch of work
# never asked to be closed — its backlog cleared as one sweep instead of along
# the work. This line names the current row and the tenure's call count on
# every worked call, so the count climbs with the work and a task cannot
# accumulate a pile unseen.
#
# It names the row by its label — its position in the sequence the user counts
# — and never by the store id. This line ships on every tool result, so an id
# here is the number the model quotes back at the user: it reported plan task 2
# as "task 4" because the tracker's id said 4 while the plan said 2.
TASK_PACING_TOOL_REMINDER = (
    "[Task lifecycle] Task {label} \"{content}\" is current — {work_count} "
    "tool calls under it. Close it with todo action=complete when its work is done."
)
# A row outside every plan's sequence has no label and no task number; it is
# named by what it says rather than given a number of its own.
TASK_PACING_TOOL_REMINDER_UNLABELLED = (
    "[Task lifecycle] \"{content}\" is current — {work_count} "
    "tool calls under it. Close it with todo action=complete when its work is done."
)
# A task content can be a paragraph; the line ships on every turn, so its
# quoted portion is capped with an ellipsis instead.
PACING_CONTENT_MAX_CHARS = 60


def _pacing_reminder(store: Any, item: Dict[str, Any], work_count: int) -> str:
    """The pacing line for one active row: label, capped content, tenure count."""
    content = str(item.get("content") or "")
    if len(content) > PACING_CONTENT_MAX_CHARS:
        content = content[:PACING_CONTENT_MAX_CHARS].rstrip() + "…"
    label = store.task_label(item.get("id", ""))
    template = (
        TASK_PACING_TOOL_REMINDER if label else TASK_PACING_TOOL_REMINDER_UNLABELLED
    )
    return template.format(
        label=label,
        content=content,
        work_count=work_count,
    )


def tool_result_reminder(agent: Any, tool_name: str) -> str:
    """One line for a tool result: the empty case, or pacing for the current task.

    Empty when the call is the lifecycle tool itself, when the lifecycle is
    off, or when the session has no task list. With no current task the line
    nags to start one, once per turn. With a current task the line paces that
    task on every worked call: the count is the tenure's non-todo calls, so it
    climbs 1, 2, 3 … as the work piles up, and a task promoted mid-turn starts
    its own count at 1.
    """
    if tool_name == "todo":
        return ""
    if not _lifecycle_enabled():
        return ""
    store = getattr(agent, "_todo_store", None)
    if store is None or not store.has_items():
        return ""
    current = _current_item(agent)
    if current is None:
        if getattr(agent, "_lifecycle_reminder_issued", False):
            return ""
        agent._lifecycle_reminder_issued = True
        return LIFECYCLE_TOOL_REMINDER
    return _pacing_reminder(store, current, store.work_count())


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


# One drain at a time. Two tool threads can finish a transition at the same
# moment; the queue is ordered, so the consumer that empties it must be single
# or two sends race into the transport and the order the store committed is
# lost at the last step.
_NOTICE_DRAIN_LOCK = threading.Lock()


def _await_delivery(result: Any) -> bool:
    """True when the callback's return value says the line was delivered.

    The contract: a boolean is honoured as-is, and anything carrying
    ``result()`` is a send handed to another thread that must finish before the
    next line is issued. Scheduling is not delivery — the gateway callback used
    to return True on the handoff, which is how two lines from one commit
    arrived inverted.
    """
    if result is False:
        return False
    waiter = getattr(result, "result", None)
    if waiter is None:
        return True
    waiter()
    return True


def _emit_pending_notices(agent: Any) -> None:
    """Deliver the store's pending lifecycle notices, one at a time.

    This is the single consumer of the notice queue. It runs right after the
    call that produced the notice — the todo write for a completion or a
    cancellation, the first non-todo call for a start — so a line ships at the
    event itself rather than at the end of the turn. A held final completion
    is released by the turn finalizer.

    Each line is delivered before the next is issued, and a notice leaves the
    queue only once its send has resolved. A send that fails, or a callback
    that reports a refusal by returning ``False``, stops the drain and leaves
    that notice and everything behind it queued — a transition the store has
    already recorded is never destroyed by a delivery failure.
    """
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    callback = getattr(agent, "todo_notice_callback", None)
    if callback is None:
        # No consumer is a fault, not a queue: a notice held here would ship
        # later as a burst, which is the grouping this path exists to prevent.
        pending = store.pending_notices()
        if pending:
            logger.warning(
                "task_manager: %d lifecycle notice(s) with no consumer attached",
                len(pending),
            )
        return
    with _NOTICE_DRAIN_LOCK:
        delivered = []
        try:
            from tools.todo_tool import format_notice

            # One notice, one message. Joining them welds a start line under
            # the completion that produced it. A held line is the turn's final
            # completion: it stays queued while the lines behind it ship, and
            # the turn finalizer releases it.
            for notice in store.pending_notices():
                if notice.get("held"):
                    continue
                line = format_notice(notice, store)
                # A notice that renders to nothing has nothing to deliver, so
                # it is spent rather than left to block the queue behind it.
                if line and not _await_delivery(callback(line)):
                    break
                delivered.append(notice)
        except Exception as exc:
            logger.warning("task_manager: lifecycle notice delivery failed: %s", exc)
        if delivered:
            store.drain_notices(delivered=delivered)


def emit_pending_notices(agent: Any) -> None:
    """Deliver the store's pending lifecycle notices (the single consumer).

    Public entry point for the executor: the first call under a row records
    the start line, and the line ships as that call returns.
    """
    _emit_pending_notices(agent)


def release_held_notices(agent: Any) -> None:
    """Release the turn's held completion lines and deliver them.

    The turn finalizer calls this after the loop, on the normal and
    interrupted paths: a held line is the turn's last completion, and it ships
    when the turn ends whether or not the response was delivered.
    """
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    store.release_held_notices()
    _emit_pending_notices(agent)
    _persist(agent)


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
    action = args.get("action")
    if action is not None:
        agent._task_lifecycle_action_issued = True
        # Which action, not merely that one ran: a completion is the one
        # transition that leaves work behind, so the audit reads it to decide
        # whether the turn may end.
        agent._task_lifecycle_last_action = action
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
    action transitioned a task. A completion inside a live plan is the one
    transition that continues instead: the next task is already current,
    so the turn starts it. Returns None when the turn is clean.

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
        last_action = getattr(agent, "_task_lifecycle_last_action", "")
        agent._task_lifecycle_last_action = ""
        if last_action != "complete":
            return None
        # A completion that leaves a live plan with more work is not a stop:
        # the close promoted the next task to current, so the turn continues
        # on it. Once per completion, by construction — the action is read and
        # cleared here. A plan that is over, a completion that closed the
        # plan's last task (no successor), and a task outside any plan all end
        # the turn exactly as before.
        store = getattr(agent, "_todo_store", None)
        current = _current_item(agent)
        if store is None or current is None:
            return None
        plan_ref = task_plan_ref(store, str(current.get("id") or ""))
        if not plan_ref or plan_is_complete(store, plan_ref):
            return None
        return LIFECYCLE_CONTINUATION_NUDGE
    # An open task is the lifecycle being in control: the current task is set,
    # so the turn is working on it by definition. The audit exists for the
    # other case — substantive work with nothing open — and it must not fire
    # here. Which tool a turn used says nothing about whether it advanced the
    # current task, and a registry lookup that returns None for an unimported
    # tool made genuine work count as none.
    if _current_item(agent) is not None:
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