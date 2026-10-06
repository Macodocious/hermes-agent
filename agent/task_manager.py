"""Task lifecycle manager — deterministic owner of the todo task state machine.

The task lifecycle (P1/P2) makes the todo list a physical, code-enforced
task state machine: the agent works on exactly one task at a time, every
transition goes through the ``todo`` tool's lifecycle actions, and the
GoalEngine loop (armed on ``begin``) pulls the agent back after every turn
until the task is done. This module owns the deterministic side of that
contract:

- ``on_todo_write`` — hooks the todo dispatch point (agent_runtime_helpers)
  so every lifecycle transition arms or clears the GoalEngine loop and
  persists the store.
- ``audit_turn_end`` — hooks the turn finalizer (turn_finalizer) so a turn
  that did work without an open task cannot end cleanly: the loop pulls the
  agent back with a continuation nudge.
- ``observe_verdict`` — hooks the goal-loop paths (gateway/run.py, cli.py,
  tui_gateway/server.py) so the judge's ``done`` verdict is the second key
  of the two-key close: a ``closing`` task finalizes to ``completed``; a
  task still ``in_progress`` when the judge says done gets one explicit
  nudge to close it, then finalizes.

Everything here is deterministic code — no LLM calls, no model discretion.
The agent is the only worker; the GoalEngine only checks and pulls back.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Continuation nudges injected by the lifecycle (mirrors the goal-loop
# continuation pattern). The finalize nudge is the one-shot "judge says
# done but the task is still open" prompt; the audit nudge is the
# "you did work with no open task" pull-back.
LIFECYCLE_FINALIZE_NUDGE = (
    "[The work looks complete, but the task is still open]\n"
    "Reason: {reason}\n\n"
    "If the task is genuinely done, call todo with action=close and "
    "item_id={item_id} now. If something still blocks completion, call "
    "todo with action=escalate and item_id={item_id} instead."
)

LIFECYCLE_AUDIT_NUDGE = (
    "[You did work this turn, but no task is open]\n"
    "Every task must be started with todo action=begin before work, and "
    "ended with action=close (or pause/escalate) when you stop. Begin the "
    "task you were working on, or explain why no task applies."
)

# The nudge delivered when a plan-carrying task finalizes while siblings of
# the same plan remain (R3 plan-level completion). The plan, not the todo
# list, is the unit of approved work — finishing one item must not strand
# the rest of a multi-item plan ("items 4-7 still pending" failure). The
# agent is pulled back to begin the next sibling.
LIFECYCLE_PLAN_NEXT_NUDGE = (
    "[Plan item {item_id} is done — more plan items remain]\n"
    "Plan: {plan}\n"
    "Next pending item: {next_id}: {next_content}\n\n"
    "Continue the approved plan: call todo with action=begin and "
    "item_id={next_id} now. If the remaining plan items are no longer "
    "wanted, mark them cancelled instead."
)

# Task-status groupings for the one status-query owner (_items_with_status).
# Single-sourced here so the lifecycle's notion of "current" and "closing"
# cannot drift between the helpers that ask the same question.
CURRENT_TASK_STATUSES: tuple = ("in_progress",)
CLOSING_TASK_STATUSES: tuple = ("closing",)

# The goal text stored for an armed task. The todo item is the task; the
# goal text is its description, so the judge evaluates the same content
# the agent sees in the task list. The task content is the specification
# the review must hold the implementation to — the goal text names it
# explicitly so the verdict is bound to the task's stated objective.
#
# R2 previously bound a plan-carrying item's goal to the whole plan file.
# That contradicted the judge's open-row rule (JUDGE_SYSTEM_PROMPT): a
# multi-item plan keeps siblings open until its last item closes, so every
# plan item's verdict saw open rows and could never clear the judge, while
# each rejection appended a rework row that blocked it harder. The
# lifecycle is task-by-task — the judge evaluates the task's own row — and
# plan completion is an aggregate computed from the store, not a verdict.
# The goal therefore names the bound task and its own content, and never
# inlines the plan.
def _goal_text_for_item(item: Dict[str, Any]) -> str:
    content = str(item.get("content") or "(no description)")
    task_id = str(item.get("id") or "").strip()
    if not task_id:
        return f"Complete the task per its specification: {content}"
    return (
        f"Complete the task per its specification: {content}\n\n"
        f"(Bound task: todo item {task_id}. Your verdict is about this task "
        "alone — the status of any other row in the task store does not bear "
        "on it.)"
    )


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


def _load_goal_manager(agent: Any) -> Any:
    """Return a GoalManager bound to the agent's session, or None.

    Best-effort: a missing goals module or session id must never break a
    turn (mirrors the goal-loop paths).
    """
    session_id = getattr(agent, "session_id", None) or ""
    if not session_id:
        return None
    try:
        from hermes_cli.goals import GoalManager
    except Exception as exc:
        logger.debug("task_manager: goals module unavailable: %s", exc)
        return None
    try:
        block = _lifecycle_config()
        raw_max_turns = block.get("max_turns", 0)
        max_turns = int(raw_max_turns or 0)
    except (TypeError, ValueError):
        max_turns = 0
    return GoalManager(session_id=session_id, default_max_turns=max_turns or 20)


def _persist(agent: Any) -> None:
    """Write-through the agent's todo store (best-effort, never raises)."""
    try:
        from hermes_cli.tasks import persist_todo_store

        persist_todo_store(agent)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: persist failed: %s", exc)


def _items_with_status(store: Any, statuses: tuple) -> list:
    """The rows whose status is in ``statuses``, in list order (priority).

    The one owner of the status query. Sites used to recompute
    "which row is current" and "which row is closing" in their own words
    — ``_current_item``, ``_closing_item`` and two inline ``next`` scans —
    with several chances to disagree about the same fact. They now all
    read through here.
    """
    if store is None:
        return []
    return [item for item in store.read() if item["status"] in statuses]


def _current_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single in_progress task, or None."""
    items = _items_with_status(getattr(agent, "_todo_store", None), CURRENT_TASK_STATUSES)
    return items[0] if items else None


def _closing_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single closing task, or None."""
    items = _items_with_status(getattr(agent, "_todo_store", None), CLOSING_TASK_STATUSES)
    return items[0] if items else None


# =============================================================================
# Shared completion predicate (one contract)
# =============================================================================
# The judge's ``done`` verdict is the second key of the two-key close, and
# ``blocked`` vetoes it. Five sites re-derived that rule in their own words;
# they now call one predicate so the two can no longer drift apart. The
# ``blocked`` definition itself is untouched — this predicate CONSUMES it.
#
# ``blocked`` carries three meanings in the judge prompt (goal unachievable /
# blocked on a system lock / awaiting user input). All three make a done
# verdict a parked stop rather than an achievement, so ``blocked`` alone
# disqualifies completion.

# The one verdict string that constitutes completion.
COMPLETION_VERDICT = "done"

# Task rows a ``done`` verdict may finalize. ``closing`` is the canonical
# second key: the agent declared the task finished and the judge agrees.
# ``in_progress`` is the bounded-hostage fallback — the judge says done but
# the agent never closed the task, so the verdict closes it explicitly rather
# than stranding it open forever (LIFECYCLE_FINALIZE_NUDGE). Every other
# status (pending / paused / escalated / completed / cancelled) fails closed:
# a done verdict is not a completion for a row that was never worked.
COMPLETION_TASK_STATUSES = frozenset({"closing", "in_progress"})


def is_completion(
    decision: Dict[str, Any], task_status: Optional[str] = None
) -> bool:
    """True iff a judge decision is a completion for the bound task.

    The single definition of the two-key close, called by every veto site:

    - the verdict must be ``done`` — a continue / wait / skipped verdict is
      not a completion;
    - ``blocked`` must be false — a blocked-awaiting-input done verdict is a
      parked stop, not an achievement;
    - when ``task_status`` is supplied, the bound row must be one a done
      verdict may finalize (``COMPLETION_TASK_STATUSES``). Any other row
      fails closed.

    ``task_status`` is the authoritative form at the two-key close, where the
    caller has already selected the row. A caller with no row to bind — or a
    veto that runs before the store is read — passes ``None``, and the
    predicate then decides on the verdict and ``blocked`` alone.

    Mechanically pure: no LLM, no I/O, no store access. The caller supplies
    the row status so the same predicate serves the live-agent, persisted-
    store, and gateway paths alike.
    """
    if str(decision.get("verdict") or "").strip() != COMPLETION_VERDICT:
        return False
    if decision.get("blocked"):
        return False
    if task_status is None:
        return True
    return str(task_status).strip() in COMPLETION_TASK_STATUSES


# Task rows that will never advance again without new input. A task in any
# other status (pending / in_progress / closing / paused / escalated) is
# still live and keeps its plan open.
TERMINAL_TASK_STATUSES = frozenset({"completed", "cancelled"})


def plan_is_complete(store: Any, plan_ref: str) -> bool:
    """True iff every task carrying ``plan_ref`` has reached a terminal state.

    The plan is the unit of approved work; the lifecycle itself is
    task-by-task. This is the *mechanical* definition of "the plan is
    complete" — a set predicate over the store rows that carry the plan
    ref. It is deliberately not a judge verdict and not a second goal: the
    judge decides one task's own row (``is_completion``), and plan
    completion is derived afterwards from what the store actually holds.

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

    Called after every todo tool execution (read or write). Arms the
    GoalEngine loop when a task begins, stays armed while a close is in
    flight (the judge's done verdict is the second key — clearing here
    would strand the task in closing forever), clears it when the task
    leaves in_progress via pause/escalate or finalizes, and persists the
    store. Deterministic: the goal is a mirror of the task state, never a
    separate decision.

    A lifecycle action issued this turn is stamped on the agent so the
    turn-end audit knows the turn ended with a legitimate transition
    (pause/escalate/close) rather than a silent stop. When
    ``tasks.lifecycle.enabled`` is false the hook is a no-op (legacy
    todo behavior).
    """
    if not _lifecycle_enabled():
        return
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return
    if args.get("action") is not None:
        agent._task_lifecycle_action_issued = True
    # ``block`` is not a status transition — it parks the loop. Handle it
    # before the arm/clear logic: the task stays in_progress, so falling
    # through would let the arm branch run and (on a changed goal text)
    # rebuild fresh state, silently dropping the park. Only the guard
    # below keeps the parked state intact on later writes.
    #
    # The action is normalized exactly as the tool normalizes it (strip +
    # lower). The tool accepts ``Block``/``BLOCK`` and parks the store, so an
    # exact-match here silently skipped the park for any variant casing —
    # the store transitioned, the loop was never parked, and the agent
    # continued against a block it had just declared.
    if str(args.get("action") or "").strip().lower() == "block":
        mgr = _load_goal_manager(agent)
        if mgr is not None:
            try:
                # park() returns True only when a park actually landed.
                # A no-op (no goal state loaded) must be logged, never
                # silent: an unlanded park is exactly the "block, yet
                # immediately continue" failure.
                if not mgr.park(str(args.get("reason") or "")):
                    logger.warning(
                        "task_manager: block parked nothing for session %s "
                        "(no active goal state) — the loop will not be held",
                        getattr(agent, "session_id", "") or "",
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("task_manager: goal park failed: %s", exc)
        _persist(agent)
        return
    current = _current_item(agent)
    closing = _closing_item(agent)
    mgr = _load_goal_manager(agent)
    target = current if current is not None else closing
    if target is not None:
        # A task is in_progress, or a close is in flight: the loop must
        # stay armed. A closing task MUST stay armed — the judge's done
        # verdict is the second key, and clearing here would strand the
        # task in closing forever. set() also covers close-from-paused,
        # where pause already cleared the goal; re-arming keeps the goal
        # text in sync with the item content.
        #
        # Guard the re-arm: set() builds a fresh state with turns_used=0
        # and created_at=now, so calling it on every todo write (including
        # the routine read-back the agent issues each turn) resets the
        # judge's turn budget and the loop can never hit max_turns — it
        # spins forever on "(1/max_turns)" messages. Only set() when there
        # is no active goal or the goal text changed; otherwise leave the
        # armed state untouched so the turn counter accumulates and the
        # budget fires. A goal the judge parked (paused/waiting/done) is
        # not active, so it re-arms on the next work turn as before.
        try:
            if mgr is not None:
                goal_text = _goal_text_for_item(target)
                state = getattr(mgr, "state", None)
                already_armed = (
                    state is not None
                    and getattr(state, "status", None) == "active"
                    and getattr(state, "goal", "") == goal_text
                )
                # A parked goal is not re-armed even when the goal text
                # moved (the agent edited the blocked item): the park is
                # the user's to release, and set() would rebuild fresh
                # state with awaiting_user_input=False — releasing the
                # barrier behind the user's back.
                parked = state is not None and getattr(
                    state, "awaiting_user_input", False
                )
                if not already_armed and not parked:
                    # The goal is stamped as a task-lifecycle goal at
                    # arming time. Scope decisions (gateway suppression,
                    # wait bypass, completion line) read this marker on
                    # the goal state itself — goal text is never parsed
                    # for lifecycle identification.
                    mgr.set(
                        goal_text,
                        lifecycle=True,
                        bound_task_id=str(target.get("id") or ""),
                    )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("task_manager: goal arm failed: %s", exc)
    else:
        # No open task and no close in flight: the loop must not run.
        # clear() is a no-op when no goal is set.
        try:
            if mgr is not None:
                mgr.clear()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("task_manager: goal clear failed: %s", exc)
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
    action closed the turn. Returns None when the turn is clean.

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
        # The turn ended with a legitimate transition (pause/close/
        # escalate) — the lifecycle is in control, not a silent stop.
        return None
    if _closing_item(agent) is not None:
        # A close is in flight; the judge's verdict decides the outcome.
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


def _bound_task_id_for_session(session_id: str) -> Optional[str]:
    """The todo row the session's lifecycle goal is bound to, or None.

    The judge's verdict is about one row; the goal state records which.
    Best-effort: a missing goals module or state is None, and the
    two-key close then falls back to its unbound behaviour.
    """
    if not session_id:
        return None
    try:
        from hermes_cli.goals import load_goal

        state = load_goal(session_id)
        return str(getattr(state, "bound_task_id", None) or "").strip() or None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: bound task lookup failed: %s", exc)
        return None


def observe_verdict(agent: Any, decision: Dict[str, Any]) -> Optional[str]:
    """Observe a goal-loop verdict and drive the two-key close (agent path).

    Called from the goal-loop paths that still hold the live agent (CLI,
    TUI). See ``_apply_verdict`` for the state machine. Persists whenever
    the store is present — the verdict may have finalized a task, and the
    write-through must survive the per-message agent rebuild. When
    ``tasks.lifecycle.enabled`` is false the observation is a no-op.
    """
    if not _lifecycle_enabled():
        return None
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return None
    decision.setdefault(
        "bound_task_id", _bound_task_id_for_session(getattr(agent, "session_id", "") or "")
    )
    nudge = _apply_verdict(store, decision)
    _persist(agent)
    return nudge


def observe_verdict_for_session(
    session_id: str, decision: Dict[str, Any]
) -> Optional[str]:
    """Observe a goal-loop verdict from the persisted store (gateway path).

    The gateway mints a fresh agent per message, so the post-turn hook has
    no live agent — load the store from SessionDB, apply the verdict, and
    persist. Best-effort: a missing row or DB failure is a no-op. When
    ``tasks.lifecycle.enabled`` is false the observation is a no-op.
    """
    if not _lifecycle_enabled():
        return None
    if not session_id:
        return None
    try:
        from hermes_cli.tasks import load_todo, save_todo

        store = load_todo(session_id)
        if store is None:
            return None
        decision.setdefault("bound_task_id", _bound_task_id_for_session(session_id))
        nudge = _apply_verdict(store, decision)
        save_todo(session_id, store)
        return nudge
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: session verdict failed: %s", exc)
        return None


def _apply_verdict(store: Any, decision: Dict[str, Any]) -> Optional[str]:
    """Apply a judge verdict to a todo store (the two-key close core).

    The judge's ``done`` verdict is the second key:

    - task ``closing`` + verdict ``done`` → finalize to ``completed``.
    - task ``in_progress`` + verdict ``done`` → finalize and return one
      explicit nudge (the agent never closed it; the nudge tells it the
      task is recorded done).
    - task ``closing`` + verdict ``continue`` → back to
      ``in_progress`` (a review rejection; the review is now owned by the
      code-review plugin, which reports out of band).
    - task ``closing`` + verdict ``wait`` → back to ``in_progress``
      (a park, not a rejection).

    Returns a continuation nudge when one is needed, else None.

    Side effect: ``decision["lifecycle_finalized_id"]`` is set to the id of
    the task this verdict actually finalized, else ``None``. A ``done``
    verdict held by the finalization hold, or one that judged a task still
    open after a plan-level continue, finalizes nothing — the gateway reads
    this to emit ``✅ Task completed`` only when the store record truly
    reached ``completed``, never off the raw verdict.
    """
    # A blocked-awaiting-input done verdict is a parked stop, not a
    # completion: the task stays in_progress and the user's next message
    # re-arms the loop. Never finalize, never nudge, never review.
    decision["lifecycle_finalized_id"] = None
    if decision.get("blocked"):
        return None
    verdict = str(decision.get("verdict") or "").strip()

    def _plan_next_nudge(item: Dict[str, Any]) -> Optional[str]:
        """R3 plan-level completion: after a plan-carrying item finalizes,
        return a continuation nudge naming the next live sibling of the
        same plan. The plan is the unit of approved work — finalizing one
        item must not strand the rest of a multi-item plan.

        The aggregate predicate is authoritative: ``plan_is_complete``
        decides whether the plan still has work, so a sibling sitting in
        ANY non-terminal status keeps the plan open and is named — not
        just the ``pending``/``paused`` rows a bare sibling scan would
        catch. None when the plan is complete or the item has no plan ref.
        """
        plan_ref = str(item.get("plan") or "").strip()
        if not plan_ref:
            return None
        if plan_is_complete(store, plan_ref):
            return None
        siblings = [
            i
            for i in store.read()
            if str(i.get("plan") or "").strip() == plan_ref
            and str(i.get("status") or "").strip() not in TERMINAL_TASK_STATUSES
            and i["id"] != item["id"]
        ]
        if not siblings:
            return None
        nxt = siblings[0]
        return LIFECYCLE_PLAN_NEXT_NUDGE.format(
            item_id=item["id"],
            plan=plan_ref,
            next_id=nxt["id"],
            next_content=str(nxt.get("content") or "(no description)"),
        )

    closing = next((i for i in store.read() if i["status"] == "closing"), None)
    if closing is not None:
        if is_completion(decision, closing["status"]):
            # The done verdict is the second key for the closing task
            # only. The begin pivot and the write-path invariant make a
            # concurrent in_progress task impossible, so nothing else is
            # finalized here — the lifecycle stays strictly sequential
            # (replaces the PR #66 overlap chain, which auto-closed a
            # task the model had begun before the judge cleared the
            # closing one).
            store.finalize(closing["id"])
            decision["lifecycle_finalized_id"] = closing["id"]
            return _plan_next_nudge(closing)
        # Judge says not done. Before reading that as THIS task's review
        # failure, confirm the verdict actually judged this task. The
        # judge is bound to one row — the goal text names it and
        # ``bound_task_id`` (stamped at arming) carries that binding onto
        # the decision. A verdict about a DIFFERENT row — a plan-level
        # continue raised while a sibling is open — must never reopen a
        # task that was not its subject and must never append a rework
        # row for it. The closing task keeps its close in flight; the
        # verdict that judges it is the one that reopens or finalizes it.
        bound = str(decision.get("bound_task_id") or "").strip()
        if bound and bound != str(closing["id"]).strip():
            return None
        # Judge says not done: the close was premature — back to work.
        # A continue verdict is a review rejection and a wait verdict is a
        # park; both return the task to in_progress. The rework task and
        # its nudge were removed with the async review — the review is now
        # owned by the code-review plugin, which reports out of band.
        store.transition("resume", closing["id"])
        return None
    current = next((i for i in store.read() if i["status"] == "in_progress"), None)
    if current is not None and is_completion(decision, current["status"]):
        # Judge believes the work is done but the agent never closed the
        # task. Finalize (bounded hostage risk) and tell the agent.
        # finalize only accepts closing tasks, so move it through the
        # close transition first.
        store.transition("close", current["id"])
        store.finalize(current["id"])
        decision["lifecycle_finalized_id"] = current["id"]
        plan_nudge = _plan_next_nudge(current)
        if plan_nudge:
            return plan_nudge
        return LIFECYCLE_FINALIZE_NUDGE.format(
            reason=str(decision.get("reason") or "judge says done"),
            item_id=current["id"],
        )
    return None
