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
  nudge to close it, then finalizes. Every finalized task also writes its
  post-close verification probe (the mandatory second set of eyes).

Everything here is deterministic code — no LLM calls, no model discretion.
The agent is the only worker; the GoalEngine only checks and pulls back.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

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

# The nudge delivered when the judge rejects a close (verdict continue/wait
# on a closing task). The task returns to in_progress and a rework task is
# appended so the failure is never lost: the agent is told exactly why the
# review failed and where the rework task sits.
LIFECYCLE_REVIEW_REJECT_NUDGE = (
    "[Task {item_id} was reviewed incomplete]\n"
    "Reason: {reason}\n\n"
    "The task is back in_progress and a rework task (id {rework_id}) was "
    "added to the list. Address the review failure before closing again."
)

# The nudge delivered when a plan-carrying task finalizes while siblings of
# the same plan remain (R3 plan-level completion). The plan, not the todo
# list, is the unit of approved work — finishing one item must not strand
# the rest of a multi-item plan ("items 4-7 still pending" failure). The
# agent is pulled back to begin the next sibling; the per-task
# authorization hold re-arms on that begin, so the user's verdict gates
# each item of the plan.
LIFECYCLE_PLAN_NEXT_NUDGE = (
    "[Plan item {item_id} is done — more plan items remain]\n"
    "Plan: {plan}\n"
    "Next pending item: {next_id}: {next_content}\n\n"
    "Continue the approved plan: call todo with action=begin and "
    "item_id={next_id} now. If the remaining plan items are no longer "
    "wanted, mark them cancelled instead."
)

# Source tag for rework tasks spawned by a rejected review (P6 lineage).
# Code-owned (task_manager), never model-authorable — _validate preserves
# the tag and the review_of parent id so the lineage depth cap and the
# fix-task lookup work.
REVIEW_SOURCE = "review"

# The goal text stored for an armed task. The todo item is the task; the
# goal text is its description, so the judge evaluates the same content
# the agent sees in the task list. The task content is the specification
# the review must hold the implementation to — the goal text names it
# explicitly so the verdict is bound to the task's stated objective.
#
# R2: when the item carries a plan ref (the writing_plan plan.md path,
# resolved by the post-close review), the goal binds the judge to the
# plan file instead of one item line — the verdict evaluates the whole
# plan's criteria, not item 1's content. The plan text is read inline
# (capped) because the judge prompt cannot rely on file access.
def _goal_text_for_item(item: Dict[str, Any]) -> str:
    plan_ref = str(item.get("plan") or "").strip()
    if plan_ref:
        plan_text = _read_plan_text(plan_ref)
        if plan_text:
            return (
                "Complete the task per its plan: "
                f"{plan_ref}\n\n"
                f"{plan_text}\n\n"
                f"(Task spec: {str(item.get('content') or '(no description)')[:200]})"
            )
    return (
        "Complete the task per its specification: "
        f"{item.get('content', '(no description)')}"
    )


# Cap on plan content bound into the goal text so an oversized plan file
# cannot balloon the judge/continuation prompt.
_GOAL_PLAN_TEXT_CAP: int = 4000


def _read_plan_text(plan_ref: str) -> str:
    """Read a plan file's text for goal binding (best-effort, capped)."""
    try:
        path = Path(plan_ref)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")[:_GOAL_PLAN_TEXT_CAP]
        return text
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: plan file read failed: %s", exc)
        return ""


def _spec_ref_for_item(item: Dict[str, Any], plan_ref: str) -> str:
    """Resolve the spec file the post-close probe verifies the task against.

    The spec artifact is named ``<plan>-<spec>.md``, so it cannot be
    derived from the plan ref alone — the item's explicit ``spec`` ref is
    the authoritative source when set. The sibling ``spec.md`` of the
    approved plan is the fallback, which is the layout ``writing_plan``
    produces.
    """
    spec_ref = str(item.get("spec") or "").strip()
    if spec_ref:
        return spec_ref
    if plan_ref:
        return str(Path(plan_ref).parent / "spec.md")
    return ""


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


def _current_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single in_progress task, or None."""
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return None
    for item in store.read():
        if item["status"] == "in_progress":
            return item
    return None


def _next_item_id(store: Any) -> str:
    """Next sequential id for a code-owned append (max existing + 1).

    Mirrors TodoStore._next_item_id so the rework task lands with a
    stable id the nudge can name. Falls back to "1" on an empty list.
    """
    numeric_ids = [
        int(item["id"]) for item in store.read() if str(item["id"]).isdigit()
    ]
    return str(max(numeric_ids, default=0) + 1)


def _closing_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single closing task, or None."""
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return None
    for item in store.read():
        if item["status"] == "closing":
            return item
    return None


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
                    mgr.set(goal_text, lifecycle=True)
                    # Per-task execution authorization (writing_plan
                    # integration): a task begun with a plan ref holds
                    # execution until the user's verdict. Stamped only on
                    # an explicit begin of a plan-carrying item — routine
                    # re-arms (the guard above) and close-in-flight never
                    # re-hold, so an already-authorized task keeps running
                    # and a closing task keeps its two-key flow.
                    if (
                        str(args.get("action") or "") == "begin"
                        and str(target.get("plan") or "").strip()
                    ):
                        try:
                            mgr.hold_authorization()
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("task_manager: authorization hold failed: %s", exc)
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


# Tool names that count as advancing the open task's plan: the lifecycle
# lever (todo transitions) and plan authoring (the write-plan tools). A
# substantive turn whose tool calls contain none of these — while an open
# task carries a plan — is work outside the plan and gets nudged.
_ADVANCE_TOOL_NAMES = frozenset({"todo", "writing_plan", "write_plan"})
_ADVANCE_TOOL_PREFIX = "plan_"


def _advances_open_plan(tool_names: list) -> bool:
    """True when the turn's tool calls show work on the open task's plan."""
    return any(
        name in _ADVANCE_TOOL_NAMES or name.startswith(_ADVANCE_TOOL_PREFIX)
        for name in tool_names
    )


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
    before = _open_item_ids(store)
    nudge = _apply_verdict(store, decision)
    review_nudge = _maybe_probe(
        getattr(agent, "session_id", "") or "", store, before, decision
    )
    _persist(agent)
    return review_nudge or nudge


def observe_verdict_for_session(session_id: str, decision: Dict[str, Any]) -> Optional[str]:
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
        before = _open_item_ids(store)
        nudge = _apply_verdict(store, decision)
        review_nudge = _maybe_probe(session_id, store, before, decision)
        save_todo(session_id, store)
        return review_nudge or nudge
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
      ``in_progress``, a rework task is appended (source=review,
      review_of=<parent id>), and a nudge returns with the judge's
      reason — the review failure is never silent.
    - task ``closing`` + verdict ``wait`` → back to ``in_progress``
      with no rework task (a park, not a rejection).

    Returns a continuation nudge when one is needed, else None.
    """
    # A blocked-awaiting-input done verdict is a parked stop, not a
    # completion: the task stays in_progress and the user's next message
    # re-arms the loop. Never finalize, never nudge, never review.
    if decision.get("blocked"):
        return None
    verdict = str(decision.get("verdict") or "").strip()

    def _plan_next_nudge(item: Dict[str, Any]) -> Optional[str]:
        """R3 plan-level completion: after a plan-carrying item finalizes,
        return a continuation nudge naming the next pending sibling of the
        same plan. The plan is the unit of approved work — finalizing one
        item must not strand the rest of a multi-item plan. None when the
        plan is complete or the item has no plan ref."""
        plan_ref = str(item.get("plan") or "").strip()
        if not plan_ref:
            return None
        siblings = [
            i
            for i in store.read()
            if str(i.get("plan") or "").strip() == plan_ref
            and i["status"] in ("pending", "paused")
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
            return _plan_next_nudge(closing)
        # Judge says not done: the close was premature — back to work.
        # A continue verdict is a review rejection: the task returns to
        # in_progress and a rework task is appended (source=review,
        # review_of=<parent id>) so the agent is pulled back to the
        # failed task with the judge's reason attached. The nudge is
        # enqueued ahead of the goal continuation (gateway/run.py), so
        # the agent sees exactly why the review rejected the close. A
        # wait verdict is a park, not a rejection — the loop resumes
        # automatically when the async thing clears, so no rework task
        # is spawned.
        store.transition("resume", closing["id"])
        if verdict != "continue":
            return None
        reason = str(decision.get("reason") or "the review found the task incomplete").strip()
        rework = {
            # Merge-mode write drops id-less items, so the rework task
            # carries the next sequential id (mirrors _next_item_id).
            "id": _next_item_id(store),
            "content": (
                f"Rework: {closing.get('content', '(no description)')} — "
                f"review failed: {reason}"
            ),
            "status": "pending",
            "source": REVIEW_SOURCE,
            "review_of": closing["id"],
        }
        try:
            store.write([rework], merge=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "task_manager: rework task append failed for task %s: %s",
                closing["id"], exc,
            )
            return LIFECYCLE_REVIEW_REJECT_NUDGE.format(
                item_id=closing["id"],
                reason=reason,
                rework_id="(append failed)",
            )
        rework_id = next(
            (i["id"] for i in store.read() if i.get("review_of") == closing["id"]),
            "(unknown)",
        )
        return LIFECYCLE_REVIEW_REJECT_NUDGE.format(
            item_id=closing["id"],
            reason=reason,
            rework_id=rework_id,
        )
    current = next((i for i in store.read() if i["status"] == "in_progress"), None)
    if current is not None and is_completion(decision, current["status"]):
        # Judge believes the work is done but the agent never closed the
        # task. Finalize (bounded hostage risk) and tell the agent.
        # finalize only accepts closing tasks, so move it through the
        # close transition first.
        store.transition("close", current["id"])
        store.finalize(current["id"])
        plan_nudge = _plan_next_nudge(current)
        if plan_nudge:
            return plan_nudge
        return LIFECYCLE_FINALIZE_NUDGE.format(
            reason=str(decision.get("reason") or "judge says done"),
            item_id=current["id"],
        )
    return None


# =============================================================================
# Post-close verification probe (P6) — the mandatory second set of eyes
# =============================================================================
# The judge's ``done`` verdict is self-certification: the same model that
# did the work decides it is done. The probe step makes the loop
# self-correcting — every finalized task writes a probe entry into
# ~/.hermes/probes/active/ (the probe-runner plugin's queue) that fires
# at the next gateway restart and verifies the implementation:
#
#   - intent (mandatory)  → the auxiliary provider judges whether the
#                           change works as intended, from the change
#                           description and the mechanical evidence.
#   - import (code tasks) → the changed modules import cleanly.
#
# The probe verifies the implementation, never the tests. Only failed
# probes report to #probe-reports — the remediation signal. The write is
# unconditional: every finalized task gets a probe (mandatory for ALL
# tasks); a failed write is logged, never silent.

# Activation for task-close probes: the change is in the tree, so the
# next gateway restart loads it and the sweep fires the probe.
PROBE_ACTIVATION = "gateway_restart"

# Cap on import checks derived from the session's changed files — the
# probe stays small; the intent check is the behavioral core.
PROBE_MAX_IMPORT_CHECKS = 4


def _open_item_ids(store: Any) -> set:
    """Ids of items that were open (in_progress/closing) before a verdict."""
    return {
        item["id"]
        for item in store.read()
        if item["status"] in ("in_progress", "closing")
    }


def _maybe_probe(
    session_id: str, store: Any, before: set, decision: Dict[str, Any]
) -> Optional[str]:
    """Write the post-close verification probe when a task just finalized.

    Called from the verdict observation paths after ``_apply_verdict``.
    ``before`` is the set of open item ids captured before the verdict was
    applied; a probe fires only when a task that was open is now
    ``completed`` — the single choke point where a task actually
    finalizes. The write is unconditional (mandatory for ALL tasks): a
    failed write is logged, never silent. Returns None — the probe is
    deferred verification, so there is no continuation nudge.
    """
    if not session_id:
        return None
    finalized = [
        item
        for item in store.read()
        if item["status"] == "completed" and item["id"] in before
    ]
    if not finalized:
        return None
    item = finalized[0]
    try:
        _write_probe(session_id, item)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "task_manager: probe write failed for task %s: %s",
            item["id"], exc,
        )
    return None


def _write_probe(session_id: str, item: Dict[str, Any]) -> None:
    """Write the probe entry for a finalized task into probes/active/.

    The probe is the verification contract for the change: an ``intent``
    check (mandatory — the auxiliary provider judges whether the change
    works as intended) plus ``import`` checks for code tasks (derived
    from the session's changed files, capped). The probe verifies the
    implementation, never the tests. It fires at the next gateway
    restart and reports only on failure.
    """
    from hermes_constants import get_hermes_home

    active_dir = get_hermes_home() / "probes" / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    # R5: the probe verifies against the attached spec — "did it do what
    # the user approved" instead of "did it do what it said". The spec ref
    # is the item's explicit spec path when set (the spec artifact is
    # named <plan>-<spec>.md, so it is not derivable from the plan ref
    # alone); otherwise the sibling spec.md of the item's approved plan,
    # which is the layout writing_plan produces.
    plan_ref = str(item.get("plan") or "").strip()
    spec_text = ""
    spec_ref = _spec_ref_for_item(item, plan_ref)
    if spec_ref:
        spec_text = _read_plan_text(spec_ref)
    probe = {
        "target": f"task:{item['id']}",
        "change": (
            f"Task {item['id']} finalized as done: "
            f"{str(item.get('content') or '(no description)')[:200]}"
            + (f" Approved plan: {plan_ref}" if plan_ref else "")
        ),
        "activation": PROBE_ACTIVATION,
        "created_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checks": [
            {
                "type": "intent",
                "prompt": (
                    "Verify the completed task's implementation against the "
                    "approved spec — did it do what the user approved, not "
                    "just what the task's own description said. Inspect the "
                    "implementation directly — test results are never "
                    "evidence, and never delegate to test runs. Task: "
                    f"{str(item.get('content') or '(no description)')[:500]}"
                    + (
                        "\nApproved spec:\n" + spec_text
                        if spec_text else ""
                    )
                ),
            }
        ],
        "status": "pending",
    }
    modules = _changed_modules(session_id)
    for module in modules[:PROBE_MAX_IMPORT_CHECKS]:
        probe["checks"].append({"type": "import", "module": module})
    path = active_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_task-{item['id']}.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(probe, handle, sort_keys=False, default_flow_style=False)
    logger.info(
        "task_manager: post-close probe written for task %s (%s)",
        item["id"], path.name,
    )


def _changed_modules(session_id: str) -> list:
    """Derive importable module names from the session's changed files.

    The session's git diff (when a repo is in scope) names the changed
    files; Python files under the repo map to dotted module names. The
    probe's import checks verify those modules import cleanly — the
    mechanical half of the verification contract. Best-effort: no repo
    or no diff yields an empty list (intent alone remains).
    """
    diff = _git_diff(session_id)
    if not diff:
        return []
    modules = []
    for line in diff.splitlines():
        if not line.startswith("diff --git"):
            continue
        parts = line.split(" b/", 1)
        if len(parts) != 2:
            continue
        path = parts[1].strip()
        if not path.endswith(".py") or path.startswith("tests/") or "/tests/" in path:
            continue
        module = path[:-3].replace("/", ".")
        if module not in modules:
            modules.append(module)
    return modules


def _git_diff(session_id: str) -> Optional[str]:
    """The uncommitted diff of the session's repo, when one is in scope."""
    try:
        from hermes_cli.tasks import _get_session_db

        db = _get_session_db()
        if db is None:
            return None
        session = db.get_session(session_id)
        if not session:
            return None
        repo_root = str(session.get("git_repo_root") or "").strip()
        if not repo_root or not Path(repo_root).is_dir():
            return None
        result = subprocess.run(
            ["git", "-C", repo_root, "diff", "--stat", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        stat = result.stdout.strip()
        if not stat:
            return None
        result = subprocess.run(
            ["git", "-C", repo_root, "diff", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        diff = result.stdout.strip()
        if not diff:
            return stat
        return f"{stat}\n\n{diff}"[:200_000]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: git diff failed: %s", exc)
        return None
