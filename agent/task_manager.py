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

# The goal text stored for an armed task. The todo item is the task; the
# goal text is its description, so the judge evaluates the same content
# the agent sees in the task list.
def _goal_text_for_item(item: Dict[str, Any]) -> str:
    return f"Complete the task: {item.get('content', '(no description)')}"


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


def _closing_item(agent: Any) -> Optional[Dict[str, Any]]:
    """The single closing task, or None."""
    store = getattr(agent, "_todo_store", None)
    if store is None:
        return None
    for item in store.read():
        if item["status"] == "closing":
            return item
    return None


def on_todo_write(agent: Any, args: Dict[str, Any]) -> None:
    """Post-write lifecycle hook for the todo dispatch point.

    Called after every todo tool execution (read or write). Arms the
    GoalEngine loop when a task begins, clears it when the current task
    leaves ``in_progress`` (pause/close/escalate), and persists the store.
    Deterministic: the goal is a mirror of the task state, never a
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
    current = _current_item(agent)
    mgr = _load_goal_manager(agent)
    if current is not None:
        # A task is in_progress: the loop must be armed. set() is
        # idempotent — re-arming on every write keeps the goal text in
        # sync with the item content without resetting the turn budget.
        try:
            if mgr is not None:
                mgr.set(_goal_text_for_item(current))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("task_manager: goal arm failed: %s", exc)
    else:
        # No open task: the loop must not run. clear() is a no-op when no
        # goal is set.
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
    if _current_item(agent) is not None:
        return None
    if _closing_item(agent) is not None:
        # A close is in flight; the judge's verdict decides the outcome.
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
    nudge = _apply_verdict(store, decision)
    _persist(agent)
    return nudge


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
    - task ``closing`` + verdict ``continue``/``wait`` → back to
      ``in_progress``; the loop keeps pulling.

    Returns a continuation nudge when one is needed, else None.
    """
    verdict = str(decision.get("verdict") or "").strip()
    closing = next((i for i in store.read() if i["status"] == "closing"), None)
    if closing is not None:
        if verdict == "done":
            store.finalize(closing["id"])
            return None
        # Judge says not done: the close was premature — back to work.
        store.transition("resume", closing["id"])
        return None
    current = next((i for i in store.read() if i["status"] == "in_progress"), None)
    if current is not None and verdict == "done":
        # Judge believes the work is done but the agent never closed the
        # task. Finalize (bounded hostage risk) and tell the agent.
        # finalize only accepts closing tasks, so move it through the
        # close transition first.
        store.transition("close", current["id"])
        store.finalize(current["id"])
        return LIFECYCLE_FINALIZE_NUDGE.format(
            reason=str(decision.get("reason") or "judge says done"),
            item_id=current["id"],
        )
    return None
