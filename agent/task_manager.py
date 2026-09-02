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
                if not already_armed:
                    mgr.set(goal_text)
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
    - task ``closing`` + verdict ``continue``/``wait`` → back to
      ``in_progress``; the loop keeps pulling.

    Returns a continuation nudge when one is needed, else None.
    """
    # A blocked-awaiting-input done verdict is a parked stop, not a
    # completion: the task stays in_progress and the user's next message
    # re-arms the loop. Never finalize, never nudge, never review.
    if decision.get("blocked"):
        return None
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
    probe = {
        "target": f"task:{item['id']}",
        "change": (
            f"Task {item['id']} finalized as done: "
            f"{str(item.get('content') or '(no description)')[:200]}"
        ),
        "activation": PROBE_ACTIVATION,
        "created_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checks": [
            {
                "type": "intent",
                "prompt": (
                    "Verify the completed task's implementation works as "
                    "intended. Inspect the implementation directly — test "
                    "results are never evidence, and never delegate to test "
                    "runs. Task: "
                    f"{str(item.get('content') or '(no description)')[:500]}"
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
