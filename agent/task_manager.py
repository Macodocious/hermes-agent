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

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# JSON-object extraction for the review verdict parser (mirrors the goal
# judge's tolerant parse: the model may wrap JSON in prose or fences).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

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
    review_nudge = _maybe_review(
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
        review_nudge = _maybe_review(session_id, store, before, decision)
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
# Post-close review (P6) — the mandatory second set of eyes
# =============================================================================
# The judge's ``done`` verdict is self-certification: the same model that
# did the work decides it is done. The review step makes the loop
# self-correcting — a parallel completion call over a frozen packet
# (implementation + the plan it was judged against) whose verdict is
# mechanically enforced:
#
#   - ``pass``        → nothing; the task stays completed.
#   - ``needs_work``  → task_manager mechanically spawns a fix task
#                       (source=review, lineage review_of:<id>) and nudges
#                       it; it queues behind whatever is current.
#   - no plan         → the review does not run and the close is
#                       mechanically held (fail-closed: never a silent
#                       review-without-a-plan).
#   - lineage depth   → past tasks.review.max_rounds the fix task is
#                       mechanically escalated (non-negotiable; prevents a
#                       runaway review → fix → review machine).
#
# The plan half of the packet resolves in order: (1) an explicit ``plan``
# ref on the item (file path or URL — snapshotted, hash recorded);
# (2) the plan is the task itself (item content self-contained); (3) the
# plan is inline in the conversation (the seed captured the origin message
# id; the packet builder pulls that session window verbatim). First that
# resolves wins; all three can coexist. Verdict/lineage/caps are
# code-owned; only findings *content* comes from the model.

REVIEW_SYSTEM_PROMPT = (
    "You are the independent reviewer in a task lifecycle. A task was just "
    "closed as done. You receive two halves: REVIEW_IMPL (the implementation "
    "evidence) and REVIEW_PLAN (the plan or proposal the implementation was "
    "supposed to follow). Judge whether the implementation actually satisfies "
    "the plan. Be strict and specific: name concrete gaps, missing pieces, or "
    "deviations with evidence. Do not rubber-stamp. Respond with JSON only:\n"
    '{"verdict": "pass" | "needs_work", "findings": [string, ...]}\n'
    "findings must be concrete, actionable statements; empty array when pass."
)

REVIEW_USER_PROMPT_TEMPLATE = (
    "REVIEW_IMPL\n"
    "===========\n"
    "{impl}\n\n"
    "REVIEW_PLAN\n"
    "===========\n"
    "{plan}\n\n"
    "Verdict: does the implementation satisfy the plan? Respond with the JSON "
    "shape only."
)

# Defaults for the review completion call (mirror the goal judge's house
# values; configurable under auxiliary.task_review).
REVIEW_MAX_TOKENS = 1024
REVIEW_TIMEOUT = 30.0

# Cap on the review lineage depth (tasks.review.max_rounds). Past the cap
# a needs_work verdict mechanically escalates the fix task instead of
# spawning another review round.
REVIEW_MAX_ROUNDS_DEFAULT = 2

# A task item's content counts as a self-contained plan (rule 2) only when
# it is substantial. A short content is a task title, not a plan — the plan
# then lives inline in the conversation (rule 3, anchored on the
# seed-captured origin message id). Mechanical discriminator: no model
# discretion, no natural-language parsing.
REVIEW_PLAN_MIN_CHARS = 200


def _review_config() -> Dict[str, Any]:
    """The tasks.review config block (best-effort, never raises)."""
    try:
        from hermes_cli.config import load_config as _load_config

        cfg = _load_config() or {}
        tasks_cfg = cfg.get("tasks") if isinstance(cfg, dict) else None
        if isinstance(tasks_cfg, dict):
            block = tasks_cfg.get("review")
            if isinstance(block, dict):
                return block
    except Exception:  # pragma: no cover - defensive
        pass
    return {}


def _review_enabled() -> bool:
    """Whether the post-close review is active (tasks.review.enabled)."""
    return bool(_review_config().get("enabled", True))


def _review_max_rounds() -> int:
    """The review lineage depth cap (tasks.review.max_rounds)."""
    try:
        raw = _review_config().get("max_rounds", REVIEW_MAX_ROUNDS_DEFAULT)
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return REVIEW_MAX_ROUNDS_DEFAULT


def _open_item_ids(store: Any) -> set:
    """Ids of items that were open (in_progress/closing) before a verdict."""
    return {
        item["id"]
        for item in store.read()
        if item["status"] in ("in_progress", "closing")
    }


def _maybe_review(
    session_id: str, store: Any, before: set, decision: Dict[str, Any]
) -> Optional[str]:
    """Run the post-close review when a task just finalized (done path).

    Called from the verdict observation paths after ``_apply_verdict``.
    ``before`` is the set of open item ids captured before the verdict was
    applied; a review fires only when a task that was open is now
    ``completed`` — the single choke point where a task actually
    finalizes. Returns a continuation nudge when a fix task was spawned
    (the loop pulls the agent back to it), else None. Deterministic and
    fail-closed: no plan → no review; transport failure → no review; the
    close is never undone by a review failure.
    """
    if not _review_enabled():
        return None
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
    packet = _plan_payload(session_id, store, item)
    if packet is None:
        # Fail-closed: no plan resolves — the review does not run and the
        # close is mechanically held (the task stays completed; the
        # absence of a review is logged, never silent).
        logger.info(
            "task_manager: review skipped for task %s — no plan resolves",
            item["id"],
        )
        return None
    verdict, findings = _run_review(packet)
    if verdict != "needs_work" or not findings:
        return None
    return _spawn_fix_task(session_id, store, item, findings)


def _plan_payload(
    session_id: str, store: Any, item: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Assemble the review packet for a finalized task, or None.

    The packet carries the implementation half (the in-progress session
    window — the conversation from the plan exchange through the close —
    plus the git diff when the session's repo is in scope) and the plan
    half, resolved in order: explicit ``plan`` ref (file path or URL,
    snapshotted with a content hash) → the item content itself
    (self-contained plan) → the inline conversation window anchored on the
    seed-captured origin message id. First that resolves wins; all three
    can coexist. Returns None when no plan resolves (fail-closed).

    A review fix task (``review_of`` set) inherits the plan of its root
    ancestor: the fix is judged against the same spec its parent was.
    """
    root = _review_root_item(store, item)
    plan = _resolve_plan(session_id, root)
    if plan is None:
        return None
    impl = _impl_payload(session_id, item)
    return {
        "task_id": item["id"],
        "plan": plan,
        "impl": impl,
    }


def _review_root_item(store: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    """Walk the review_of lineage to the root ancestor (or the item itself)."""
    by_id = {i["id"]: i for i in store.read()}
    current = item
    seen = set()
    while True:
        parent_id = str(current.get("review_of") or "").strip()
        if not parent_id or parent_id in seen:
            return current
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return current
        current = parent


def _resolve_plan(session_id: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the plan half of the review packet (rule 1 → 2 → 3)."""
    ref = str(item.get("plan") or "").strip()
    if ref:
        snapshot = _snapshot_plan_ref(ref)
        if snapshot is not None:
            return {"source": "ref", "ref": ref, "content": snapshot}
    content = str(item.get("content") or "").strip()
    if len(content) >= REVIEW_PLAN_MIN_CHARS:
        return {"source": "item", "content": content}
    origin = str(item.get("origin") or "").strip()
    if origin:
        window = _inline_plan_window(session_id, origin)
        if window:
            return {"source": "inline", "origin": origin, "content": window}
    return None


def _snapshot_plan_ref(ref: str) -> Optional[str]:
    """Snapshot an explicit plan ref: local file → content; URL → fetched text.

    Best-effort and bounded: a missing file, failed fetch, or oversized
    content yields None (the resolution falls through to the next rule).
    """
    try:
        if ref.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(ref, timeout=10) as response:
                raw = response.read(200_000).decode("utf-8", errors="replace")
            return raw[:200_000]
        path = Path(ref).expanduser()
        if path.is_file():
            raw = path.read_text(encoding="utf-8", errors="replace")
            return raw[:200_000]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: plan ref snapshot failed: %s", exc)
    return None


def _inline_plan_window(session_id: str, origin: str) -> Optional[str]:
    """Pull the conversation window from the origin message to the close.

    The seed captured the triggering message id (the plan exchange); the
    packet builder fetches that session window verbatim from SessionDB so
    the reviewer sees the proposal the implementation was judged against.
    Best-effort: a missing DB or unknown origin id yields None.
    """
    try:
        from hermes_cli.tasks import _get_session_db

        db = _get_session_db()
        if db is None:
            return None
        messages = db.get_messages(session_id, limit=512) or []
        start = None
        for i, message in enumerate(messages):
            if str(message.get("platform_message_id") or "") == origin:
                start = i
                break
        if start is None:
            return None
        window = messages[start:]
        lines = []
        for message in window:
            role = str(message.get("role") or "?")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"[{role}] {content[:2000]}")
        if not lines:
            return None
        return "\n".join(lines)[:200_000]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: inline plan window failed: %s", exc)
        return None


def _impl_payload(session_id: str, item: Dict[str, Any]) -> str:
    """The implementation half: the in-progress session window + git diff.

    The window is the conversation from the origin message (or the start
    of the session when no origin was captured) through the close — the
    tool calls and results that constitute the implementation. When the
    session's repo is in scope (git_repo_root recorded), the uncommitted
    diff is appended so the reviewer sees the actual code changes.
    Best-effort: any failure degrades to the task content alone.
    """
    parts = []
    window = _session_window(session_id, item)
    if window:
        parts.append("SESSION WINDOW (implementation evidence):\n" + window)
    diff = _git_diff(session_id)
    if diff:
        parts.append("GIT DIFF (uncommitted changes):\n" + diff)
    if not parts:
        parts.append(
            "No session window or git diff available; the task content is: "
            + str(item.get("content") or "(no description)")
        )
    return "\n\n".join(parts)


def _session_window(session_id: str, item: Dict[str, Any]) -> Optional[str]:
    """The conversation window from the origin (or session start) to now."""
    try:
        from hermes_cli.tasks import _get_session_db

        db = _get_session_db()
        if db is None:
            return None
        messages = db.get_messages(session_id, limit=512) or []
        origin = str(item.get("origin") or "").strip()
        start = 0
        if origin:
            for i, message in enumerate(messages):
                if str(message.get("platform_message_id") or "") == origin:
                    start = i
                    break
        lines = []
        for message in messages[start:]:
            role = str(message.get("role") or "?")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"[{role}] {content[:2000]}")
        if not lines:
            return None
        return "\n".join(lines)[:200_000]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("task_manager: session window failed: %s", exc)
        return None


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


def _run_review(packet: Dict[str, Any]) -> tuple:
    """Run the parallel review completion call over the frozen packet.

    Mirrors the house judge pattern (goals.judge_goal): route through
    ``call_llm(task=\"task_review\", temperature=0, ...)``, parse the JSON
    verdict from ``resp.choices[0].message.content``, and fail open on
    transport — a review failure never blocks the close. Returns
    ``(verdict, findings)`` with verdict ``\"pass\"`` or ``\"needs_work\"``.
    """
    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("task_manager: review client import failed: %s", exc)
        return "pass", []
    prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
        impl=str(packet.get("impl") or "")[:200_000],
        plan=str(packet.get("plan") or "")[:200_000],
    )
    try:
        resp = call_llm(
            task="task_review",
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=REVIEW_MAX_TOKENS,
            timeout=REVIEW_TIMEOUT,
        )
    except Exception as exc:
        logger.info("task_manager: review call failed (%s) — failing open", exc)
        return "pass", []
    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""
    return _parse_review_response(raw)


def _parse_review_response(raw: str) -> tuple:
    """Parse the reviewer's JSON reply. Fail-open on unusable output."""
    if not raw:
        return "pass", []
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return "pass", []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return "pass", []
    if not isinstance(data, dict):
        return "pass", []
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict != "needs_work":
        return "pass", []
    findings = data.get("findings")
    if not isinstance(findings, list):
        return "needs_work", []
    cleaned = [str(f).strip() for f in findings if str(f).strip()]
    return ("needs_work", cleaned) if cleaned else ("pass", [])


def _spawn_fix_task(
    session_id: str, store: Any, item: Dict[str, Any], findings: list
) -> Optional[str]:
    """Mechanically spawn the fix task for a needs_work verdict.

    The fix task is born ``pending`` (source=review, lineage
    review_of:<id>) and nudged — it queues behind whatever is current per
    the pivot rule. Past the lineage depth cap (tasks.review.max_rounds)
    the fix task is mechanically escalated instead (non-negotiable;
    prevents a runaway review → fix → review machine). Returns the
    continuation nudge when a fix task was spawned, else None.
    """
    root = _review_root_item(store, item)
    depth = _review_lineage_depth(store, root["id"])
    if depth >= _review_max_rounds():
        store.write(
            [
                {
                    "id": f"review-{root['id']}-{depth + 1}",
                    "content": (
                        f"[REVIEW ESCALATED] Task {root['id']} failed review "
                        f"round {depth + 1} (cap {_review_max_rounds()}). "
                        f"Findings: {'; '.join(findings)}"
                    ),
                    "status": "escalated",
                    "source": "review",
                    "review_of": root["id"],
                }
            ],
            merge=True,
        )
        logger.info(
            "task_manager: review lineage cap reached for task %s — escalated",
            root["id"],
        )
        return None
    store.write(
        [
            {
                "id": f"review-{root['id']}-{depth + 1}",
                "content": (
                    f"[REVIEW] Task {root['id']} needs work. "
                    f"Findings: {'; '.join(findings)}"
                ),
                "status": "pending",
                "source": "review",
                "review_of": root["id"],
            }
        ],
        merge=True,
    )
    fix = next(
        (i for i in store.read() if i.get("review_of") == root["id"]),
        None,
    )
    if fix is None:
        return None
    return (
        "[The completed task was reviewed and needs work]\n"
        f"Reason: {len(findings)} finding(s) from the post-close review.\n\n"
        f"Begin the fix task {fix['id']} with todo action=begin and "
        "item_id={id} now.".format(id=fix["id"])
    )


def _review_lineage_depth(store: Any, task_id: str) -> int:
    """How many review rounds already exist for this task's lineage.

    Walks the whole chain: a fix task's own review spawns the next fix
    task, so the depth is the number of review_of links reachable from
    the original task (direct children plus their children, recursively).
    """
    items = store.read()
    by_parent: Dict[str, list] = {}
    for item in items:
        parent = str(item.get("review_of") or "").strip()
        if parent:
            by_parent.setdefault(parent, []).append(item["id"])
    depth = 0
    frontier = [task_id]
    while frontier:
        next_frontier = []
        for parent in frontier:
            for child in by_parent.get(parent, []):
                depth += 1
                next_frontier.append(child)
        frontier = next_frontier
    return depth
