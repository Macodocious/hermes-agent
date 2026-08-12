"""Persistent session task state — the todo store across sessions.

The in-memory TodoStore lives on the AIAgent and dies with it; the gateway
mints a fresh agent per message, so without persistence the task list
reverts to whatever the history scan can replay. This module persists the
store to SessionDB's ``state_meta`` table keyed by ``todo:<session_id>``
(mirroring ``goal:<session_id>`` in hermes_cli/goals.py) and carries it
across the compression boundary via ``migrate_todo_to_session``.

Design notes / invariants:

- The DB row is the source of truth; the history scan in
  ``AIAgent._hydrate_todo_store`` remains the legacy fallback for sessions
  with no row.
- Writes are best-effort and never raise: a broken DB must not wedge the
  agent's turn (mirrors goals.py).
- Migration copies to the child and removes the parent row so exactly one
  active todo row exists per logical conversation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _meta_key(session_id: str) -> str:
    return f"todo:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    Mirrors hermes_cli/goals.py: cache one instance per hermes_home path so
    profile switches still pick up the right DB, and fail soft so tests and
    non-standard launchers can still use the helpers.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("TaskState: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("TaskState: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_todo(session_id: str) -> Optional[Any]:
    """Load the persisted TodoStore for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("TaskState: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        from tools.todo_tool import TodoStore

        return TodoStore.from_json(raw)
    except Exception as exc:
        logger.warning(
            "TaskState: could not parse stored todo for %s: %s", session_id, exc
        )
        return None


def save_todo(session_id: str, store: Any) -> None:
    """Persist a TodoStore to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), store.to_json())
    except Exception as exc:
        logger.debug("TaskState: set_meta failed: %s", exc)


def persist_todo_store(agent: Any) -> None:
    """Write-through helper: persist an agent's in-memory todo store.

    Called by the todo execution paths after a mutating call. Respects
    ``_persist_disabled`` so background forks that share a session_id can
    never clobber the owner's row (the curator-takeover guard), and skips
    agents without a session id or store. Best-effort, never raises.
    """
    if getattr(agent, "_persist_disabled", False):
        return
    session_id = getattr(agent, "session_id", None)
    store = getattr(agent, "_todo_store", None)
    if not session_id or store is None:
        return
    try:
        save_todo(session_id, store)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("TaskState: write-through failed: %s", exc)


def clear_todo(session_id: str) -> bool:
    """Delete the persisted todo row for a session.

    Returns True when a row was removed, False when there was nothing to
    remove or the DB was unavailable. Best-effort and never raises.
    """
    if not session_id:
        return False
    db = _get_session_db()
    if db is None:
        return False
    try:
        if db.get_meta(_meta_key(session_id)) is None:
            return False
        db.delete_meta(_meta_key(session_id))
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("TaskState: clear_todo failed: %s", exc)
        return False


def migrate_todo_to_session(
    old_session_id: str, new_session_id: str, *, reason: str = ""
) -> bool:
    """Carry a persisted todo store from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session,
    but ``load_todo`` does a flat ``todo:<session_id>`` lookup with no
    parent-lineage walk — so an active task list silently dies at the
    compaction boundary (the same hazard goals.py documents for /goal).
    Copy the store onto the new session and remove the old row so exactly
    one active todo row exists per logical conversation.

    Returns True when a store was migrated, False when there was nothing
    to migrate or the DB was unavailable. Best-effort and never raises —
    a failure here must not block compression.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        store = load_todo(old_session_id)
        if store is None:
            return False
        if not store.has_items() and not store.pending_captures():
            # Nothing live to carry; drop the stale row instead of
            # persisting empty state on the child.
            clear_todo(old_session_id)
            return False
        # Don't clobber a store already set on the child (e.g. a resumed
        # lineage that re-established its own task list).
        if load_todo(new_session_id) is not None:
            return False
        save_todo(new_session_id, store)
        clear_todo(old_session_id)
        logger.debug(
            "TaskState: migrated todo %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("TaskState: todo migration failed: %s", exc)
        return False
