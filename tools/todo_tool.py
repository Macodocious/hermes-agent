#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides an in-memory task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations. The state
lives on the AIAgent instance (one per session) and is re-injected into
the conversation after context compression events.

Design:
- Single `todo` tool: provide `todos` param to write, omit to read
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Bounds on persisted todo state. The todo list is a planning aid the model
# re-reads after every context-compression event (see format_for_injection),
# so unbounded item content or count defeats the compression it rides through.
# These caps keep a single oversized item (whether authored by the model or
# replayed from caller-supplied history on the API server) from inflating the
# re-injection block. Generous relative to real plans — a todo item is a short
# task description, and active lists are a handful of items, not hundreds.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Upper bound on a single todo tool-result payload accepted during history
# hydration. The gateway/API server replays caller-supplied conversation
# history to rebuild the store, so an oversized forged result is dropped
# before it is parsed and re-injected (see AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"

# Bounds on the captured-request candidate buffer (P3). The buffer is
# re-injected every turn alongside the active list, so it is bounded the
# same way as the item list: a handful of candidates, content capped at the
# same limit as todo items.
MAX_CAPTURED_REQUESTS = 8
# Lifecycle states for a captured request candidate. "captured" awaits
# disposition; "merged"/"addressed"/"rejected" are archived outcomes.
CAPTURE_STATUSES = {"captured", "merged", "addressed", "rejected"}
# Provenance tag for user-sourced items (P4). Agent-authored items carry no
# source key (backward compatible); user-sourced items are tagged and
# protected from silent replacement.
USER_SOURCE = "user"


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        # Persistent candidate buffer for new-request capture (P3). Each
        # entry: {id, content, source: "user", status, captured_at}. Bounded
        # to MAX_CAPTURED_REQUESTS; disposition is the model's explicit,
        # recorded action (see disposition_captures).
        self._captures: List[Dict[str, str]] = []
        self._next_capture_id = 1

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        if not merge:
            # Replace mode: new list entirely. User-sourced active items are
            # append-only (P4): a replace may mark them (complete/cancel) but
            # never silently drop them, so any active user item missing from
            # the new list is re-appended at the tail.
            incoming = [self._validate(t) for t in self._dedupe_by_id(todos)]
            incoming_ids = {item["id"] for item in incoming}
            for item in self._items:
                if (
                    item.get("source") == USER_SOURCE
                    and item["status"] in {"pending", "in_progress"}
                    and item["id"] not in incoming_ids
                ):
                    incoming.append(item)
            self._items = incoming
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields the LLM actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = self._cap_content(str(t["content"]).strip())
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        # Bound total item count so a replayed/oversized list can't grow the
        # re-injection block without limit. Keep the highest-priority head
        # (list order is priority).
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty.
        """
        if not self._items:
            return None

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression.
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        if not active_items:
            return None

        lines = ["[Your active task list was preserved across context compression]"]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']})")

        return "\n".join(lines)

    def format_for_turn(self) -> Optional[str]:
        """Render the per-turn task block merged into the user turn (P2/P3).

        Unlike ``format_for_injection`` (compression-only snapshot), this is
        the always-on visibility block: the active list with the single
        ``in_progress`` item flagged as the current task, plus any captured
        request candidates awaiting disposition. Returns None when there is
        nothing to show (no active items and no pending captures) so an idle
        list injects zero tokens.
        """
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        pending_captures = self.pending_captures()
        if not active_items and not pending_captures:
            return None

        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }
        lines = ["[Active tasks]"]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            suffix = "   ← CURRENT TASK" if item["status"] == "in_progress" else ""
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']}){suffix}")
        if pending_captures:
            lines.append("[Captured requests]")
            for capture in pending_captures:
                lines.append(f"- [captured] {capture['id']}. {capture['content']}")
            lines.append(
                "Disposition each captured request: merge into your plan, "
                "mark addressed, or reject."
            )
        return "\n".join(lines)

    def capture_request(self, content: str) -> Optional[Dict[str, str]]:
        """Capture a user turn as a candidate request (P3, deterministic tier).

        Structural capture only — no natural-language parsing. The model
        dispositions the candidate later via ``disposition_captures``; junk
        is rejected explicitly, never filtered here. Returns the capture
        entry, or None for empty content.
        """
        content = str(content).strip()
        if not content:
            return None
        capture = {
            "id": f"c{self._next_capture_id}",
            "content": self._cap_content(content),
            "source": USER_SOURCE,
            "status": "captured",
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        self._next_capture_id += 1
        self._captures.append(capture)
        # Bound the buffer: keep the most recent candidates, dropping the
        # oldest first (archived entries included — the audit trail is the
        # DB row, which persists the same bounded buffer).
        if len(self._captures) > MAX_CAPTURED_REQUESTS:
            self._captures = self._captures[-MAX_CAPTURED_REQUESTS:]
        return capture

    def pending_captures(self) -> List[Dict[str, str]]:
        """Candidates still awaiting disposition."""
        return [c for c in self._captures if c["status"] == "captured"]

    def disposition_captures(
        self, dispositions: Any
    ) -> List[Dict[str, str]]:
        """Record explicit dispositions for captured candidates (P3).

        Accepts a mapping ``{capture_id: status}`` or a list of
        ``{"id": ..., "status": ...}`` dicts. Only known capture ids and
        valid lifecycle statuses are applied; everything else is ignored so
        a malformed model payload cannot corrupt the buffer. Returns the
        still-pending captures after the update.
        """
        if isinstance(dispositions, dict):
            pairs = dispositions.items()
        elif isinstance(dispositions, list):
            pairs = [
                (str(d.get("id", "")).strip(), str(d.get("status", "")).strip().lower())
                for d in dispositions
                if isinstance(d, dict)
            ]
        else:
            return self.pending_captures()

        by_id = {c["id"]: c for c in self._captures}
        for capture_id, status in pairs:
            capture = by_id.get(capture_id)
            if capture is not None and status in CAPTURE_STATUSES:
                capture["status"] = status
        return self.pending_captures()

    def to_json(self) -> str:
        """Serialize the full store state for persistence (P1)."""
        return json.dumps(
            {
                "items": self._items,
                "captures": self._captures,
                "next_capture_id": self._next_capture_id,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "TodoStore":
        """Restore a store from ``to_json`` output (P1).

        Raises ValueError on malformed input so callers can fall back to
        the history-scan path (mirrors ``GoalState.from_json``). Items are
        re-validated through the same pipeline as writes; capture entries
        are normalized defensively and invalid ones dropped.
        """
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("todo state must be a JSON object")
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("todo state 'items' must be a list")

        store = cls()
        store._items = [store._validate(t) for t in items]

        captures = data.get("captures")
        if isinstance(captures, list):
            for c in captures:
                if not isinstance(c, dict):
                    continue
                capture_id = str(c.get("id", "")).strip()
                content = str(c.get("content", "")).strip()
                if not capture_id or not content:
                    continue
                status = str(c.get("status", "captured")).strip().lower()
                if status not in CAPTURE_STATUSES:
                    status = "captured"
                store._captures.append({
                    "id": capture_id,
                    "content": store._cap_content(content),
                    "source": USER_SOURCE,
                    "status": status,
                    "captured_at": str(c.get("captured_at", "")).strip(),
                })

        # Never reuse a capture id that already exists in the restored
        # buffer; the parsed counter wins only when it is strictly larger.
        max_existing = 0
        for c in store._captures:
            try:
                max_existing = max(max_existing, int(c["id"].lstrip("c")) or 0)
            except ValueError:
                continue
        parsed_counter = data.get("next_capture_id")
        try:
            parsed_counter = int(parsed_counter) if parsed_counter is not None else 0
        except (TypeError, ValueError):
            parsed_counter = 0
        store._next_capture_id = max(parsed_counter, max_existing + 1)
        return store

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate oversized todo content to MAX_TODO_CONTENT_CHARS.

        A single huge item would otherwise inflate the post-compression
        re-injection block (format_for_injection) without bound. Keep the
        head — the actionable part of a task description — plus a marker.
        """
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status is valid.
        Returns a clean dict with only {id, content, status}.
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        validated = {"id": item_id, "content": content, "status": status}
        # Provenance (P4): only the user tag is recognized; agent-authored
        # items stay untagged for backward compatibility.
        if str(item.get("source", "")).strip().lower() == USER_SOURCE:
            validated["source"] = USER_SOURCE
        return validated

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # Non-dict items get a synthetic key so _validate can handle them
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
    dispositions: Any = None,
) -> str:
    """
    Single entry point for the todo tool. Reads or writes depending on params.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.
        store: the TodoStore instance from the AIAgent.
        dispositions: optional mapping/list of captured-request dispositions
            (P3). Applied after any write so a single call can both update
            the plan and disposition captured requests.

    Returns:
        JSON string with the full current list, pending captured requests,
        and summary metadata.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        # Guard: LLM sometimes sends todos as a JSON string instead of a list
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(
                f"todos must be a list, got {type(todos).__name__}"
            )
        items = store.write(todos, merge)
    else:
        items = store.read()

    if dispositions is not None:
        store.disposition_captures(dispositions)

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "captures": store.pending_captures(),
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# Behavioral guidance is baked into the description so it's part of the
# static tool schema (cached, never changes mid-conversation).

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled, "
        "source?: user}\n"
        "Items tagged source=user are the user's own tasks: you may mark "
        "them completed/cancelled but never silently drop them.\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Captured requests: when the response includes 'captures', "
        "disposition each one explicitly via the 'dispositions' parameter "
        "(merge into your plan, mark addressed, or reject). "
        "Never leave a captured request undispositioned.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status"
                        },
                        "source": {
                            "type": "string",
                            "enum": ["user"],
                            "description": (
                                "Provenance tag. Only 'user' is recognized; "
                                "omit for agent-authored items."
                            )
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False
            },
            "dispositions": {
                "type": "array",
                "description": (
                    "Captured-request dispositions: list of "
                    "{id: capture_id, status: merged|addressed|rejected}. "
                    "Disposition every captured request explicitly."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Captured request id (e.g. c1)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["merged", "addressed", "rejected"],
                            "description": "Disposition outcome"
                        }
                    },
                    "required": ["id", "status"]
                }
            }
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False),
        dispositions=args.get("dispositions"), store=kw.get("store")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
