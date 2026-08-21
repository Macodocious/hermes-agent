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
import re
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

# ---------------------------------------------------------------------------
# User-message scaffold stripping (P5)
#
# The gateway prepends structural scaffolding to inbound user messages:
# the Discord triggering-message envelope, reply-to pointers, and
# (optionally) message timestamps. That scaffolding is routing metadata for
# the model, not task content — when it is stored verbatim as a seeded task
# or a captured request it pollutes the todo list and /task output (the
# "[Triggering message id: ...]" rows).
#
# System notes and goal-loop continuations are NOT stripped here: they are
# exclusion markers. The seed/capture callers check the stripped text
# against the excluded prefixes and skip the message entirely, so the
# markers must survive stripping to be seen.
#
# Stripping is structural and leading-only: each pattern is matched at the
# very start of the text (after any already-stripped prefixes), so genuine
# user content that merely contains a similar phrase mid-message is never
# touched. The sender-name prefix ("[Mac] ...") is only removed when it
# directly follows a stripped header line, because a bare "[Mac] ..." at the
# start of a message is the user's own text on platforms that do not add
# sender prefixes.
# ---------------------------------------------------------------------------

# Discord: "[Triggering message id: `123` — use as `message_id` for
# reply/react/pin via the discord tools.]" (gateway/run.py).
_TRIGGERING_MESSAGE_ID_RE = re.compile(
    r"^\[Triggering message id: `[^`]+`[^\]]*\]\s*"
)
# Reply pointers: "[Replying to: \"...\"]" and
# "[Replying to your previous message: \"...\"]" (gateway/run.py).
_REPLY_TO_RE = re.compile(r"^\[Replying to(?: your previous message)?: \"[^\"]*\"\]\s*")
# Optional gateway message timestamps: "[Wed 2026-08-13 09:15:00 UTC]"
# and the older "[2026-08-13T09:15:00+00:00]" format
# (gateway/message_timestamps.py).
_TIMESTAMP_RE = re.compile(
    r"^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [^\]]+\]\s*"
)
_ISO_TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}T[^\]]+\]\s*")
# Sender-name prefix ("[Mac] ...") — only stripped when it follows a header.
_SENDER_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")

_HEADER_STRIP_PATTERNS = (
    _TRIGGERING_MESSAGE_ID_RE,
    _REPLY_TO_RE,
    _TIMESTAMP_RE,
    _ISO_TIMESTAMP_RE,
)


def strip_user_scaffold(text: Any) -> str:
    """Strip leading gateway scaffolding from a user message.

    Removes, in order: the Discord triggering-message envelope, reply-to
    pointers, and message timestamps. A sender-name prefix (``[Name]``) is
    removed only when it directly follows one of those headers. System notes
    and goal-loop continuations are left intact — they are exclusion markers
    the callers check before storing content. Returns the stripped text;
    non-string input is stringified first.
    """
    text = str(text or "")
    stripped_header = False
    changed = True
    while changed:
        changed = False
        for pattern in _HEADER_STRIP_PATTERNS:
            match = pattern.match(text)
            if match:
                text = text[match.end():]
                stripped_header = True
                changed = True
    if stripped_header:
        sender_match = _SENDER_PREFIX_RE.match(text)
        if sender_match:
            text = text[sender_match.end():]
    return text.strip()


class TodoStore:
    """
    In-memory todo list. One instance per AIAgent (one per session).

    Items are ordered -- list position is priority. Each item has:
      - id: sequential integer identifier (store-assigned, 1..N in list order)
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
        # True while the only item is the deterministic seed of the opening
        # user message (P5). The model has not yet authored the plan; the
        # per-turn block nudges it to rename the seed into a concise title
        # on the first todo write. Any model write clears the flag.
        self._seeded = False

    def seed_from_user_message(self, text: Any) -> Optional[Dict[str, str]]:
        """Seed an empty store with the user's opening message (P5).

        Stores the scaffold-stripped text as the active user-sourced task
        and marks the store as seeded so the per-turn block can nudge the
        model to rename it. Returns the seeded item, or None when the text
        is empty after stripping or the store already has items.
        """
        if self.has_items():
            return None
        content = strip_user_scaffold(text)
        if not content:
            return None
        self._seeded = True
        self._items = [
            {"id": "1", "content": self._cap_content(content), "status": "in_progress", "source": USER_SOURCE}
        ]
        return self._items[0].copy()

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        # Detect which task (if any) this write starts before mutating the
        # list, so the invariant enforcement can keep it current.
        started = self.detect_task_start(todos, merge=merge)
        # Any model write means the plan is now model-owned: the seeded
        # placeholder (if any) is no longer the only item, so the rename
        # nudge stops rendering.
        self._seeded = False
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
                    # New item -- validate fully, assign the next sequential
                    # id (merge mode keeps existing ids stable so merge-by-id
                    # stays safe), append to end
                    validated = self._validate(t)
                    validated["id"] = self._next_item_id()
                    existing[validated["id"]] = validated
                    self._items.append(validated)
                    # The write-started placeholder resolves to the assigned
                    # id so the invariant keeps THIS task current.
                    if started is not None and started["id"] == item_id:
                        started["id"] = validated["id"]
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
        # Enforce the single-current-task invariant (see
        # _enforce_invariants). The started task (if any) is the one that
        # stays current. Replace mode then renumbers to sequential ids;
        # merge mode keeps existing ids stable (merge-by-id stays safe).
        self._enforce_invariants(started_id=(started or {}).get("id"))
        if not merge:
            self._renumber_items(self._items)
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def _enforce_invariants(self, started_id: Optional[str] = None) -> None:
        """Enforce the single-current-task invariant.

        At most one item may be ``in_progress``. The kept item is the task
        this write started (``started_id``, the same transition
        ``detect_task_start`` detects) when it is present and
        ``in_progress``; otherwise the first ``in_progress`` in list order
        (list order is priority). Any others demote to ``pending``. This
        makes two ``[>]`` rows impossible at the data layer regardless of
        what the model writes, and makes a task switch via merge keep the
        newly started task current.

        Renumbering is NOT part of this method: replace-mode writes and
        the load path renumber (see _renumber_items); merge-mode writes
        keep existing ids stable so merge-by-id stays safe.
        """
        current_id = started_id
        if current_id is not None:
            current = next(
                (item for item in self._items if item["id"] == current_id),
                None,
            )
            if current is None or current["status"] != "in_progress":
                current_id = None  # Started item not current in the new list
        if current_id is None:
            current = next(
                (item for item in self._items if item["status"] == "in_progress"),
                None,
            )
        for item in self._items:
            if item["status"] == "in_progress" and item is not current:
                item["status"] = "pending"

    @classmethod
    def _renumber_items(cls, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Renumber a list of validated items to sequential ids (1..N)."""
        for index, item in enumerate(items, start=1):
            item["id"] = str(index)
        return items

    def _next_item_id(self) -> str:
        """Next sequential id for merge-mode new items (max existing + 1)."""
        numeric_ids = [int(item["id"]) for item in self._items if item["id"].isdigit()]
        return str(max(numeric_ids, default=0) + 1)

    def detect_task_start(
        self, todos: Optional[List[Dict[str, Any]]], merge: bool = False
    ) -> Optional[Dict[str, str]]:
        """Detect which task (if any) a pending write transitions to in_progress.

        A task "starts" when an item that was not ``in_progress`` before the
        call becomes ``in_progress``: either an existing item moving from
        another status, or a brand-new item created directly as
        ``in_progress``. Re-asserting the same item as ``in_progress`` is not
        a start. Returns the first started item (list order is priority), or
        None when no item starts. Mirrors the write pipeline's normalization
        (string payloads, list shape, content fallback) so the detection
        matches what ``write`` will actually store.
        """
        if todos is None:
            return None
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(todos, list):
            return None

        previous = {item["id"]: item for item in self._items}
        for t in todos:
            if not isinstance(t, dict):
                continue
            item_id = str(t.get("id", "")).strip()
            if not item_id:
                continue
            status = str(t.get("status", "")).strip().lower()
            if status != "in_progress":
                continue
            prev_item = previous.get(item_id)
            if prev_item is not None and prev_item["status"] == "in_progress":
                continue  # Re-asserting the current task is not a start
            content = str(t.get("content", "")).strip()
            if not content and prev_item is not None:
                content = prev_item["content"]
            if not content:
                content = "(no description)"
            return {
                "id": item_id,
                "content": self._cap_content(content),
                "status": "in_progress",
            }
        return None

    def format_for_injection(self) -> Optional[str]:
        """
        Render the task list for post-compression injection.

        Completed items are injected alongside active ones under a
        "Completed — do not redo" header, so the full list survives
        compression instead of vanishing when nothing is active. An
        all-completed list still renders (the compressed history keeps a
        visible task anchor); cancelled items are excluded because they
        were deliberately abandoned. Returns None only when the store is
        empty.
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

        # Completed items are visible (marked [x]) so finished work
        # persists across compression without being re-proposed; the
        # header explicitly flags them as done to prevent redoing.
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        completed_items = [
            item for item in self._items if item["status"] == "completed"
        ]

        lines = ["[Your active task list was preserved across context compression]"]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            lines.append(f"- {marker} {item['content']} ({item['status']})")
        if completed_items:
            lines.append("Completed — do not redo:")
            for item in completed_items:
                lines.append(f"- [x] {item['content']}")

        return "\n".join(lines)

    def format_for_turn(self) -> Optional[str]:
        """Render the per-turn task block merged into the user turn (P2/P3).

        Unlike ``format_for_injection`` (compression-only snapshot), this is
        the always-on visibility block: the active list with the single
        ``in_progress`` item flagged as the current task, completed items
        marked ``[x]``, plus any captured request candidates awaiting
        disposition. Returns None when there is nothing to show (no items
        at all and no pending captures) so an idle list injects zero
        tokens.
        """
        active_items = [
            item for item in self._items
            if item["status"] in {"pending", "in_progress"}
        ]
        completed_items = [
            item for item in self._items if item["status"] == "completed"
        ]
        pending_captures = self.pending_captures()
        if not active_items and not completed_items and not pending_captures:
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
            lines.append(f"- {marker} {item['content']}")
        if completed_items:
            lines.append("Completed:")
            for item in completed_items:
                lines.append(f"- [x] {item['content']}")
        if self._seeded and len(active_items) == 1:
            lines.append(
                "Rename the seeded task above into a concise title on your "
                "first todo write (the seed is the user's opening message)."
            )
        if pending_captures:
            lines.append("[Captured requests]")
            for capture in pending_captures:
                lines.append(f"- [ ] {capture['content']} (captured)")
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
        content = strip_user_scaffold(content)
        if not content:
            return None
        capture = {
            "id": str(self._next_capture_id),
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
                "seeded": self._seeded,
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
        # Heal polluted persisted state on load: enforce the same
        # single-current-task invariant and renumber to sequential ids.
        store._enforce_invariants()
        store._renumber_items(store._items)
        # Restore the seeded flag (P5). Old persisted state without the key
        # defaults to False, so pre-existing sessions never render the nudge.
        store._seeded = bool(data.get("seeded", False))

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
        # Legacy persisted ids may carry the old "c" prefix (pre-inline
        # scheme); strip it so the counter stays numeric.
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
            content = TodoStore._cap_content(strip_user_scaffold(content))

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
        "Ids are store-assigned: replace-mode writes renumber the list to "
        "1..N in list order; merge-mode writes keep existing ids stable and "
        "assign new items the next integer. Pass any unique placeholder "
        "and read back the assigned id.\n"
        "Items tagged source=user are the user's own tasks: you may mark "
        "them completed/cancelled but never silently drop them.\n"
        "List order is priority. Only ONE item in_progress at a time — "
        "the store enforces this: starting a new task automatically demotes "
        "the previous current task to pending.\n"
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
