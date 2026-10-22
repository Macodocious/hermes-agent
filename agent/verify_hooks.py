"""Verification-loop helpers for the round-end continuation gates.

When the agent has edited code and is about to verify/finish, the loop fires the
``pre_verify`` hook (user directives resolved by
:func:`hermes_cli.plugins.get_pre_verify_continue_message`). A directive keeps
the agent going one more turn — run a check, defer it, tidy the diff — instead of
stopping immediately.

The ``pre_delivery`` gate (resolved by
:func:`hermes_cli.plugins.get_pre_delivery_answer`) fires on every turn
end, not only a file-edit turn, and is bounded by its own key.

The shipped coding guidance lives on the evidence-based verification-stop nudge
(``agent/verification_stop.py``), not as a second default stop gate. That keeps
the default token cost tied to the existing "missing verification evidence"
decision while preserving ``pre_verify`` for user/plugin policy.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from utils import is_truthy_value

DEFAULT_MAX_VERIFY_NUDGES = 3

# ── Tool evidence for the pre_delivery gate ─────────────────────────────────
# A call alone proves only that something ran; a ``curl`` returning 503 still
# ran. Each call therefore carries a head-and-tail excerpt of its result, under
# a per-result and an overall cap so the payload stays bounded.
TOOL_EXCERPT_CHARS = 2048
TOOL_EVIDENCE_CHARS = 16384
TOOL_EVIDENCE_CALLS = 32

# Mirrors verify-on-stop's ``max_attempts`` so both turn-end gates share one
# cost ceiling.
DEFAULT_MAX_PRE_DELIVERY_NUDGES = 2

# Shipped guidance appended to the verification-stop nudge when code lacks fresh
# verification evidence. Wording mirrors the user-facing "clean your work"
# workflow, but does not create its own extra model turn.
CODING_VERIFY_GUIDANCE = (
    "[Coding] Before you run tests/linters or call this done: if this is "
    "creative UI/visual work, hold off on tests and linters until the user says "
    "they like the result or you're about to commit. And before every commit, "
    "clean your work: keep it KISS/DRY, match the surrounding code style, and be "
    "elitist, shorthand, clever, concise, efficient, and elegant."
)


def max_verify_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_verify`` continue directives per turn (>= 0)."""
    agent_cfg = _agent_cfg(config)
    raw = agent_cfg.get("max_verify_nudges")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VERIFY_NUDGES


def max_pre_delivery_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_delivery`` holds per turn (>= 0)."""
    agent_cfg = _agent_cfg(config)
    raw = agent_cfg.get("max_pre_delivery_nudges")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PRE_DELIVERY_NUDGES


def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return the optional guidance appended to verification-stop nudges."""
    if not is_truthy_value(_agent_cfg(config).get("verify_guidance", True), default=True):
        return None
    return CODING_VERIFY_GUIDANCE


def build_tool_evidence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The turn's tool calls: name, arguments, and a capped result excerpt.

    The turn starts at the last real user message. Synthetic user messages (a
    hold nudge, a kanban stop nudge) do not move that boundary, so a later
    submission still sees what the same turn measured.
    """
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        if any(str(key).endswith("_synthetic") for key in message):
            continue
        start = index
        break

    names: dict[str, tuple[str, str]] = {}
    calls: list[dict[str, Any]] = []
    for message in messages[start:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                names[str(call.get("id") or "")] = (
                    str(function.get("name") or ""),
                    str(function.get("arguments") or ""),
                )
            continue
        if role != "tool":
            continue
        name, arguments = names.get(str(message.get("tool_call_id") or ""), ("", ""))
        calls.append(
            {
                "name": str(message.get("name") or name),
                "arguments": arguments,
                "excerpt": _excerpt(message.get("content", "")),
            }
        )

    return _bounded(calls)


def _excerpt(content: Any) -> str:
    """A result's first and last half of the per-result cap."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    if len(text) <= TOOL_EXCERPT_CHARS:
        return text
    half = TOOL_EXCERPT_CHARS // 2
    return text[:half] + "\n…\n" + text[-half:]


def _bounded(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every name and argument, excerpts newest-first, cut at the budget.

    A call's excerpt that would cross the cap is dropped; older calls keep their
    name and arguments with an empty excerpt, so no call disappears whole.
    """
    recent = calls[-TOOL_EVIDENCE_CALLS:]
    kept = [
        {"name": call["name"], "arguments": call["arguments"], "excerpt": ""}
        for call in recent
    ]
    budget = TOOL_EVIDENCE_CHARS
    for index in range(len(recent) - 1, -1, -1):
        excerpt = recent[index]["excerpt"]
        if len(excerpt) > budget:
            break
        kept[index]["excerpt"] = excerpt
        budget -= len(excerpt)
    return kept


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = [
    "CODING_VERIFY_GUIDANCE",
    "DEFAULT_MAX_PRE_DELIVERY_NUDGES",
    "DEFAULT_MAX_VERIFY_NUDGES",
    "build_tool_evidence",
    "coding_verify_guidance",
    "max_pre_delivery_nudges",
    "max_verify_nudges",
]
