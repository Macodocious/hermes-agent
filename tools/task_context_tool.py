#!/usr/bin/env python3
"""
Task Context Declaration Tool - Per-Turn Temperature Override Selection

Lets the agent declare, in the moment, whether the current work is `general`
conversation or `coding` output. The declaration is recorded on the agent and
applied at the pre-LLM-call temperature chokepoint for the next API request.

The tool's schema is only exposed when the operator has enabled the
temperature override (``agent.temperature.override.enabled: true`` in
config.yaml) — enforced via ``check_fn``. Without the override, the tool
never exists in the schema, costs no tokens, and the base temperature
applies to every request.
"""

from typing import Optional

# The only valid declaration values.
VALID_CONTEXTS = ("general", "coding")


def declare_task_context(context: Optional[str], agent=None) -> str:
    """Record the agent's per-turn task-context declaration.

    Args:
        context: The declared context — ``"general"`` or ``"coding"``.
        agent:   The AIAgent instance, injected by the tool executor.

    Returns:
        JSON string confirming the recorded declaration.
    """
    from tools.registry import tool_error, tool_result

    if agent is None:
        return tool_error("Agent context unavailable; declaration not recorded.")
    if not isinstance(context, str):
        return tool_error("context must be a string: 'general' or 'coding'.")
    normalized = context.strip().lower()
    if normalized not in VALID_CONTEXTS:
        return tool_error(
            f"context must be one of {list(VALID_CONTEXTS)}, got {context!r}."
        )

    agent._declared_task_context = normalized
    return tool_result(
        declared=normalized,
        applied_to="next API request",
    )


def check_task_context_requirements() -> bool:
    """Expose the schema only while the temperature override is enabled."""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
    except Exception:
        return False
    agent_config = config.get("agent")
    if not isinstance(agent_config, dict):
        return False
    temperature = agent_config.get("temperature")
    if not isinstance(temperature, dict):
        return False
    return bool((temperature.get("override") or {}).get("enabled"))


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

TASK_CONTEXT_SCHEMA = {
    "name": "declare_task_context",
    "description": (
        "Declare whether the current work is `general` conversation or "
        "`coding` output. Call this the moment the nature of your work "
        "changes: before starting or finishing a coding task, call it with "
        "the new context. The declaration sets the sampling temperature of "
        "your next response — `general` for conversation, `coding` for "
        "code work — as configured by the operator's temperature override. "
        "Default when never called: `general`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "enum": list(VALID_CONTEXTS),
                "description": (
                    "The task context you are declaring: `general` for "
                    "conversation, `coding` for coding work."
                ),
            },
        },
        "required": ["context"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="declare_task_context",
    toolset="task_context",
    schema=TASK_CONTEXT_SCHEMA,
    handler=lambda args, **kw: declare_task_context(
        context=args.get("context"), agent=kw.get("agent")),
    check_fn=check_task_context_requirements,
    emoji="🎛️",
)