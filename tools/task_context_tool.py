#!/usr/bin/env python3
"""
Task Context Declaration Tool - Per-Turn Temperature Override Selection

Lets the agent declare, in the moment, whether the current work is `general`
conversation or `coding` output. The declaration is recorded on the agent and
applied at the pre-LLM-call temperature chokepoint for the next API request.

The tool also carries an optional user-directed `temperature` argument: when
the operator tells the agent to run a request at a specific temperature
(e.g. "retry with temperature 1.0"), the agent declares that exact value and
the chokepoint applies it to the next API request.

The tool's schema is only exposed when the operator has enabled the
temperature override (``agent.temperature.override.enabled: true`` in
config.yaml) — enforced via ``check_fn``. Without the override, the tool
never exists in the schema, costs no tokens, and the base temperature
applies to every request.
"""

from typing import Optional

# The only valid declaration values.
VALID_CONTEXTS = ("general", "coding")

# Bounds on user-directed temperatures. OpenAI-compatible sampling range.
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0


def declare_task_context(
    context: Optional[str],
    temperature: Optional[float] = None,
    agent=None,
) -> str:
    """Record the agent's per-turn task-context declaration.

    Args:
        context: The declared context — ``"general"`` or ``"coding"``.
        temperature: Optional user-directed explicit temperature for the
            next API request (e.g. the operator said "retry with
            temperature 1.0"). Takes precedence over the context knob for
            that request.
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

    explicit = None
    if temperature is not None:
        try:
            explicit = float(temperature)
        except (TypeError, ValueError):
            return tool_error(f"temperature must be a number, got {temperature!r}.")
        if not (MIN_TEMPERATURE <= explicit <= MAX_TEMPERATURE):
            return tool_error(
                f"temperature must be between {MIN_TEMPERATURE} and "
                f"{MAX_TEMPERATURE}, got {explicit}."
            )

    agent._declared_task_context = normalized
    agent._declared_explicit_temperature = explicit
    payload: dict = {
        "declared": normalized,
        "applied_to": "next API request",
    }
    if explicit is not None:
        payload["explicit_temperature"] = explicit
    return tool_result(payload)


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
        "Default when never called: `general`. When the operator explicitly "
        "asks for a specific temperature (e.g. \"retry with temperature "
        "1.0\"), call this with `context` set to the current work type and "
        "`temperature` set to the exact value they stated."
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
            "temperature": {
                "type": "number",
                "description": (
                    "Optional user-directed explicit temperature for the "
                    "next API request (0.0–2.0). Use ONLY when the operator "
                    "explicitly states a temperature. Takes precedence over "
                    "the context knob for that request."
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
        context=args.get("context"),
        temperature=args.get("temperature"),
        agent=kw.get("agent")),
    check_fn=check_task_context_requirements,
    emoji="🎛️",
)