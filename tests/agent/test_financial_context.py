"""The financial context selects the operator's financial temperature knob."""

from __future__ import annotations

from types import SimpleNamespace

from agent.chat_completion_helpers import _resolve_effective_temperature
from tools.task_context_tool import VALID_CONTEXTS


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        temperature={
            "value": 0.0,
            "override": {"enabled": True, "general": 1.0, "coding": 0.0, "financial": 0.1},
        },
        _declared_task_context="financial",
        _declared_explicit_temperature=None,
    )


def test_financial_is_a_valid_context() -> None:
    assert "financial" in VALID_CONTEXTS


def test_financial_declaration_picks_the_financial_knob() -> None:
    assert _resolve_effective_temperature(_agent()) == 0.1


def test_unset_financial_falls_back_to_the_base_value() -> None:
    agent = _agent()
    agent.temperature["override"].pop("financial")
    assert _resolve_effective_temperature(agent) == 0.0
