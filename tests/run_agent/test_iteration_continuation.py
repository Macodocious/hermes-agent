"""Iteration-budget continuation: extend a turn past budget exhaustion.

When a turn exhausts its iteration budget the loop summarises progress, folds
that summary into the agent's own context, resets the budget, and continues —
mirroring how context compaction lets work continue past a limit. The summary
is context for the agent and must never reach the user as the turn's response.

These tests exercise ``_try_iteration_continuation`` directly (the behavior
owner) plus the ``append_to_history`` contract on ``handle_max_iterations``
that keeps the terminal fallback unchanged.
"""

from types import SimpleNamespace

import pytest

from agent.conversation_loop import _try_iteration_continuation
from agent.iteration_budget import IterationBudget


class _ContinuationAgent:
    """Minimal agent surface the continuation helper touches."""

    def __init__(
        self,
        *,
        max_iterations=5,
        continuations_used=0,
        max_continuations=3,
        summary="progress so far: analysed files, wrote helper",
        summary_returned=True,
    ):
        self.max_iterations = max_iterations
        self._iteration_continuations = continuations_used
        self._max_iteration_continuations = max_continuations
        self.iteration_budget = IterationBudget(max_iterations)
        for _ in range(max_iterations):
            self.iteration_budget.consume()  # exhaust it
        self._api_call_count = max_iterations
        self._summary = summary
        self._summary_returned = summary_returned
        self.statuses = []
        self.summary_calls = []

    def _emit_status(self, message):
        self.statuses.append(message)

    def _handle_max_iterations(self, messages, api_call_count, append_to_history=True):
        self.summary_calls.append(
            {
                "api_call_count": api_call_count,
                "append_to_history": append_to_history,
                "messages_len": len(messages),
            }
        )
        if not self._summary_returned:
            return ""
        return self._summary


def _tail_messages():
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "tool result"},
    ]


def test_continuation_fires_and_resets_both_counters():
    """A continuation returns 0 and restores the call count and budget."""
    agent = _ContinuationAgent()
    messages = _tail_messages()

    resumed = _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert resumed == 0
    assert agent._api_call_count == 0
    assert agent.iteration_budget.used == 0
    assert agent.iteration_budget.remaining == agent.max_iterations
    assert agent._iteration_continuations == 1


def test_summary_is_folded_into_context_and_never_returned():
    """The summary lands as tagged history; nothing is handed back as output."""
    agent = _ContinuationAgent()
    messages = _tail_messages()

    result = _try_iteration_continuation(agent, messages, agent._api_call_count)

    # The helper's own return value is a resume count, not the summary text.
    assert result == 0
    assert result != agent._summary

    folded = messages[-1]
    assert folded["role"] == "user"
    # Terminated with the compaction end marker so a weak model cannot replay
    # the summary as fresh input (#11475, #14521).
    assert folded["content"].startswith(agent._summary)
    # Tagged exactly like a compaction summary so frontends exclude it.
    assert folded.get("_compressed_summary") is True


def test_summary_request_does_not_mutate_history():
    """The continuation asks for the summary without appending to history."""
    agent = _ContinuationAgent()
    messages = _tail_messages()
    before = len(messages)

    _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert agent.summary_calls == [
        {
            "api_call_count": agent.max_iterations,
            "append_to_history": False,
            "messages_len": before,
        }
    ]
    # Exactly one message added — the folded summary. No synthetic request.
    assert len(messages) == before + 1
    assert all(
        "maximum number of tool-calling iterations" not in (m.get("content") or "")
        for m in messages
    )


def test_cap_is_enforced_then_falls_through():
    """Once the allowance is spent, no continuation happens."""
    agent = _ContinuationAgent(continuations_used=3, max_continuations=3)
    messages = _tail_messages()
    before = list(messages)

    resumed = _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert resumed is None
    assert messages == before
    assert agent.summary_calls == []
    assert agent._iteration_continuations == 3


def test_cap_of_zero_disables_continuation():
    """max_iteration_continuations=0 restores the legacy terminal behavior."""
    agent = _ContinuationAgent(max_continuations=0)
    messages = _tail_messages()
    before = list(messages)

    resumed = _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert resumed is None
    assert messages == before
    assert agent.summary_calls == []


def test_empty_summary_falls_through_without_folding():
    """No summary means nothing to fold in — the turn ends normally."""
    agent = _ContinuationAgent(summary_returned=False)
    messages = _tail_messages()
    before = list(messages)

    resumed = _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert resumed is None
    assert messages == before
    # The attempt still consumed one continuation of the allowance.
    assert agent._iteration_continuations == 1


def test_missing_attributes_degrade_to_legacy_behavior():
    """An agent lacking the continuation attrs must not raise."""
    agent = SimpleNamespace(
        max_iterations=5,
        iteration_budget=IterationBudget(5),
        _api_call_count=5,
    )
    agent._emit_status = lambda message: None
    agent._handle_max_iterations = lambda *a, **k: "summary"
    messages = _tail_messages()
    before = list(messages)

    resumed = _try_iteration_continuation(agent, messages, 5)

    assert resumed is None
    assert messages == before


def test_role_alternation_is_preserved():
    """The folded summary must not create consecutive same-role messages."""
    agent = _ContinuationAgent()
    messages = _tail_messages()

    _try_iteration_continuation(agent, messages, agent._api_call_count)

    roles = [m["role"] for m in messages]
    assert not any(
        prev == curr == "assistant" for prev, curr in zip(roles, roles[1:])
    )
    # tool -> user: the summary is user-role because the loop's next message is
    # always the model's assistant response, so assistant here would collide.
    assert roles[-2:] == ["tool", "user"]


def test_status_line_is_emitted_once_per_continuation():
    """Each continuation emits the existing budget-exhausted status."""
    agent = _ContinuationAgent()
    messages = _tail_messages()

    _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert len(agent.statuses) == 1
    assert "Iteration budget exhausted" in agent.statuses[0]
    assert "summarise" in agent.statuses[0]


@pytest.mark.parametrize("used,max_allowed", [(0, 1), (1, 2), (2, 3)])
def test_continuation_allowed_until_cap(used, max_allowed):
    agent = _ContinuationAgent(continuations_used=used, max_continuations=max_allowed)
    messages = _tail_messages()

    resumed = _try_iteration_continuation(agent, messages, agent._api_call_count)

    assert resumed == 0
    assert agent._iteration_continuations == used + 1
