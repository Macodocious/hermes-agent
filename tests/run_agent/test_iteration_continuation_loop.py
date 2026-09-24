"""Loop-level regression coverage for iteration-budget continuation.

The defect these tests pin down: the continuation trigger originally lived on
``elif not agent.iteration_budget.consume():`` *inside* the loop body, whose
``while`` condition already required ``iteration_budget.remaining > 0`` to
enter. ``consume()`` therefore always returned ``True`` there, the branch was
unreachable, and an exhausted turn fell through to the finalizer — which
surfaced the summary to the user. The helper-level suite passed throughout,
because calling the helper directly proves only that it works in isolation,
never that the loop ever calls it.

These tests drive the *real* loop via ``run_conversation`` and assert on loop
behaviour: how many model calls happen, whether the turn continues past
exhaustion, and whether the folded summary stays out of the final response.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _text_response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _tool_response(name="web_search", arguments="{}", call_id="call_1"):
    """A response that keeps the turn mid-work: the tail becomes a tool result."""
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            session_id="iteration-continuation-loop-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    # Tools are stubbed at the execution seam: each call appends a tool result
    # so the loop re-enters and re-evaluates the budget gate, exactly as a real
    # work-in-progress turn would.
    instance.valid_tool_names = ["web_search"]

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls or []:
            messages.append(
                {
                    "role": "tool",
                    "name": call.function.name,
                    "tool_call_id": call.id,
                    "content": "tool ok",
                }
            )

    instance._execute_tool_calls = MagicMock(side_effect=_fake_execute)
    return instance


def test_continuation_extends_turn_past_exhaustion(agent, monkeypatch):
    """Exhaustion mid-work must continue the turn, not surface a summary.

    Budget 1, one continuation allowed. The first call is a tool call, so the
    budget runs out with work in progress. The loop must then fold a summary
    into context, reset the budget, and make a SECOND model call — which
    returns the real final answer. The folded summary must never be the
    response the user sees.
    """
    calls = {"n": 0}

    def scripted(_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_response()
        return _text_response("final answer after continuation")

    agent.valid_tool_names = ["web_search"]
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    agent._max_iteration_continuations = 1
    agent._interruptible_api_call = scripted
    agent._handle_max_iterations = MagicMock(return_value="folded continuation summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("do the work")

    # The loop made a call after exhaustion — the continuation fired.
    assert calls["n"] == 2, "loop did not continue past the exhausted budget"
    assert agent._iteration_continuations == 1
    # The continuation summary was produced once and kept as context...
    agent._handle_max_iterations.assert_called_once()
    # ...and the user sees the model's answer, never the summary.
    assert result["final_response"] == "final answer after continuation"
    assert result["final_response"] != "folded continuation summary"
    # The folded summary is alternation-safe: the loop's defensive repair pass
    # must not have had to touch the history, and the summary must sit as a
    # user-role message tagged as a compressed summary.
    folded = [
        m for m in result["messages"] if m.get("_compressed_summary")
    ]
    assert len(folded) == 1
    assert folded[0]["role"] == "user"
    assert "folded continuation summary" in folded[0]["content"]
    roles = [m["role"] for m in result["messages"]]
    assert not any(
        a == b == "assistant" for a, b in zip(roles, roles[1:])
    ), f"consecutive assistant messages: {roles}"


def test_continuation_cap_bounds_the_turn(agent, monkeypatch):
    """The continuation cap must stop a turn that cannot finish.

    Budget 1, cap 2, and a model that always returns a tool call. Without the
    cap this would loop forever. Exactly two continuations must happen, then
    the turn finalizes through the terminal fallback.
    """
    calls = {"n": 0}

    def always_tool(_kwargs):
        calls["n"] += 1
        return _tool_response(call_id=f"call_{calls['n']}")

    agent.valid_tool_names = ["web_search"]
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    agent._max_iteration_continuations = 2
    agent._interruptible_api_call = always_tool
    agent._handle_max_iterations = MagicMock(return_value="folded continuation summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("do the work")

    # 1 initial iteration + 2 continuations = 3 model calls, then it stops.
    assert calls["n"] == 3, f"expected 3 calls, got {calls['n']}"
    assert agent._iteration_continuations == 2
    # Cap spent: 2 continuation summaries, then the terminal fallback calls it
    # once more — 3 total.
    assert agent._handle_max_iterations.call_count == 3
    assert result["turn_exit_reason"].startswith("max_iterations_reached")


def test_cap_zero_disables_continuation(agent, monkeypatch):
    """cap 0 disables the feature: exhaustion finalizes exactly as before."""
    calls = {"n": 0}

    def always_tool(_kwargs):
        calls["n"] += 1
        return _tool_response(call_id=f"call_{calls['n']}")

    agent.valid_tool_names = ["web_search"]
    agent.max_iterations = 1
    agent.iteration_budget.max_total = 1
    agent._max_iteration_continuations = 0
    agent._interruptible_api_call = always_tool
    agent._handle_max_iterations = MagicMock(return_value="folded continuation summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("do the work")

    # No continuation: a single iteration, then the terminal fallback.
    assert calls["n"] == 1
    assert agent._iteration_continuations == 0
    assert result["turn_exit_reason"].startswith("max_iterations_reached")
