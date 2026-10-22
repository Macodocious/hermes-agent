"""Pre-delivery continuation gate (agent/conversation_loop.py, agent/verify_hooks.py).

The `pre_delivery` hook lets a plugin hold ANY turn end — not only a turn that
mutated files — before the candidate answer is delivered. These tests cover the
bound, the no-consumer no-op, the hold-then-deliver path, and fail-open on a
raising hook.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent as ra
from agent import verify_hooks
from run_agent import AIAgent


def _response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
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
            session_id="pre-delivery-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=3,
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
    return instance


class TestMaxPreDeliveryNudges:
    def test_default_when_unset(self):
        assert (
            verify_hooks.max_pre_delivery_nudges({})
            == verify_hooks.DEFAULT_MAX_PRE_DELIVERY_NUDGES
        )
        assert (
            verify_hooks.max_pre_delivery_nudges({"agent": {}})
            == verify_hooks.DEFAULT_MAX_PRE_DELIVERY_NUDGES
        )

    def test_reads_and_coerces(self):
        assert verify_hooks.max_pre_delivery_nudges({"agent": {"max_pre_delivery_nudges": 5}}) == 5
        assert verify_hooks.max_pre_delivery_nudges({"agent": {"max_pre_delivery_nudges": "2"}}) == 2
        assert verify_hooks.max_pre_delivery_nudges({"agent": {"max_pre_delivery_nudges": -1}}) == 0

    def test_bad_value_falls_back(self):
        assert (
            verify_hooks.max_pre_delivery_nudges({"agent": {"max_pre_delivery_nudges": "x"}})
            == verify_hooks.DEFAULT_MAX_PRE_DELIVERY_NUDGES
        )


def test_pre_delivery_flag_is_ephemeral_scaffolding():
    """Both the withheld candidate and its nudge must never reach durable state."""
    assert "_pre_delivery_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS
    assert ra._is_ephemeral_scaffolding(
        {"role": "assistant", "content": "withheld draft", "_pre_delivery_synthetic": True}
    )
    assert ra._is_ephemeral_scaffolding(
        {"role": "user", "content": "[System: verify]", "_pre_delivery_synthetic": True}
    )
    assert not ra._is_ephemeral_scaffolding({"role": "assistant", "content": "delivered"})


def test_no_consumer_turn_is_unchanged(agent, monkeypatch):
    """With no `pre_delivery` hook registered, the turn behaves natively."""
    agent._interruptible_api_call = lambda _kwargs: _response("plain answer")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("just answer")

    assert result["final_response"] == "plain answer"
    assert result["completed"] is True
    assert agent._pre_delivery_nudges == 0
    assert emitted == []
    assert [m["role"] for m in result["messages"]] == ["user", "assistant"]


def test_hold_withholds_candidate_then_delivers_the_next(agent, monkeypatch):
    """A hook that holds once: the first candidate is withheld, the second delivered."""
    answers = iter([_response("unverified claim"), _response("verified answer")])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_delivery"),
        patch(
            "hermes_cli.plugins.get_pre_delivery_answer",
            side_effect=[("continue", "go verify that claim"), None],
        ),
        patch("agent.verify_hooks.max_pre_delivery_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == "verified answer"
    assert result["completed"] is True
    assert agent._pre_delivery_nudges == 1
    # The withheld candidate was surfaced as an interim message; the delivered
    # answer is the final response, not an interim.
    assert emitted == ["unverified claim"]
    # The withheld draft is stripped from the returned history; only the
    # delivered answer survives.
    contents = [m.get("content") for m in result["messages"]]
    assert "unverified claim" not in contents
    assert "verified answer" in contents
    assert all(not m.get("_pre_delivery_synthetic") for m in result["messages"])


def test_bound_stops_at_configured_value(agent, monkeypatch):
    """A hook that always holds is capped by agent.max_pre_delivery_nudges."""
    agent.max_iterations = 4
    agent.iteration_budget.max_total = 4
    answers = iter([
        _response("candidate one"),
        _response("candidate two"),
        _response("candidate three"),
    ])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_delivery"),
        patch(
            "hermes_cli.plugins.get_pre_delivery_answer",
            return_value=("continue", "keep verifying"),
        ),
        patch("agent.verify_hooks.max_pre_delivery_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("answer me")

    # Two holds, then the third candidate is delivered.
    assert agent._pre_delivery_nudges == 2
    assert result["final_response"] == "candidate three"
    assert result["completed"] is True


def test_raising_hook_releases_the_turn(agent, monkeypatch):
    """Fail-open: a raising `pre_delivery` hook lets the turn finish."""
    agent._interruptible_api_call = lambda _kwargs: _response("the answer")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_delivery"),
        patch(
            "hermes_cli.plugins.get_pre_delivery_answer",
            side_effect=RuntimeError("hook crashed"),
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("answer me")

    assert result["final_response"] == "the answer"
    assert result["completed"] is True
    assert agent._pre_delivery_nudges == 0


class TestToolEvidence:
    """The turn's calls reach the hook as name, arguments, and an excerpt."""

    def test_a_call_carries_name_arguments_and_excerpt(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "line one"},
        ]
        assert verify_hooks.build_tool_evidence(messages) == [
            {"name": "read_file", "arguments": '{"path":"a"}', "excerpt": "line one"}
        ]

    def test_an_earlier_turn_is_excluded(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "tool", "tool_call_id": "old", "name": "old", "content": "old result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "second"},
        ]
        assert verify_hooks.build_tool_evidence(messages) == []

    def test_a_hold_nudge_does_not_move_the_boundary(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x"},
            {"role": "assistant", "content": "draft", "_pre_delivery_synthetic": True},
            {"role": "user", "content": "hold", "_pre_delivery_synthetic": True},
        ]
        assert len(verify_hooks.build_tool_evidence(messages)) == 1

    def test_an_excerpt_is_capped_head_and_tail(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "c1", "name": "t", "content": "a" * 5000},
        ]
        excerpt = verify_hooks.build_tool_evidence(messages)[0]["excerpt"]
        assert len(excerpt) == verify_hooks.TOOL_EXCERPT_CHARS + 3
        assert excerpt.startswith("a")
        assert excerpt.endswith("a")

    def test_only_the_newest_calls_fit_the_budget(self):
        messages = [{"role": "user", "content": "go"}]
        for index in range(verify_hooks.TOOL_EVIDENCE_CALLS + 5):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"c{index}",
                    "name": "t",
                    "content": "x" * 1024,
                }
            )
        evidence = verify_hooks.build_tool_evidence(messages)
        assert len(evidence) == verify_hooks.TOOL_EVIDENCE_CALLS
        # Newest excerpt fills first; the oldest ones keep their name only.
        assert evidence[-1]["excerpt"] == "x" * 1024
        assert evidence[0]["excerpt"] == ""


def test_a_replacement_answer_is_delivered(agent, monkeypatch):
    """A hook answering with replacement text ships that text, not the draft."""
    answers = iter([_response("draft with a claim"), _response("second draft")])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_delivery"),
        patch(
            "hermes_cli.plugins.get_pre_delivery_answer",
            side_effect=[("continue", "hold this"), ("replace", "cleaned answer")],
        ),
        patch("agent.verify_hooks.max_pre_delivery_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("answer me")

    assert result["final_response"] == "cleaned answer"
    assert result["completed"] is True
    contents = [m.get("content") for m in result["messages"]]
    assert "cleaned answer" in contents
    assert "second draft" not in contents
