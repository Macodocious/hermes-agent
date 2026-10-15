"""Reasoning repetition-loop detection and early abort.

Pins the contract:

- ``_reasoning_loop_test`` names the test that fired, or returns None.
- A stream whose reasoning repeats is abandoned mid-flight: the response is
  stamped ``finish_reason="length"`` and carries the name of the test that
  fired, and the remaining chunks are never consumed.
- A stream whose reasoning does not repeat is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import (
    FINISH_REASON_LENGTH,
    REASONING_LOOP_ABORT_ATTR,
    REASONING_LOOP_REISSUE_ATTR,
    REASONING_LOOP_REISSUE_TEMPERATURE_DELTA,
    REASONING_LOOP_SPAN_CHARS,
)
from agent.chat_completion_helpers import (
    _effective_call_temperature,
    _reasoning_loop_test,
    _resolve_effective_temperature,
)


def _make_reasoning_chunk(reasoning=None, content=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=None,
        reasoning_content=reasoning, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._fire_reasoning_delta = lambda text: None
    agent._fire_stream_delta = lambda text: None
    return agent


class TestReasoningLoopTest:
    def test_span_fires_on_a_repeated_block(self):
        block = "".join(
            f"line {i} of the reasoning. " for i in range(100)
        )[:REASONING_LOOP_SPAN_CHARS]
        assert len(block) == REASONING_LOOP_SPAN_CHARS
        assert _reasoning_loop_test(block + block) == "span"

    def test_cycle_fires_on_the_degenerate_spam(self):
        assert _reasoning_loop_test("hmm: hmm, " * 40) == "cycle"

    def test_clean_reasoning_does_not_fire(self):
        text = "".join(
            f"step {i}: check the next item and move on. " for i in range(200)
        )
        assert _reasoning_loop_test(text) is None

    def test_incrementing_reasoning_does_not_fire(self):
        # Known gap: a loop whose every iteration differs (a counter advancing)
        # contains no verbatim repeat, so neither test sees it.
        text = "".join(f"browser_activate_agent_another_{i}, " for i in range(2000))
        assert _reasoning_loop_test(text) is None

    def test_short_text_does_not_fire(self):
        assert _reasoning_loop_test("") is None
        assert _reasoning_loop_test("hmm: hmm, ") is None


class TestStreamingAbort:
    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_repeating_reasoning_aborts_the_stream(
        self, _mock_close, mock_create, monkeypatch
    ):
        consumed = {"chunks": 0}

        def _looping_stream():
            for _ in range(2000):
                consumed["chunks"] += 1
                yield _make_reasoning_chunk(reasoning="hmm: hmm, ")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _looping_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].finish_reason == FINISH_REASON_LENGTH
        assert getattr(response, REASONING_LOOP_ABORT_ATTR) == "cycle"
        assert consumed["chunks"] < 2000, "the stream must be abandoned, not drained"

    @patch("run_agent.AIAgent._create_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    def test_clean_reasoning_is_untouched(
        self, _mock_close, mock_create, monkeypatch
    ):
        def _clean_stream():
            for i in range(200):
                yield _make_reasoning_chunk(reasoning=f"step {i}: check the next item. ")
            yield _make_reasoning_chunk(content="done", finish_reason="stop")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: _clean_stream()
        mock_create.return_value = mock_client

        agent = _make_agent()
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].finish_reason == "stop"
        assert getattr(response, REASONING_LOOP_ABORT_ATTR, None) is None


LOOP_TEXT = "LOOPING-REASONING-MARKER " * 20


class TestReissueAfterAbort:
    """A loop-aborted turn is reissued, and the aborted generation is dropped."""

    @patch("run_agent.AIAgent._build_system_prompt")
    @patch("run_agent.AIAgent._interruptible_streaming_api_call")
    @patch("run_agent.AIAgent._interruptible_api_call")
    def test_loop_abort_reissues_the_turn(self, mock_api, mock_stream, mock_sys):
        from run_agent import AIAgent

        mock_sys.return_value = "system prompt"

        def _response(content, finish_reason, reasoning=None, loop_test=None):
            choice = MagicMock()
            choice.message.content = content
            choice.message.tool_calls = None
            choice.message.refusal = None
            choice.message.reasoning_content = reasoning
            choice.message.reasoning = None
            choice.message.reasoning_details = None
            choice.finish_reason = finish_reason
            response = MagicMock()
            response.choices = [choice]
            response.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            response.model = "test-model"
            response.id = "test-id"
            response.reasoning_loop_aborted = loop_test
            return response

        aborted = _response(
            None, FINISH_REASON_LENGTH, reasoning=LOOP_TEXT, loop_test="cycle"
        )
        answered = _response("the answer", "stop")
        mock_stream.side_effect = [aborted, answered]
        mock_api.side_effect = [aborted, answered]

        agent = AIAgent(
            model="test/model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
        )
        agent.client = MagicMock()

        result = agent.run_conversation(
            user_message="do the thing", conversation_history=[]
        )

        assert mock_api.call_count + mock_stream.call_count >= 2, (
            "the aborted turn must be reissued"
        )
        assert result.get("final_response") == "the answer"
        for msg in result.get("messages", []):
            assert LOOP_TEXT not in str(msg.get("content") or ""), (
                "the aborted generation must not be appended to the conversation"
            )
            assert LOOP_TEXT not in str(msg.get("reasoning") or "")


class TestReissueSampling:
    def test_a_reissued_call_is_sampled_at_n_plus_point_one(self):
        """The reissue must not repeat the deterministic call that looped."""
        agent = _make_agent()
        setattr(agent, "temperature", 0.0)
        assert _resolve_effective_temperature(agent) == 0.0
        setattr(agent, REASONING_LOOP_REISSUE_ATTR, True)
        assert _effective_call_temperature(agent) == (
            0.0 + REASONING_LOOP_REISSUE_TEMPERATURE_DELTA
        )
        assert _resolve_effective_temperature(agent) == 0.0, (
            "the configured temperature is untouched"
        )
        assert _effective_call_temperature(agent) == 0.0, (
            "the reissue mark is consumed by the call it marks"
        )
