"""Regression test: DeepSeek V4 typed thinking blocks must not leak into content.

DeepSeek V4 (via Ollama Cloud) returns assistant content as a list of typed
blocks:

    [{"type": "thinking", "thinking": "..."}, {"type": "output", "text": "..."}]

``extract_reasoning`` captures the thinking text into the reasoning field, but
``flatten_message_text`` does not exclude ``"thinking"`` blocks, so the
deliberation previously flattened into visible content and leaked to storage,
transcripts, and gateway delivery (observed: 14 Discord chunks of
chain-of-thought).

The fix strips ``type == "thinking"`` blocks from the content list in
``build_assistant_message`` before flattening, gated on
``_needs_deepseek_tool_reasoning()`` so no other provider's content handling
changes.
"""

from __future__ import annotations

from types import SimpleNamespace

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _typed_blocks(
    thinking: str = "Let me think about this carefully...",
    output: str = "Here is the answer.",
) -> list[dict]:
    # The leak shape: the thinking block carries the "text" key (in
    # flatten_message_text's _TEXT_KEYS), so without the strip it flattens
    # into visible content. A "thinking"-keyed block is dropped by flatten
    # for every provider and would not exercise the fix.
    return [
        {"type": "thinking", "text": thinking},
        {"type": "output", "text": output},
    ]


def _assistant_message(content) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        codex_reasoning_items=None,
        codex_message_items=None,
        tool_calls=None,
    )


class TestDeepSeekThinkingBlockStrip:
    """build_assistant_message strips typed thinking blocks for DeepSeek."""

    def test_deepseek_thinking_block_stripped_from_content(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = _assistant_message(_typed_blocks())

        msg = agent._build_assistant_message(assistant_message, "stop")

        assert msg["content"] == "Here is the answer."
        assert "Let me think about this carefully" in msg["reasoning"]

    def test_deepseek_plain_string_content_unchanged(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = _assistant_message("Plain string response.")

        msg = agent._build_assistant_message(assistant_message, "stop")

        assert msg["content"] == "Plain string response."

    def test_non_deepseek_typed_blocks_unchanged(self) -> None:
        agent = _make_agent(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
        )
        assistant_message = _assistant_message(_typed_blocks())

        msg = agent._build_assistant_message(assistant_message, "stop")

        assert (
            msg["content"]
            == "Let me think about this carefully...\nHere is the answer."
        )

    def test_deepseek_non_thinking_blocks_preserved(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = _assistant_message(
            [
                {"type": "text", "text": "First part."},
                {"type": "output", "text": "Second part."},
            ]
        )

        msg = agent._build_assistant_message(assistant_message, "stop")

        assert msg["content"] == "First part.\nSecond part."

    def test_deepseek_thinking_type_case_and_whitespace_insensitive(self) -> None:
        agent = _make_agent(provider="deepseek", model="deepseek-v4-flash")
        assistant_message = _assistant_message(
            [
                {"type": " Thinking ", "thinking": "Deliberation."},
                {"type": "output", "text": "Answer."},
            ]
        )

        msg = agent._build_assistant_message(assistant_message, "stop")

        assert msg["content"] == "Answer."
