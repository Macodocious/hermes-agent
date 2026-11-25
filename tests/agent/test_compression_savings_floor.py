"""A compaction that frees almost nothing must not be committed, and the
provider's next comfortable prompt reading must not re-arm it.

Incident shape (2026-11-24): a ~300K-token transcript compacted 46 times in
thirty minutes.  Every pass assembled a summary as large as the window it
replaced, so nothing was actually freed; the turn's own provider call then
reported a prompt comfortably under the threshold, which cleared
``_ineffective_compression_count`` and re-armed the next pass.

The guard has to survive that reading, because the reading is what every turn
produces.  A pass that frees less than the savings floor is refused (the
transcript comes back unchanged) and counted in a counter the reading never
touches; two of those block automatic compaction for the session.

Only the two public entry points of the compressor are exercised here --
``compress()`` and the provider reading via ``update_from_response()`` -- so the
test fails for the loop's own reason rather than for internals drift.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make_compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test-model",
        threshold_percent=0.65,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    defaults.update(kwargs)
    with patch("agent.context_compressor.get_model_context_length", return_value=96000):
        return ContextCompressor(**defaults)


def _build_session(n_turns: int, words_per_turn: int = 40) -> list:
    base_text = " ".join(["a"] * words_per_turn)
    messages = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base_text} (user turn {i})"})
        messages.append({"role": "assistant", "content": f"{base_text} (assistant turn {i})"})
    return messages


def _transcript_chars(messages: list) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


class TestSubFloorCompactionIsRefused:
    def test_worthless_pass_does_not_rearm_after_a_comfortable_provider_reading(self):
        """Two worthless passes, each followed by a comfortable reading, must block."""
        comp = _make_compressor(config_context_length=96000, summary_target_ratio=0.45)
        messages = _build_session(30, words_per_turn=2400)
        comp.last_prompt_tokens = 73_000
        # A summary larger than the transcript: the pass frees nothing and adds its
        # own overhead -- the incident's 40K-character handoff summary (415 -> 416
        # messages) landing in a window it cannot shrink.
        huge_summary = "x" * (_transcript_chars(messages) * 2)

        with patch.object(comp, "_generate_summary", return_value=huge_summary):
            for _ in range(2):
                comp.compress(messages, current_tokens=73_000)
                # The fake provider: comfortable, well under the 72K threshold.
                comp.update_from_response({"prompt_tokens": 40_000})

            assert not comp.should_compress(73_000), (
                "a transcript that cannot be improved must stay blocked after the "
                "provider reports a comfortable prompt; otherwise the next turn "
                "compacts again and the loop runs for as long as the session does"
            )
            assert comp._no_gain_compression_count >= 2

    def test_refused_pass_returns_the_transcript_unchanged(self):
        """A refusal commits nothing: same list contents, no summary inserted."""
        comp = _make_compressor(config_context_length=96000, summary_target_ratio=0.45)
        messages = _build_session(30, words_per_turn=2400)
        original = [dict(m) for m in messages]
        huge_summary = "x" * (_transcript_chars(messages) * 2)

        with patch.object(comp, "_generate_summary", return_value=huge_summary):
            result = comp.compress(messages, current_tokens=73_000)

        assert result == original, "a refused pass must not rewrite the transcript"
        assert comp._last_compression_made_progress is False


class TestFloorClearingCompactionStillWorks:
    def test_a_real_compaction_is_not_refused(self):
        """The refusal must not cost the normal case: a small summary still compacts."""
        comp = _make_compressor(config_context_length=96000, summary_target_ratio=0.45)
        messages = _build_session(30, words_per_turn=2400)

        with patch.object(comp, "_generate_summary", return_value="short summary"):
            result = comp.compress(messages, current_tokens=73_000)

        assert len(result) < len(messages)
        assert comp._no_gain_compression_count == 0
