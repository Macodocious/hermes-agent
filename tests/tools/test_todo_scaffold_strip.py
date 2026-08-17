"""Tests for P5: user-message scaffold stripping and the seeded-task rename nudge.

Covers ``strip_user_scaffold`` (structural, leading-only stripping of
gateway envelopes, reply pointers, system notes, timestamps, and the
sender-name prefix), the seeded-store lifecycle (``seed_from_user_message``,
flag persistence, nudge rendering, flag clear on write), and the capture
path's scaffold stripping.
"""

from __future__ import annotations

import pytest

from tools.todo_tool import (
    TodoStore,
    strip_user_scaffold,
)


class TestStripUserScaffold:
    def test_strips_discord_triggering_envelope(self):
        text = (
            "[Triggering message id: `1538797674298474547` — use as "
            "`message_id` for reply/react/pin via the discord tools.]\n\n"
            "[Mac] Regarding the stock-tracker plugin, what do you honestly "
            "think of the current algorithm for scoring?"
        )
        assert strip_user_scaffold(text) == (
            "Regarding the stock-tracker plugin, what do you honestly "
            "think of the current algorithm for scoring?"
        )

    def test_strips_envelope_without_sender_prefix(self):
        text = (
            "[Triggering message id: `123` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n"
            "Just the message"
        )
        assert strip_user_scaffold(text) == "Just the message"

    def test_strips_reply_to_pointer(self):
        text = '[Replying to: "what did you mean?"]\n\nMy answer'
        assert strip_user_scaffold(text) == "My answer"

    def test_strips_reply_to_own_message_pointer(self):
        text = '[Replying to your previous message: "ok"]\n\nNext'
        assert strip_user_scaffold(text) == "Next"

    def test_keeps_system_note_intact(self):
        # System notes are exclusion markers: the seed/capture callers
        # check the stripped text against the excluded prefixes and skip
        # the message entirely, so the marker must survive stripping.
        text = "[System note: A new message has arrived]\n\nDo the thing"
        assert strip_user_scaffold(text) == text

    def test_keeps_goal_continuation_intact(self):
        # Goal-loop continuations are exclusion markers, same as system
        # notes — never stripped.
        text = "[Continuing toward your standing goal]\nGoal: x\n\nNext step"
        assert strip_user_scaffold(text) == text

    def test_strips_timestamp_prefix(self):
        text = "[Wed 2026-08-13 09:15:00 UTC] Hello there"
        assert strip_user_scaffold(text) == "Hello there"

    def test_strips_iso_timestamp_prefix(self):
        text = "[2026-08-13T09:15:00+00:00] Hello there"
        assert strip_user_scaffold(text) == "Hello there"

    def test_strips_combined_envelope_then_sender(self):
        text = (
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n"
            "[Mac] Both in one PR"
        )
        assert strip_user_scaffold(text) == "Both in one PR"

    def test_keeps_bare_sender_prefix(self):
        # A bare "[Mac] ..." at the start is the user's own text on
        # platforms that do not add sender prefixes — never stripped.
        assert strip_user_scaffold("[Mac] Both in one PR") == "[Mac] Both in one PR"

    def test_keeps_mid_message_phrases(self):
        # Structural, leading-only: a similar phrase mid-message is content.
        text = "I said [Triggering message id: `1`] in my last message"
        assert strip_user_scaffold(text) == text

    def test_keeps_plain_text_unchanged(self):
        text = "Build the thing"
        assert strip_user_scaffold(text) == text

    def test_empty_and_whitespace(self):
        assert strip_user_scaffold("") == ""
        assert strip_user_scaffold("   ") == ""
        assert strip_user_scaffold(None) == ""

    def test_strips_envelope_plus_timestamp_plus_sender(self):
        text = (
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n"
            "[Wed 2026-08-13 09:15:00 UTC] [Mac] Do it"
        )
        assert strip_user_scaffold(text) == "Do it"


class TestSeedFromUserMessage:
    def test_seeds_stripped_content(self):
        store = TodoStore()
        item = store.seed_from_user_message(
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n[Mac] Build the thing"
        )
        assert item is not None
        assert item["content"] == "Build the thing"
        assert item["status"] == "in_progress"
        assert item["source"] == "user"
        assert store._seeded is True

    def test_no_seed_when_store_has_items(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        assert store.seed_from_user_message("New request") is None
        assert store._seeded is False

    def test_no_seed_for_empty_after_strip(self):
        store = TodoStore()
        assert store.seed_from_user_message(
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]"
        ) is None
        assert store.read() == []
        assert store._seeded is False

    def test_write_clears_seeded_flag(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        assert store._seeded is True
        store.write([{"id": "1", "content": "Renamed task", "status": "in_progress"}])
        assert store._seeded is False

    def test_seeded_flag_round_trips_through_json(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        restored = TodoStore.from_json(store.to_json())
        assert restored._seeded is True
        assert restored.read()[0]["content"] == "Build the thing"

    def test_old_json_without_seeded_key_defaults_false(self):
        import json

        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        data = json.loads(store.to_json())
        del data["seeded"]
        restored = TodoStore.from_json(json.dumps(data))
        assert restored._seeded is False


class TestRenameNudge:
    def test_nudge_renders_while_seeded(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        block = store.format_for_turn()
        assert block is not None
        assert "[Active tasks]" in block
        assert "Build the thing" in block
        assert "Rename the seeded task" in block

    def test_no_nudge_after_model_write(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        store.write([{"id": "1", "content": "Renamed", "status": "in_progress"}])
        block = store.format_for_turn() or ""
        assert "Rename the seeded task" not in block

    def test_no_nudge_when_not_seeded(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        block = store.format_for_turn() or ""
        assert "Rename the seeded task" not in block

    def test_no_nudge_when_seeded_store_has_extra_items(self):
        # Defensive guard: a store restored with seeded=True but multiple
        # items (e.g. corrupted persisted state) must not render the nudge.
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        store.write(
            [{"id": "2", "content": "Second", "status": "pending"}],
            merge=True,
        )
        store._seeded = True  # simulate the defensive state directly
        block = store.format_for_turn() or ""
        assert "Rename the seeded task" not in block


class TestCaptureStripsScaffold:
    def test_capture_strips_envelope_and_sender(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        capture = store.capture_request(
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n[Mac] Also do this"
        )
        assert capture is not None
        assert capture["content"] == "Also do this"

    def test_capture_rejects_scaffold_only(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        capture = store.capture_request(
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]"
        )
        assert capture is None
