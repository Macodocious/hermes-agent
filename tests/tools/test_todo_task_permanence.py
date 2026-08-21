"""Tests for the task-permanence store contract (P1/P3/P4).

Covers the TodoStore additions: to_json/from_json persistence round-trip,
the P3 captured-request buffer (capture, bounds, disposition), the P4
user-source provenance (append-only protection), and the P2 per-turn
injection renderer (format_for_turn).
"""

from __future__ import annotations

import pytest

from tools.todo_tool import (
    MAX_CAPTURED_REQUESTS,
    MAX_TODO_CONTENT_CHARS,
    TodoStore,
    todo_tool,
)


class TestToJsonRoundTrip:
    def test_empty_store_round_trip(self):
        store = TodoStore()
        restored = TodoStore.from_json(store.to_json())
        assert restored.read() == []
        assert restored.pending_captures() == []

    def test_items_round_trip(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Ship the feature", "status": "in_progress"},
            {"id": "2", "content": "Write docs", "status": "pending"},
        ])
        restored = TodoStore.from_json(store.to_json())
        assert restored.read() == store.read()

    def test_captures_round_trip(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        store.capture_request("Also fix the bug")
        restored = TodoStore.from_json(store.to_json())
        pending = restored.pending_captures()
        assert len(pending) == 1
        assert pending[0]["content"] == "Also fix the bug"
        assert pending[0]["source"] == "user"
        assert pending[0]["status"] == "captured"

    def test_capture_counter_survives_round_trip(self):
        store = TodoStore()
        store.capture_request("one")
        store.capture_request("two")
        restored = TodoStore.from_json(store.to_json())
        # New captures after restore must not collide with restored ids.
        restored.capture_request("three")
        ids = [c["id"] for c in restored._captures]
        assert len(ids) == len(set(ids))

    def test_malformed_json_raises_value_error(self):
        with pytest.raises(ValueError):
            TodoStore.from_json("not json")
        with pytest.raises(ValueError):
            TodoStore.from_json('{"items": "nope"}')

    def test_from_json_revalidates_items(self):
        # Oversized content is capped again on load; invalid statuses reset.
        store = TodoStore()
        store.write([{"id": "1", "content": "A" * (MAX_TODO_CONTENT_CHARS + 100), "status": "pending"}])
        restored = TodoStore.from_json(store.to_json())
        assert len(restored.read()[0]["content"]) <= MAX_TODO_CONTENT_CHARS


class TestCaptureBuffer:
    def test_capture_requires_content(self):
        store = TodoStore()
        assert store.capture_request("   ") is None
        assert store.pending_captures() == []

    def test_capture_adds_candidate(self):
        store = TodoStore()
        capture = store.capture_request("Fix the bug")
        assert capture is not None
        assert capture["status"] == "captured"
        assert capture["source"] == "user"
        assert capture["id"] == "1"
        assert "captured_at" in capture
        assert len(store.pending_captures()) == 1

    def test_capture_truncates_long_content(self):
        store = TodoStore()
        store.capture_request("X" * (MAX_TODO_CONTENT_CHARS + 100))
        pending = store.pending_captures()
        assert len(pending[0]["content"]) <= MAX_TODO_CONTENT_CHARS
        assert pending[0]["content"].endswith("[truncated]")

    def test_capture_buffer_is_bounded(self):
        store = TodoStore()
        for i in range(MAX_CAPTURED_REQUESTS + 5):
            store.capture_request(f"request {i}")
        assert len(store._captures) <= MAX_CAPTURED_REQUESTS
        # The most recent candidates survive; the oldest were dropped.
        pending = store.pending_captures()
        assert pending[-1]["content"] == f"request {MAX_CAPTURED_REQUESTS + 4}"


class TestDisposition:
    def test_disposition_by_list(self):
        store = TodoStore()
        store.capture_request("one")
        store.capture_request("two")
        pending = store.pending_captures()
        remaining = store.disposition_captures([
            {"id": pending[0]["id"], "status": "merged"},
        ])
        assert len(remaining) == 1
        assert remaining[0]["id"] == pending[1]["id"]

    def test_disposition_by_mapping(self):
        store = TodoStore()
        store.capture_request("one")
        capture = store.pending_captures()[0]
        remaining = store.disposition_captures({capture["id"]: "rejected"})
        assert remaining == []

    def test_unknown_ids_and_statuses_ignored(self):
        store = TodoStore()
        store.capture_request("one")
        store.disposition_captures([{"id": "nope", "status": "merged"}])
        store.disposition_captures([{"id": "nope", "status": "bogus"}])
        assert len(store.pending_captures()) == 1

    def test_invalid_disposition_payload_ignored(self):
        store = TodoStore()
        store.capture_request("one")
        assert len(store.disposition_captures("garbage")) == 1
        assert len(store.disposition_captures([42])) == 1

    def test_all_outcomes_are_valid(self):
        store = TodoStore()
        for outcome in ("merged", "addressed", "rejected"):
            store.capture_request(outcome)
        captures = store._captures
        store.disposition_captures([
            {"id": captures[0]["id"], "status": "merged"},
            {"id": captures[1]["id"], "status": "addressed"},
            {"id": captures[2]["id"], "status": "rejected"},
        ])
        assert store.pending_captures() == []


class TestProvenance:
    def test_user_source_tag_survives_write(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "User task", "status": "pending", "source": "user"},
        ])
        assert store.read()[0]["source"] == "user"

    def test_agent_items_stay_untagged(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Agent task", "status": "pending"}])
        assert "source" not in store.read()[0]

    def test_replace_cannot_silently_drop_active_user_item(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "User task", "status": "in_progress", "source": "user"},
        ])
        # Model replaces the list with only its own new plan.
        store.write([{"id": "9", "content": "New plan", "status": "pending"}])
        items = store.read()
        contents = [item["content"] for item in items]
        assert "User task" in contents  # user item re-appended
        assert "New plan" in contents
        assert items[-1]["content"] == "User task"  # appended at the tail
        # Sequential numbering schema: ids are 1..N in list order.
        assert [item["id"] for item in items] == ["1", "2"]

    def test_user_item_can_be_marked_completed(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "User task", "status": "pending", "source": "user"},
        ])
        store.write([{"id": "1", "status": "completed"}], merge=True)
        assert store.read()[0]["status"] == "completed"

    def test_completed_user_item_is_not_restored(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Done task", "status": "completed", "source": "user"},
        ])
        store.write([{"id": "9", "content": "New plan", "status": "pending"}])
        contents = [item["content"] for item in store.read()]
        assert "Done task" not in contents  # completed user items may leave the list


class TestFormatForTurn:
    def test_empty_store_returns_none(self):
        store = TodoStore()
        assert store.format_for_turn() is None

    def test_active_items_rendered_with_current_task(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Waiting", "status": "pending"},
            {"id": "2", "content": "Working", "status": "in_progress"},
        ])
        text = store.format_for_turn()
        assert text is not None
        assert "[Active tasks]" in text
        assert "- [ ] Waiting" in text
        assert "← CURRENT TASK" not in text
        assert "Working" in text

    def test_completed_items_not_rendered(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Done", "status": "completed"},
            {"id": "2", "content": "Active", "status": "pending"},
        ])
        text = store.format_for_turn()
        assert text is not None
        assert "Done" not in text
        assert "Active" in text

    def test_captures_rendered_with_disposition_nudge(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Busy", "status": "in_progress"}])
        store.capture_request("New request")
        text = store.format_for_turn()
        assert text is not None
        assert "[Captured requests]" in text
        assert "- [ ] New request (captured)" in text
        assert "Disposition each captured request" in text

    def test_captures_without_items_still_render(self):
        store = TodoStore()
        store.capture_request("Only a capture")
        text = store.format_for_turn()
        assert text is not None
        assert "[Captured requests]" in text

    def test_dispositioned_captures_not_rendered(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Busy", "status": "in_progress"}])
        store.capture_request("Handled")
        capture = store._captures[0]
        store.disposition_captures([{"id": capture["id"], "status": "addressed"}])
        text = store.format_for_turn()
        assert text is not None
        assert "[Captured requests]" not in text


class TestTodoToolDispositions:
    def test_tool_accepts_dispositions(self):
        store = TodoStore()
        store.capture_request("one")
        capture = store._captures[0]
        result = todo_tool(
            dispositions=[{"id": capture["id"], "status": "merged"}],
            store=store,
        )
        import json
        payload = json.loads(result)
        assert payload["captures"] == []

    def test_tool_response_includes_pending_captures(self):
        store = TodoStore()
        store.capture_request("one")
        import json
        payload = json.loads(todo_tool(store=store))
        assert len(payload["captures"]) == 1
        assert payload["captures"][0]["content"] == "one"
