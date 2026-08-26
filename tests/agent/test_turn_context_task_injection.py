"""Tests for the P2/P3 turn prologue injection (agent/turn_context.py).

Exercises ``_apply_task_injection_and_capture`` and the capture rule
directly, plus the build_turn_context integration through the existing
``_build`` helper pattern. All structural: no natural-language parsing.
"""

from __future__ import annotations

import types

import pytest

from agent.turn_context import (
    _apply_task_injection_and_capture,
    _last_assistant_issued_clarify,
    _seed_todo_store_from_user_message,
    _should_capture_user_request,
)
from tools.todo_tool import TodoStore


class TestShouldCaptureUserRequest:
    def test_captures_when_active_items_and_no_clarify(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        messages = [
            {"role": "user", "content": "fix the thing"},
            {"role": "assistant", "content": "on it"},
        ]
        assert _should_capture_user_request(messages, "also do this", store) is True

    def test_no_capture_when_list_empty(self):
        store = TodoStore()
        messages = [{"role": "user", "content": "hi"}]
        assert _should_capture_user_request(messages, "do this", store) is False

    def test_no_capture_when_last_assistant_asked_clarify(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-1", "type": "function",
                 "function": {"name": "clarify", "arguments": "{}"}}
            ]},
        ]
        assert _should_capture_user_request(messages, "my answer", store) is False

    def test_no_capture_for_system_note_prefix(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        messages = [{"role": "user", "content": "old"}]
        assert _should_capture_user_request(
            messages, "[System note: A new message has arrived]", store
        ) is False

    def test_no_capture_for_goal_continuation_prefix(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        messages = [{"role": "user", "content": "old"}]
        assert _should_capture_user_request(
            messages, "[Continuing toward your standing goal]\nGoal: x", store
        ) is False

    def test_no_capture_for_empty_message(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        messages = []
        assert _should_capture_user_request(messages, "   ", store) is False

    def test_no_capture_when_message_equals_seeded_task(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        messages = [{"role": "user", "content": "old"}]
        # The seed already made this exact message the active task; capturing
        # it again would duplicate it as both a task and a candidate.
        assert _should_capture_user_request(messages, "Build the thing", store) is False

    def test_captures_identical_message_after_seed_renamed(self):
        store = TodoStore()
        store.seed_from_user_message("Build the thing")
        # A model write clears the seeded state; the same text later is a
        # genuine re-request, not the seed duplicate.
        store.write([{"id": "1", "content": "Build the thing", "status": "in_progress"}])
        messages = [{"role": "user", "content": "old"}]
        assert _should_capture_user_request(messages, "Build the thing", store) is True

    def test_clarify_detection_with_object_tool_calls(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        tool_call = types.SimpleNamespace(
            id="call-1",
            type="function",
            function=types.SimpleNamespace(name="clarify", arguments="{}"),
        )
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        ]
        assert _last_assistant_issued_clarify(messages) is True


class TestApplyTaskInjection:
    def test_injects_block_prefixing_user_message(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Working", "status": "in_progress"}])
        agent = _agent_with_store(store)
        user_msg = {"role": "user", "content": "hello"}
        _apply_task_injection_and_capture(agent, [], user_msg)
        assert user_msg["content"].startswith("[Active tasks]")
        assert user_msg["content"].endswith("hello")

    def test_no_injection_when_store_empty(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        user_msg = {"role": "user", "content": "hello"}
        _apply_task_injection_and_capture(agent, [], user_msg)
        assert user_msg["content"] == "hello"

    def test_no_injection_when_no_store(self):
        agent = types.SimpleNamespace()
        user_msg = {"role": "user", "content": "hello"}
        _apply_task_injection_and_capture(agent, [], user_msg)  # must not raise
        assert user_msg["content"] == "hello"

    def test_captures_qualifying_turn_and_injects(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        agent = _agent_with_store(store)
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ok"},
        ]
        user_msg = {"role": "user", "content": "NEW REQUEST"}
        _apply_task_injection_and_capture(agent, messages, user_msg)
        assert len(store.pending_captures()) == 1
        assert store.pending_captures()[0]["content"] == "NEW REQUEST"
        assert "[Captured requests]" in user_msg["content"]

    def test_multimodal_content_is_prefixed(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        agent = _agent_with_store(store)
        user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}],
        }
        _apply_task_injection_and_capture(agent, [], user_msg)
        content = user_msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "[Active tasks]" in content[0]["text"]
        # Original parts preserved after the injected block.
        assert content[-1]["type"] == "image_url"


def _agent_with_store(store):
    return types.SimpleNamespace(_todo_store=store)


class TestSeedTodoStoreFromUserMessage:
    def test_seeds_empty_store_with_user_message(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "Build the thing")
        items = store.read()
        assert len(items) == 1
        assert items[0]["content"] == "Build the thing"
        # Seeds are born pending: the lifecycle begins with an explicit
        # begin transition, never an implicit current task.
        assert items[0]["status"] == "pending"
        assert items[0]["source"] == "user"
        assert store._seeded is True

    def test_seed_strips_gateway_scaffold(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(
            agent,
            "[Triggering message id: `1` — use as `message_id` for "
            "reply/react/pin via the discord tools.]\n\n"
            "[Mac] Build the thing",
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["content"] == "Build the thing"
        assert "Triggering message id" not in items[0]["content"]
        assert "[Mac]" not in items[0]["content"]

    def test_no_seed_when_store_has_items(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "busy", "status": "in_progress"}])
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "New request")
        assert len(store.read()) == 1
        assert store.read()[0]["content"] == "busy"

    def test_no_seed_when_no_store(self):
        agent = types.SimpleNamespace()  # no _todo_store
        _seed_todo_store_from_user_message(agent, "hi")  # must not raise

    def test_no_seed_for_empty_message(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "   ")
        assert store.read() == []

    def test_no_seed_for_system_note_prefix(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "[System note: A new message has arrived]")
        assert store.read() == []

    def test_seeded_item_survives_replace_write(self):
        # P4: a user-sourced active item is append-only — a model replace
        # that omits it must not silently drop it.
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "Build the thing")
        store.write([{"id": "2", "content": "Different plan", "status": "pending"}])
        contents = [item["content"] for item in store.read()]
        assert "Build the thing" in contents
        assert "Different plan" in contents

    def test_seeded_item_round_trips_through_json(self):
        store = TodoStore()
        agent = _agent_with_store(store)
        _seed_todo_store_from_user_message(agent, "Build the thing")
        restored = TodoStore.from_json(store.to_json())
        items = restored.read()
        assert len(items) == 1
        assert items[0]["content"] == "Build the thing"
        assert items[0]["source"] == "user"
