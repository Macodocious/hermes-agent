from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    GatewayRunner,
    _DEAD_CHANNEL_ERROR_MARKERS,
    _is_dead_channel_send_failure,
)
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE


class FakeFailedAdapter:
    """Minimal adapter: send always fails with the given error text."""

    def __init__(self, error_text: str = "Unknown Channel"):
        self.error_text = error_text
        self.calls = []
        self.callbacks = {}
        self._pending_messages = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=False, error=self.error_text)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


def _goal_continuation_event(source, goal="finish the task"):
    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal=goal),
        message_type=MessageType.TEXT,
        source=source,
    )


def _runner_with_engine(session_key: str) -> GatewayRunner:
    """Bare GatewayRunner with the engine state an active session would have."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {session_key: object()}
    runner._running_agents_ts = {session_key: 1.0}
    runner._queued_events = {}
    runner._pending_messages = {}
    return runner


def test_dead_channel_marker_matches_discord_10003_and_false_positives():
    """Regression: Discord's 10003 ('Unknown Channel'/'Unknown Thread') must
    be recognized as a dead delivery channel.

    classify_send_error only matches ``"chat not found"`` for chat-level
    failures, so 10003 was classified ``unknown`` and the orphaned session
    engine kept hammering the deleted channel.  The dedicated marker set is
    the eviction trigger; it must also stay quiet for non-terminal failures.
    """
    assert _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="Unknown Channel")
    )
    assert _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="404: Unknown Thread (10003)")
    )
    assert _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="chat not found")
    )
    assert _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="Bad Request: Unknown Channel")
    )

    # False positives must not evict.
    assert not _is_dead_channel_send_failure(SimpleNamespace(success=True))
    assert not _is_dead_channel_send_failure(None)
    assert not _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="rate limited (429)")
    )
    assert not _is_dead_channel_send_failure(
        SimpleNamespace(success=False, error="Unauthorized (403)")
    )

    assert "unknown channel" in _DEAD_CHANNEL_ERROR_MARKERS


@pytest.mark.asyncio
async def test_handle_dead_channel_eviction_tears_down_engine_and_is_idempotent():
    """Regression: a channel-gone send failure must stop the orphaned engine.

    The session row/transcript is preserved (SessionStore has no delete API),
    but the running agent, FIFO goal continuations, and the running slot must
    be released so heartbeat/typing/status sends stop hammering the deleted
    channel.  A second eviction for the same session is a no-op.
    """
    session_key = "discord:parent-channel:thread-123"
    runner = _runner_with_engine(session_key)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )

    # Adapter present: FIFO continuations are cleared through it.
    adapter = FakeFailedAdapter()
    adapter._pending_messages[session_key] = _goal_continuation_event(source)
    runner._queued_events[session_key] = [
        _goal_continuation_event(source, goal="queued continuation")
    ]
    runner.adapters = {Platform.DISCORD: adapter}

    first = await runner._handle_dead_channel_eviction(
        session_key, source, error_text="Unknown Channel"
    )

    assert first is True
    assert runner._session_is_dead(session_key) is True
    # Running slot released.
    assert session_key not in runner._running_agents
    assert session_key not in runner._running_agents_ts
    # FIFO goal continuations dropped; the transcript key stays out of the queue.
    assert adapter._pending_messages.get(session_key) is None
    assert runner._queued_events.get(session_key) in (None, [])

    # Idempotent: a second call for the already-dead session is a no-op.
    second = await runner._handle_dead_channel_eviction(
        session_key, source, error_text="Unknown Channel"
    )
    assert second is False


@pytest.mark.asyncio
async def test_goal_status_notice_failure_triggers_dead_channel_eviction():
    """Regression: a failed goal-status send to a deleted channel must evict.

    The post-delivery callback fires even when the send failed, so
    _send_goal_status_notice sees the 10003 failure and must tear the
    orphaned engine down rather than leave the heartbeat and status loop
    spinning against the missing channel.
    """
    session_key = "discord:parent-channel:thread-123"
    runner = _runner_with_engine(session_key)

    adapter = FakeFailedAdapter(error_text="Unknown Channel")
    runner.adapters = {Platform.DISCORD: adapter}
    runner.session_store = SimpleNamespace(
        _generate_session_key=lambda source: session_key
    )
    # Eviction resolves session_id through the store lock helper; a bare
    # namespace has neither _lock nor _entries, so fall back on '' (no armed
    # goal) — the engine teardown still happens.
    runner.session_store._lock = None
    runner.session_store._entries = {}

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="parent-channel",
        thread_id="thread-123",
    )

    await runner._send_goal_status_notice(source, "⏳ Goal parked — waiting on user")

    assert adapter.calls == [
        {
            "chat_id": "parent-channel",
            "content": "⏳ Goal parked — waiting on user",
            "reply_to": None,
            "metadata": {"thread_id": "thread-123"},
        }
    ]
    assert runner._session_is_dead(session_key) is True
    assert session_key not in runner._running_agents
