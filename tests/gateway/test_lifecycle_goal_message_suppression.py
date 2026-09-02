"""Tests for task-lifecycle goal message suppression (PR #51 instances).

Task-lifecycle goals are armed by the todo tool with the exact
"Complete the task: <content>" text. For those goals only, the per-turn
judge progress lines ("↻ Continuing", "⏳ Goal parked") are suppressed —
the todo tool already surfaces Started/Completed/Stopped bubbles — and the
done verdict renders as "✅ Task completed: <name>". Native /goal verdicts
are untouched.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


def _make_runner_with_adapter(session_id: str = None):
    from gateway.run import GatewayRunner
    import uuid

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    src = _make_source()
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


@pytest.mark.asyncio
async def test_lifecycle_goal_continue_suppresses_progress_line(hermes_home):
    """A lifecycle goal's per-turn '↻ Continuing' line must not be sent.

    The judge still runs and the continuation prompt still enqueues — only
    the chat delivery of the progress line is suppressed.
    """
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still working", False, None, False, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await asyncio.sleep(0.05)

    assert adapter.sends == [], f"expected no progress line, got {adapter.sends}"
    assert adapter._pending_messages, "continuation prompt must still be enqueued"


@pytest.mark.asyncio
async def test_lifecycle_goal_done_rewrites_completed_message(hermes_home):
    """A lifecycle goal's done verdict renders '✅ Task completed: <name>'
    instead of the native '✓ Goal achieved' line."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
        )
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    msg = adapter.sends[0]
    assert "✅ Task completed: ship the feature" in msg["content"]
    assert "Goal achieved" not in msg["content"]


@pytest.mark.asyncio
async def test_native_goal_continue_keeps_progress_line(hermes_home):
    """Native /goal verdicts are untouched: '↻ Continuing' still reaches the
    user for goals that do not carry the lifecycle prefix."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False, None, False, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    assert adapter._pending_messages, "continuation prompt must be enqueued"


@pytest.mark.asyncio
async def test_lifecycle_goal_done_keeps_final_response(hermes_home):
    """A lifecycle goal's done verdict must NOT suppress the turn's final
    response — the agent's prose ships via the conversation, and the
    deterministic '✅ Task completed' line lands after it (deferred to
    post-delivery)."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
        )
        await asyncio.sleep(0.05)

    assert suppress is None
    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    assert "✅ Task completed: ship the feature" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_native_goal_done_returns_no_suppress_flag(hermes_home):
    """Native /goal verdicts are untouched: a done verdict must NOT
    suppress the turn's final response."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "docs polished", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I polished the docs.",
        )
        await asyncio.sleep(0.05)

    assert suppress is None
    assert len(adapter.sends) == 1
    assert "Goal achieved" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_lifecycle_goal_continue_returns_no_suppress_flag(hermes_home):
    """A lifecycle goal still in progress must NOT suppress the turn's
    final response — only the done verdict does."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still working", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await asyncio.sleep(0.05)

    assert suppress is None
    assert adapter.sends == [], f"expected no progress line, got {adapter.sends}"
    assert adapter._pending_messages, "continuation prompt must still be enqueued"


@pytest.mark.asyncio
async def test_call_site_keeps_final_response_on_done(hermes_home):
    """Full-path: on a lifecycle done turn, the call site must return the
    agent's final response unchanged — the prose ships via the
    conversation, and the deferred '✅ Task completed' line lands after
    it (post-delivery)."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    # Wire the REAL _handle_message pipeline.
    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=None)
    runner._handle_message_with_agent = AsyncMock(return_value="The goal is complete. Here is the wrap-up prose.")
    runner.session_store.get_or_create_session.return_value = session_entry

    event = MessageEvent(
        text="finish the task",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-call-site",
    )

    result = await runner._handle_message(event)

    # The agent's wrap-up prose must reach the adapter unchanged.
    assert result == "The goal is complete. Here is the wrap-up prose."


@pytest.mark.asyncio
async def test_call_site_keeps_final_response_when_not_suppressed(hermes_home):
    """Full-path: when _post_turn_goal_continuation returns False, the
    call site must return the agent's final response unchanged."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=False)
    runner._handle_message_with_agent = AsyncMock(return_value="normal response text")
    runner.session_store.get_or_create_session.return_value = session_entry

    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-call-site-2",
    )

    result = await runner._handle_message(event)

    assert result == "normal response text"
