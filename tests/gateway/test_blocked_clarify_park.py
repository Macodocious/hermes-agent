"""Tests for Bug 2: mechanically forced clarify on a ``blocked`` goal verdict.

When the goal judge returns a ``blocked`` done verdict (the agent parked on
a human decision), the gateway must force the agent's question through the
existing clarify machinery — an open-ended parked entry whose answer is
captured by the text-intercept and re-dispatched as a real user turn. No
schema changes, no gates, no synthetic continuation, no "⚡ Stopped".

The blocked verdict tuple from ``judge_goal`` is
``(verdict, reason, parse_failed, wait_directive, transport_failed, blocked)``
— blocked is the 6th element; it flows into the decision dict as
``decision["blocked"]``.
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


@pytest.fixture(autouse=True)
def _clarify_index_isolation():
    """The clarify registry (_entries/_session_index) is module-global.
    Clear it after every test so parked entries never leak across tests."""
    from tools import clarify_gateway as clar

    yield
    with clar._lock:
        clar._entries.clear()
        clar._session_index.clear()


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
    """Minimal adapter that records send() and send_clarify() invocations.

    Deliberately NEGATIVE on ``register_post_delivery_callback`` so the
    parked-clarify delivery path takes the deterministic direct-await branch
    (the callback path is exercised implicitly by the existing status-notice
    tests) and the sends are observable before the hook returns.
    """

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []
        self.clarify_sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()

    async def send_clarify(self, chat_id, question, choices, clarify_id, session_key, metadata=None):
        self.clarify_sends.append(
            {
                "chat_id": chat_id,
                "question": question,
                "choices": choices,
                "clarify_id": clarify_id,
                "session_key": session_key,
                "metadata": metadata,
            }
        )

        class _R:
            success = True

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
async def test_blocked_verdict_parks_clarify_with_agent_question(hermes_home):
    """A blocked done verdict must park an OPEN-ENDED clarify entry whose
    prompt is the agent's own question, suppress the "✓ Goal achieved"
    status line, and enqueue NO synthetic continuation."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task per its specification: decide the api shape", lifecycle=True)

    # judge_goal returns (verdict, reason, parse_failed, wait_directive,
    # transport_failed, blocked) — blocked=True is the 6th element.
    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "needs your decision", False, None, False, True),
    ):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Which API shape should I use — REST or GraphQL?",
        )
        await asyncio.sleep(0.05)

    # No progress/completion status line is sent for the parked stop.
    assert adapter.sends == [], f"expected no status line, got {adapter.sends}"
    assert suppress is False

    # A parked open-ended clarify entry exists for the session.
    from tools import clarify_gateway as clar

    session_key = build_session_key(src)
    pending = clar.get_pending_for_session(session_key, include_choice_prompts=True)
    assert pending is not None, "a clarify entry must be parked"
    assert pending.choices is None, "parked entry must be open-ended (text-intercept path)"
    assert pending.awaiting_text is True
    assert pending.question == "Which API shape should I use — REST or GraphQL?"

    # The redirect record lets the text-intercept forward the answer.
    redirects = getattr(runner, "_parked_clarify_redirects", None) or {}
    meta = redirects.get(session_key)
    assert meta is not None, "redirect record must be present"
    assert meta["clarify_id"] == pending.clarify_id

    # The prompt is rendered on the platform after the response ships.
    assert len(adapter.clarify_sends) == 1, f"expected clarify send, got {adapter.clarify_sends}"
    assert adapter.clarify_sends[0]["question"] == "Which API shape should I use — REST or GraphQL?"
    assert adapter.clarify_sends[0]["choices"] is None

    # No synthetic continuation was enqueued — the loop is parked.
    assert adapter._pending_messages == {}, "no continuation may be enqueued"


@pytest.mark.asyncio
async def test_blocked_native_goal_also_parks(hermes_home):
    """The forced clarify must apply to NATIVE /goal blocked verdicts too —
    the marker only scopes suppression, not the mechanical clarify."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("pick a database")

    with patch(
        "hermes_cli.goals.judge_goal",
        return_value=("done", "needs your decision", False, None, False, True),
    ):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="SQLite or Postgres?",
        )
        await asyncio.sleep(0.05)

    from tools import clarify_gateway as clar

    pending = clar.get_pending_for_session(build_session_key(src), include_choice_prompts=True)
    assert pending is not None, "a native /goal blocked verdict must park too"
    assert pending.question == "SQLite or Postgres?"

    # Native non-blocked done verdicts still ship the normal status line.
    assert suppress is False


@pytest.mark.asyncio
async def test_blocked_answer_reenters_dispatch_via_intercept(hermes_home):
    """The user's answer to the parked clarify must be captured by the
    text-intercept and re-dispatched as a real user turn — NOT swallowed
    (there is no agent thread waiting on it)."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    # Park a clarify exactly as the blocked hook does.
    from tools import clarify_gateway as clar

    session_key = build_session_key(src)
    clar.clear_session(session_key)
    cid = "parked-test-01"
    clar.register(
        clarify_id=cid,
        session_key=session_key,
        question="Which API shape — REST or GraphQL?",
        choices=None,
    )
    runner._parked_clarify_redirects = {session_key: {"source": src, "clarify_id": cid}}

    # Full-path _handle_message with a no-op agent (the intercept resolves
    # the parked clarify before any agent work).
    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=False)
    runner._handle_message_with_agent = AsyncMock(return_value="normal response text")
    runner.session_store.get_or_create_session.return_value = session_entry

    event = MessageEvent(
        text="Let's go with REST.",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-answer-1",
    )

    result = await runner._handle_message(event)

    # The intercept acknowledged with "" (no duplicate post), resolved the
    # entry, and forwarded the answer into dispatch.
    assert result == ""
    assert clar.get_pending_for_session(session_key, include_choice_prompts=True) is None, (
        "parked clarify must be resolved once answered"
    )
    assert session_key not in (getattr(runner, "_parked_clarify_redirects", None) or {}), (
        "redirect record must be consumed after forwarding"
    )

    # Forwarded event sits in the FIFO as a real user turn.
    forwarded = adapter._pending_messages.get(session_key)
    assert forwarded is not None, "the answer must be enqueued for dispatch"
    assert forwarded.text == "Let's go with REST."
    assert forwarded.source is src


@pytest.mark.asyncio
async def test_blocked_answer_forwarded_event_preempts_next_turn(hermes_home):
    """A message that arrives while a forwarded answer is already queued
    must land in the overflow list — FIFO ordering preserved."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from tools import clarify_gateway as clar

    session_key = build_session_key(src)
    clar.clear_session(session_key)
    cid = "parked-test-02"
    clar.register(
        clarify_id=cid,
        session_key=session_key,
        question="Choose A or B?",
        choices=None,
    )
    runner._parked_clarify_redirects = {session_key: {"source": src, "clarify_id": cid}}

    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=False)
    runner._handle_message_with_agent = AsyncMock(return_value="normal response text")
    runner.session_store.get_or_create_session.return_value = session_entry

    # A message is already enqueued in the FIFO slot.
    adapter._pending_messages[session_key] = MessageEvent(
        text="older queued message",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-older",
        channel_prompt=None,
    )

    event = MessageEvent(
        text="A",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-answer-2",
    )
    result = await runner._handle_message(event)

    # The answer is forwarded to the FIFO OVERFLOW and the older message
    # keeps the slot — the answer is not lost and order is preserved.
    assert result == ""
    overflow = runner._queued_events.get(session_key) or []
    assert any(e.text == "A" for e in overflow), "answer must be queued in the overflow"
    assert adapter._pending_messages[session_key].text == "older queued message"
