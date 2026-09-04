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
    """A lifecycle goal's done verdict on a REAL user turn must NOT
    suppress the turn's final response — the agent's prose ships via the
    conversation, and the deterministic '✅ Task completed' line lands
    after it (deferred to post-delivery)."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    assert suppress is False
    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    assert "✅ Task completed: ship the feature" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_lifecycle_goal_done_on_continuation_turn_suppresses(hermes_home):
    """A lifecycle goal's done verdict on a SYNTHETIC goal-continuation
    turn (user_initiated=False) must return the suppress flag when the
    judge already said done on the PRECEDING real user turn
    (last_verdict == 'done' — the rejection-gate refusal path): the
    wrap-up prose on that turn is a SECOND conclusion (the conversation
    prose already shipped on the real user turn), so the call site
    blanks it and the deferred '✅ Task completed' line is the only
    completion message."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")
    # The judge said done on the preceding real user turn (rejection
    # gate refused finalization — resume() never touches last_verdict).
    # Persist so the hook's fresh GoalManager sees it.
    mgr.state.last_verdict = "done"
    from hermes_cli.goals import save_goal

    save_goal(session_entry.session_id, mgr.state)

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="The goal is complete — stating so explicitly and stopping.",
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    assert suppress is True
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
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    assert suppress is False
    assert len(adapter.sends) == 1
    assert "Goal achieved" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_lifecycle_goal_continue_returns_no_suppress_flag(hermes_home):
    """A lifecycle goal still in progress must NOT suppress the turn's
    final response — only the done verdict on a continuation turn does."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still working", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    assert suppress is False
    assert adapter.sends == [], f"expected no progress line, got {adapter.sends}"
    assert adapter._pending_messages, "continuation prompt must still be enqueued"


@pytest.mark.asyncio
async def test_call_site_keeps_final_response_on_done(hermes_home):
    """Full-path: on a lifecycle done turn that is NOT suppressed (real
    user turn — hook returns False), the call site must return the
    agent's final response unchanged — the prose ships via the
    conversation, and the deferred '✅ Task completed' line lands after
    it (post-delivery)."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    # Wire the REAL _handle_message pipeline.
    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=False)
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
async def test_call_site_blanks_final_response_when_suppressed(hermes_home):
    """Full-path: when _post_turn_goal_continuation returns True (done
    verdict on a synthetic goal-continuation turn), the call site must
    blank the final response — the redundant wrap-up prose never ships,
    and the deferred '✅ Task completed' line is the only completion
    message."""
    from gateway.run import GatewayRunner

    runner, adapter, session_entry, src = _make_runner_with_adapter()

    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._check_slash_access = lambda *a, **k: None
    runner._post_turn_goal_continuation = AsyncMock(return_value=True)
    runner._handle_message_with_agent = AsyncMock(return_value="The goal is complete. Here is the wrap-up prose.")
    runner.session_store.get_or_create_session.return_value = session_entry

    event = MessageEvent(
        text="[Continuing toward your standing goal]\nGoal: Complete the task: finish the task",
        message_type=MessageType.TEXT,
        source=src,
        message_id="m-call-site-suppressed",
    )

    result = await runner._handle_message(event)

    # The redundant wrap-up prose must be blanked.
    assert result is None


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


# ──────────────────────────────────────────────────────────────────────
# Part 1 — prose discriminator (last_verdict) + Part 2 — finalization hold
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejection_gate_path_continuation_done_suppresses(hermes_home):
    """Rejection-gate path (Sep 1): the judge said DONE on the real user
    turn and the user rejected finalization — the hold is armed and
    last_verdict stays 'done' (resume never touches it). The synthetic
    continuation turn's done verdict is a SECOND done: its wrap-up prose
    must be suppressed (suppress flag True) and the task must stay
    active (no ✅ line) until the user's final finalization."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    # Turn 1 — real user turn: judge says done, user rejects finalization.
    with (
        patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)),
        patch("gateway.run._is_lifecycle_rejection_message", return_value=True),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
            user_message="Wrong again.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # The rejection gate armed the hold and kept the task active; the
    # judge DID say done — resume() never touches last_verdict. Re-load
    # from the DB — the hook ran on a fresh GoalManager.
    from hermes_cli.goals import load_goal

    state = load_goal(session_entry.session_id)
    assert state is not None
    assert state.awaiting_finalization is True
    assert state.status == "active"
    assert state.last_verdict == "done"

    # Turn 2 — synthetic continuation turn: judge says done again.
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="The goal is complete — stating so explicitly and stopping.",
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    # SECOND done → suppressed; task held ongoing (no ✅ line). Re-load
    # from the DB — the hook ran on a fresh GoalManager.
    from hermes_cli.goals import load_goal

    state = load_goal(session_entry.session_id)
    assert suppress is True
    assert state is not None
    assert state.status == "active", "task must stay ongoing until final finalization"
    assert adapter.sends == [], f"expected no completion line, got {adapter.sends}"
    assert adapter._pending_messages, "continuation prompt must still be enqueued"


@pytest.mark.asyncio
async def test_working_continuation_path_first_done_ships(hermes_home):
    """Working-continuation path (Sep 3): the judge said CONTINUE on the
    real user turn (last_verdict == 'continue'), so the conversation
    prose did NOT deliver the conclusion. The synthetic continuation
    turn's done verdict is the FIRST done: its prose must ship (suppress
    flag False) and the deferred ✅ line lands after it."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: verify the refactor")
    # The preceding real user turn's judge said continue. Persist so the
    # hook's fresh GoalManager sees it.
    mgr.state.last_verdict = "continue"
    from hermes_cli.goals import save_goal

    save_goal(session_entry.session_id, mgr.state)

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the refactor is verified", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Verification complete: all imports resolve and the test suite passes.",
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    # FIRST done → prose ships; ✅ deferred post-delivery.
    assert suppress is False
    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    assert "✅ Task completed: verify the refactor" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_finalization_hold_keeps_task_ongoing(hermes_home):
    """Part 2 hold behavior: with the finalization hold armed, a
    synthetic continuation turn's done verdict must be HELD — treated as
    a continue, task stays active, no ✅ line, wrap-up prose suppressed —
    until the user's next real message finalizes."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")
    # Simulate a prior rejected finalization. Persist so the hook's fresh
    # GoalManager sees it.
    mgr.hold_finalization()
    mgr.state.last_verdict = "done"
    from hermes_cli.goals import save_goal

    save_goal(session_entry.session_id, mgr.state)

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="The goal is complete — stating so explicitly and stopping.",
            user_initiated=False,
        )
        await asyncio.sleep(0.05)

    # Held: suppressed, task still active, no completion line. Re-load
    # from the DB — the hook ran on a fresh GoalManager.
    from hermes_cli.goals import load_goal

    state = load_goal(session_entry.session_id)
    assert suppress is True
    assert state is not None
    assert state.status == "active", "held done verdict must keep the task active"
    assert state.awaiting_finalization is True, "hold must stay armed"
    assert adapter.sends == [], f"expected no completion line, got {adapter.sends}"
    assert adapter._pending_messages, "continuation prompt must still be enqueued"


@pytest.mark.asyncio
async def test_final_finalization_releases_hold_and_finalizes(hermes_home):
    """Part 2 final-finalization behavior: a real user turn whose done
    verdict STANDS (not a rejection) is the user's final finalization —
    the hold is released, the task finalizes, and the ✅ line ships."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")
    # Simulate a prior rejected finalization. Persist so the hook's fresh
    # GoalManager sees it.
    mgr.hold_finalization()
    mgr.state.last_verdict = "done"
    from hermes_cli.goals import save_goal

    save_goal(session_entry.session_id, mgr.state)

    with (
        patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)),
        patch("gateway.run._is_lifecycle_rejection_message", return_value=False),
    ):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Done — confirmed working.",
            user_message="Looks good, ship it.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # Final finalization: hold released, task done, ✅ shipped. Re-load
    # from the DB — the hook ran on a fresh GoalManager.
    from hermes_cli.goals import load_goal

    state = load_goal(session_entry.session_id)
    assert suppress is False
    assert state is not None
    assert state.status == "done", "task must finalize on the user's final finalization"
    assert state.awaiting_finalization is False, "hold must be released"
    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    assert "✅ Task completed: ship the feature" in adapter.sends[0]["content"]
