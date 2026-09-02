"""Tests for gateway /goal verdict-message delivery.

The judge verdict message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
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
    # Default to a unique session_id so xdist parallel runs on the same worker
    # don't see each other's GoalManager state (DEFAULT_DB_PATH gets frozen at
    # module-import time, defeating per-test HERMES_HOME monkeypatches).
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
async def test_goal_verdict_done_sent_via_adapter_send(hermes_home):
    """When the judge says done, the '✓ Goal achieved' message must reach
    the user through the adapter's ``send()`` method."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("ship the feature")

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="I shipped the feature.",
        )
        # fire-and-forget create_task — give the loop a tick
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1, f"expected 1 send, got {len(adapter.sends)}: {adapter.sends}"
    msg = adapter.sends[0]
    assert msg["chat_id"] == "c1"
    assert "Goal achieved" in msg["content"]
    assert "the feature shipped" in msg["content"]


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the judge says continue, both the 'continuing' status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
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

    # Status line sent back
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    # Continuation prompt enqueued for next turn
    assert adapter._pending_messages, "continuation prompt must be enqueued in pending_messages"


@pytest.mark.asyncio
async def test_lifecycle_rejection_gate_refuses_done(hermes_home):
    """A task-lifecycle goal whose judge says DONE must be refused when the
    triggering user message rejects the answer: the done flip is undone
    (resume keeps the turn budget), the decision is forced to continue, and
    no 'Goal achieved' status line is delivered."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: answer the question")

    with (
        patch.object(
            GoalManager,
            "evaluate_after_turn",
            return_value={
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": "the question was answered",
                "message": "✓ Goal achieved: the question was answered",
            },
        ),
        patch("gateway.run._is_lifecycle_rejection_message", return_value=True),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Here is the answer.",
            user_message="Wrong again.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # Done flip undone — the goal is active again, not done.
    assert mgr.state is not None
    assert mgr.state.status == "active"
    # No "Goal achieved" status line delivered.
    assert adapter.sends == [], f"expected no sends, got {adapter.sends}"
    # The loop continues — a continuation prompt is enqueued.
    assert adapter._pending_messages, "continuation prompt must be enqueued"


@pytest.mark.asyncio
async def test_lifecycle_wait_bypass_on_user_message(hermes_home):
    """A real user message must release a parked task-lifecycle goal so the
    judge evaluates the fresh exchange instead of staying parked."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: polish the docs")
    mgr.wait_for_seconds(seconds=60, reason="backoff")

    with patch.object(
        GoalManager,
        "evaluate_after_turn",
        return_value={
            "status": "active",
            "should_continue": True,
            "continuation_prompt": "keep going",
            "verdict": "continue",
            "reason": "noop",
            "message": "",
        },
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="docs updated",
            user_message="keep going",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # The wait barrier was cleared by the bypass.
    assert mgr.state is not None
    assert mgr.state.waiting_on_pid is None


@pytest.mark.asyncio
async def test_native_goal_untouched_by_rejection(hermes_home):
    """A native /goal (no lifecycle prefix) must keep base behavior: a done
    verdict stands even when the user message reads as a rejection — the
    lifecycle rejection gate is scoped to task-lifecycle goals only."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("answer the question")

    with (
        patch("hermes_cli.goals.judge_goal", return_value=("done", "answered", False, None, False, False)),
        patch("gateway.run._is_lifecycle_rejection_message", return_value=True),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Here is the answer.",
            user_message="Wrong again.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # Done verdict stands — "Goal achieved" delivered.
    assert len(adapter.sends) == 1
    assert "Goal achieved" in adapter.sends[0]["content"]


@pytest.mark.asyncio
async def test_goal_verdict_budget_exhausted_sends_pause(hermes_home):
    """When the budget is exhausted, a '⏸ Goal paused' message must be sent
    and no further continuation enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager, save_goal

    mgr = GoalManager(session_entry.session_id, default_max_turns=2)
    state = mgr.set("tiny goal", max_turns=2)
    state.turns_used = 2
    save_goal(session_entry.session_id, state)

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="still partial",
        )
        await asyncio.sleep(0.05)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages


@pytest.mark.asyncio
async def test_goal_verdict_skipped_when_no_active_goal(hermes_home):
    """No goal set → the hook is a no-op. Nothing is sent, nothing enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    await runner._post_turn_goal_continuation(
        session_entry=session_entry,
        source=src,
        final_response="anything",
    )
    await asyncio.sleep(0.05)

    assert adapter.sends == []
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_goal_verdict_survives_adapter_without_send(hermes_home):
    """Bad adapter (no ``send`` attribute) must not crash the judge hook."""
    runner, _adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("survive missing send")

    class _NoSendAdapter:
        def __init__(self):
            self._pending_messages: dict = {}

    runner.adapters[Platform.TELEGRAM] = _NoSendAdapter()

    with patch("hermes_cli.goals.judge_goal", return_value=("done", "ok", False, None, False, False)):
        # must not raise
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="whatever",
        )
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_blocked_done_skips_rejection_gate(hermes_home):
    """A blocked-awaiting-input done verdict must skip the lifecycle
    rejection gate entirely: no auxiliary call, no forced continue, no
    'Goal achieved' status line — the agent's question ships as the final
    response and the loop stops for the user."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: resolve the merge")

    with (
        patch(
            "hermes_cli.goals.judge_goal",
            return_value=("done", "Agent is blocked awaiting user direction", False, None, False, True),
        ),
        patch("gateway.run._is_lifecycle_rejection_message", return_value=True) as rejection_gate,
    ):
        suppress = await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Blocked on your direction: option 1 or option 2?",
            user_message="Which option do you want?",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # The rejection gate must never have been consulted.
    rejection_gate.assert_not_called()
    # No continuation enqueued — the loop stops for the user.
    assert not adapter._pending_messages
    # No "Goal achieved" status line for a blocked stop.
    assert adapter.sends == []
    # The agent's question is NOT suppressed (it ships as the final response).
    assert suppress is None


@pytest.mark.asyncio
async def test_blocked_done_does_not_finalize_task(hermes_home):
    """A blocked-awaiting-input done verdict must not finalize a closing
    task or emit the completion line — the task stays in_progress."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager
    from tools.todo_tool import TodoStore

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: resolve the merge")

    # In-memory store: SessionDB's DEFAULT_DB_PATH is frozen at import time,
    # so the real load_todo/save_todo would touch the host state.db. Patch
    # them to the in-memory store to keep the test hermetic.
    store = TodoStore()
    # A fresh store assigns sequential ids on write (merge keeps ids stable
    # only for items already present) — capture the assigned id for the
    # transitions below.
    store.write([{"id": "t-1", "content": "resolve the merge", "status": "pending"}])
    item_id = store.read()[0]["id"]
    store.transition("begin", item_id)
    store.transition("close", item_id)  # closing — awaiting the judge's second key
    saved: list[TodoStore] = []

    with (
        patch(
            "hermes_cli.goals.judge_goal",
            return_value=("done", "blocked awaiting user direction", False, None, False, True),
        ),
        patch("hermes_cli.tasks.load_todo", return_value=store),
        patch("hermes_cli.tasks.save_todo", side_effect=lambda _sid, s: saved.append(s)),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Blocked on your direction.",
            user_message="Which option?",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # The closing task was NOT finalized by the blocked verdict.
    items = saved[0].read()
    assert any(i["id"] == item_id and i["status"] == "closing" for i in items)
    assert not any(i["id"] == item_id and i["status"] == "completed" for i in items)
    # No completion line delivered.
    assert adapter.sends == []


@pytest.mark.asyncio
async def test_rejection_gate_fails_open_on_auxiliary_error(hermes_home):
    """When the auxiliary rejection check errors, the judge's done verdict
    must stand (fail-open) — no forced continue, no synthetic continuation."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("Complete the task: ship the feature")

    with (
        patch("hermes_cli.goals.judge_goal", return_value=("done", "the feature shipped", False, None, False, False)),
        # The real _is_lifecycle_rejection_message catches provider errors
        # and returns False — the gate only refuses on positive evidence.
        patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError('"rejected"')),
    ):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="Shipped.",
            user_message="That's wrong.",
            user_initiated=True,
        )
        await asyncio.sleep(0.05)

    # Done verdict stands — the lifecycle completion line is delivered,
    # no continuation.
    assert len(adapter.sends) == 1
    assert "Task completed" in adapter.sends[0]["content"]
    assert not adapter._pending_messages
