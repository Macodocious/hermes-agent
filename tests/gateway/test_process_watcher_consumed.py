"""A consumed agent completion must never be re-posted as chat text.

``notify_on_complete`` makes the completion the agent's: the watcher either
injects it as an internal turn, or — when the agent already took the result
via ``process(wait)`` / ``process(log)`` — stays silent. Before this guard the
consumed case fell through into the text-only branch and posted the raw
``[Background process … finished with exit code N~ …]`` blob into the chat,
duplicating a result the agent was already acting on.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tools.process_registry as pr_module
from gateway.config import Platform
from gateway.run import GatewayRunner


class _StubRegistry:
    """Minimal stand-in for the process registry's read surface."""

    def __init__(self, session, *, consumed: bool):
        self._session = session
        self._consumed = consumed

    def get(self, session_id):
        return self._session

    def is_completion_consumed(self, session_id):
        return self._consumed


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))


def _session():
    return SimpleNamespace(
        exited=True,
        exit_code=1,
        output_buffer="COMMITTED 327eb179bd\n",
        command="bash sync.sh",
        started_at=1000.0,
        completion_reason="exited",
        termination_source="",
    )


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._load_background_notifications_mode = lambda: "all"
    runner._deliver_completion_notification = AsyncMock(return_value=True)
    return runner


def _watcher(*, agent_notify: bool):
    return {
        "session_id": "proc_test",
        "check_interval": 0.01,
        "session_key": "agent:main:telegram:dm:42",
        "platform": "telegram",
        "chat_id": "42",
        "thread_id": "",
        "user_id": "",
        "user_name": "",
        "message_id": "",
        "notify_on_complete": agent_notify,
    }


async def _run(runner, watcher):
    await asyncio.wait_for(runner._run_process_watcher(watcher), timeout=2)


@pytest.mark.asyncio
async def test_consumed_agent_completion_is_not_posted(monkeypatch):
    """The agent already has the output — the chat must stay clean."""
    adapter = _Adapter()
    runner = _runner(adapter)
    monkeypatch.setattr(
        pr_module, "process_registry", _StubRegistry(_session(), consumed=True)
    )

    await _run(runner, _watcher(agent_notify=True))

    assert adapter.sent == [], f"consumed completion leaked into chat: {adapter.sent!r}"
    assert runner._deliver_completion_notification.await_count == 0


@pytest.mark.asyncio
async def test_unconsumed_agent_completion_still_injects_internal_turn(monkeypatch):
    """The guard must not swallow the agent's own completion delivery."""
    adapter = _Adapter()
    runner = _runner(adapter)
    monkeypatch.setattr(
        pr_module, "process_registry", _StubRegistry(_session(), consumed=False)
    )

    await _run(runner, _watcher(agent_notify=True))

    assert runner._deliver_completion_notification.await_count == 1
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_non_agent_watcher_still_posts_text_notification(monkeypatch):
    """The guard is scoped to agent_notify — plain watchers keep the text path."""
    adapter = _Adapter()
    runner = _runner(adapter)
    monkeypatch.setattr(
        pr_module, "process_registry", _StubRegistry(_session(), consumed=True)
    )

    await _run(runner, _watcher(agent_notify=False))

    assert len(adapter.sent) == 1
    assert "finished with exit code 1" in adapter.sent[0][1]
