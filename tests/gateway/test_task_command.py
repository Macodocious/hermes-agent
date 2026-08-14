"""Tests for the gateway /task slash command."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_state import AsyncSessionDB
from tools.todo_tool import TodoStore


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, *, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry, *, platform: Platform = Platform.TELEGRAM):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session_title.return_value = None
    runner._session_db._db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _make_session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_store() -> TodoStore:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "Working thing", "status": "in_progress"},
            {"id": "3", "content": "Next thing", "status": "pending"},
        ]
    )
    store.capture_request("Captured ask")
    return store


@pytest.mark.asyncio
async def test_task_command_shows_current_task_and_full_list():
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    running_agent._todo_store = _make_store()
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/task"))

    assert "**Working on:**" not in result
    assert "- [x] 1. Done thing" in result
    assert "- [>] 2. Working thing" in result
    assert "← CURRENT TASK" not in result
    assert "- [ ] 3. Next thing" in result
    assert "**Captured requests:**" in result
    assert "- [captured] c1. Captured ask" in result
    running_agent.interrupt.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_task_command_does_not_require_running_agent(monkeypatch):
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry)
    session_key = build_session_key(_make_source())
    runner.session_store._entries = {
        session_key: SimpleNamespace(session_id="sess-1")
    }
    monkeypatch.setattr(
        "hermes_cli.tasks.load_todo",
        lambda session_id: _make_store() if session_id == "sess-1" else None,
    )

    result = await runner._handle_message(_make_event("/task"))

    assert "**Working on:**" not in result
    assert "- [>] 2. Working thing" in result
    assert "← CURRENT TASK" not in result


@pytest.mark.asyncio
async def test_task_command_empty_store_reports_empty():
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    running_agent._todo_store = TodoStore()
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/task"))

    assert result == "The task list for this session is empty."


@pytest.mark.asyncio
async def test_task_command_no_in_progress_reports_none():
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "Next thing", "status": "pending"},
        ]
    )
    running_agent._todo_store = store
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/task"))

    assert "**Working on:**" not in result
    assert "- [x] 1. Done thing" in result
    assert "- [ ] 2. Next thing" in result


@pytest.mark.asyncio
async def test_task_command_no_store_and_no_row_reports_absent(monkeypatch):
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry)
    monkeypatch.setattr("hermes_cli.tasks.load_todo", lambda session_id: None)

    result = await runner._handle_message(_make_event("/task"))

    assert "No task list for this session yet" in result
