"""Tests for the plugin → agent handoff mechanism.

A plugin slash command may return a handoff dict
``{"agent_continue": <seed>, "response": <ack>}``. The gateway shows the
ack immediately, rewrites the turn to the seed message, and falls through
to the agent so the flow continues conversationally (the /blueprint
agent_seed pattern). The unknown-command guard must not kill the
fall-through: plugin commands are not in GATEWAY_KNOWN_COMMANDS, so the
guard relies on is_gateway_known_command()'s lazy plugin lookup.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner, adapter


def _stub_plugin(monkeypatch, handler):
    """Register a fake plugin command so is_gateway_known_command() and the
    plugin dispatch branch both resolve it."""
    import gateway.run as gateway_run
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"plan": {"description": "Plan", "args_hint": ""}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: handler if name == "plan" else None,
    )


@pytest.mark.asyncio
async def test_plugin_handoff_rewrites_event_and_falls_through(monkeypatch):
    """A handoff dict shows the ack, rewrites event.text to the seed, and
    reaches the agent (the guard must not kill the fall-through)."""
    runner, adapter = _make_runner()
    _stub_plugin(
        monkeypatch,
        lambda args: {"agent_continue": "seed message", "response": "ack text"},
    )

    seen = {}

    async def _capture(event, source, _quick_key, _run_generation):
        seen["text"] = event.text
        return "agent reply"

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("/plan"))

    assert result == "agent reply"
    assert seen["text"] == "seed message"
    adapter.send.assert_awaited_once()
    ack_args = adapter.send.await_args.args
    assert ack_args[1] == "ack text"


@pytest.mark.asyncio
async def test_plugin_handoff_without_ack_skips_send(monkeypatch):
    """A handoff dict without a response still falls through; no ack is sent."""
    runner, adapter = _make_runner()
    _stub_plugin(
        monkeypatch,
        lambda args: {"agent_continue": "seed message"},
    )

    seen = {}

    async def _capture(event, source, _quick_key, _run_generation):
        seen["text"] = event.text
        return "agent reply"

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("/plan"))

    assert result == "agent reply"
    assert seen["text"] == "seed message"
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_plugin_handoff_plain_result_returns_string(monkeypatch):
    """A plain (non-handoff) plugin result is returned directly, with no
    agent turn and no ack."""
    runner, adapter = _make_runner()
    _stub_plugin(monkeypatch, lambda args: "plain result")

    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plain plugin result leaked to the agent")
    )

    result = await runner._handle_message(_make_event("/plan"))

    assert result == "plain result"
    adapter.send.assert_not_awaited()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_plugin_handoff_async_handler_awaited(monkeypatch):
    """Async plugin handlers returning handoff dicts are awaited before the
    handoff branch runs."""
    runner, adapter = _make_runner()

    async def _async_handler(args):
        return {"agent_continue": "seed from async", "response": "ack async"}

    _stub_plugin(monkeypatch, _async_handler)

    seen = {}

    async def _capture(event, source, _quick_key, _run_generation):
        seen["text"] = event.text
        return "agent reply"

    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_event("/plan"))

    assert result == "agent reply"
    assert seen["text"] == "seed from async"
    adapter.send.assert_awaited_once()
    ack_args = adapter.send.await_args.args
    assert ack_args[1] == "ack async"
