"""Tests for the plugin → interactive picker mechanism.

A plugin slash command may return a picker dict
``{"picker": {"title", "choices", "on_selected"}, "response": <ack>}``. The
gateway renders the platform's native select menu via the adapter's
``send_choice_picker``, and the selection resumes the session as the next turn
so the flow continues conversationally.

This mirrors the gateway's own ``/model`` and ``/reasoning`` pickers: the
capability is detected on the adapter *type*, and a platform without it (or a
failed send) falls back to the command's text response rather than posting a
raw result dict.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.plugins import is_plugin_picker_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
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


class _PickerAdapter:
    """Adapter whose *type* exposes ``send_choice_picker`` (the gate the
    dispatch checks via ``getattr(type(adapter), 'send_choice_picker', None)``)."""

    def __init__(self, success: bool = True):
        self.calls = []
        self._success = success

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=self._success, message_id="m1")


class _NoPickerAdapter:
    """Adapter with no choice-picker capability."""


def _make_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
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
    runner._adapter_for_source = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._session_key_for_source = lambda source: "sess-key"
    return runner


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


def _picker_result(on_selected=None, response="Choosing a repository..."):
    return {
        "picker": {
            "title": "Which repository?",
            "choices": [
                {"value": "custom", "label": "Non-GitHub project"},
                {"value": "hermes-agent", "label": "hermes-agent"},
            ],
            "on_selected": on_selected or (lambda chat_id, value: ""),
        },
        "response": response,
    }


class TestPluginPickerDispatch:
    @pytest.mark.asyncio
    async def test_picker_result_renders_menu(self, monkeypatch):
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        _stub_plugin(monkeypatch, lambda args: _picker_result())

        result = await runner._handle_message(_make_event("/plan"))

        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["title"] == "Which repository?"
        assert [c["value"] for c in call["choices"]] == [
            "custom",
            "hermes-agent",
        ]
        assert call["chat_id"] == "c1"
        assert callable(call["on_choice_selected"])
        assert result == "Choosing a repository..."

    @pytest.mark.asyncio
    async def test_picker_result_falls_back_without_capability(self, monkeypatch):
        """A platform with no picker returns the text response, never a dict."""
        runner = _make_runner(_NoPickerAdapter())
        _stub_plugin(monkeypatch, lambda args: _picker_result())

        result = await runner._handle_message(_make_event("/plan"))

        assert result == "Choosing a repository..."
        assert not isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_picker_result_falls_back_when_send_fails(self, monkeypatch):
        adapter = _PickerAdapter(success=False)
        runner = _make_runner(adapter)
        _stub_plugin(monkeypatch, lambda args: _picker_result())

        result = await runner._handle_message(_make_event("/plan"))

        assert len(adapter.calls) == 1  # attempted, then fell back
        assert result == "Choosing a repository..."

    @pytest.mark.asyncio
    async def test_selection_resumes_session_with_selected_text(self, monkeypatch):
        """Selecting an option invokes the plugin callback and spawns a turn."""
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        seen = {}

        def _on_selected(chat_id, value):
            seen["callback"] = (chat_id, value)
            return f"Repository set to {value}."

        spawned = []
        runner._spawn_plugin_picker_turn = (
            lambda source, text: spawned.append((source, text))
        )
        _stub_plugin(monkeypatch, lambda args: _picker_result(on_selected=_on_selected))

        await runner._handle_message(_make_event("/plan"))

        callback = adapter.calls[0]["on_choice_selected"]
        result_text = await callback("c1", "hermes-agent")

        assert seen["callback"] == ("c1", "hermes-agent")
        assert result_text == "Repository set to hermes-agent."
        assert len(spawned) == 1
        assert spawned[0][1] == "Repository set to hermes-agent."

    @pytest.mark.asyncio
    async def test_selection_with_empty_text_does_not_spawn(self, monkeypatch):
        """A callback returning nothing still answers the picker, but injects
        no turn (the plugin declined the selection)."""
        adapter = _PickerAdapter()
        runner = _make_runner(adapter)
        spawned = []
        runner._spawn_plugin_picker_turn = (
            lambda source, text: spawned.append((source, text))
        )
        _stub_plugin(monkeypatch, lambda args: _picker_result(
            on_selected=lambda chat_id, value: ""
        ))

        await runner._handle_message(_make_event("/plan"))
        callback = adapter.calls[0]["on_choice_selected"]
        result_text = await callback("c1", "hermes-agent")

        assert result_text == "Selected."
        assert spawned == []


class TestIsPluginPickerResult:
    def test_accepts_well_formed_result(self):
        assert is_plugin_picker_result(_picker_result()) is True

    @pytest.mark.parametrize(
        "result",
        [
            None,
            "plain string",
            {"agent_continue": "seed"},  # handoff, not picker
            {"picker": None},
            {"picker": {}},  # no title/choices/on_selected
            {"picker": {"title": "T", "choices": []}},  # no callback
            {"picker": {"title": "T", "on_selected": lambda c, v: ""}},  # no choices
            {"picker": {"title": "", "choices": [], "on_selected": lambda c, v: ""}},
            {"picker": {"title": "T", "choices": "not-a-list", "on_selected": lambda c, v: ""}},
            {"picker": {"title": "T", "choices": [], "on_selected": "not-callable"}},
        ],
    )
    def test_rejects_malformed_results(self, result):
        assert is_plugin_picker_result(result) is False

    def test_handoff_result_is_not_a_picker(self):
        """The two shapes stay distinguishable so dispatch order is safe."""
        assert is_plugin_picker_result({"agent_continue": "x", "response": "y"}) is False
