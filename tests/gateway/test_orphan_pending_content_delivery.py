"""Tests for delivering captured tool-turn content after a run completes.

Regression: with ``display.interim_assistant_messages`` disabled (e.g.
Discord), assistant text written in the same turn as tool calls is captured
on ``_last_content_with_tools`` but never transmitted. When the model's
final response is non-empty but different (e.g. "Plan presented above —
awaiting your go-ahead" while the actual plan sat in the dropped turn), the
captured content is orphaned. The gateway now delivers it as its own
message before the final response (see the pending-content block in
gateway/run.py, mirroring ``_clarify_callback_sync``).
"""

import importlib
import re
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.session import SessionSource


class OrphanCaptureAdapter(BasePlatformAdapter):
    """Records ``send`` calls; clarify methods are unused in these tests."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="text-1")

    async def send_clarify(
        self,
        chat_id,
        question,
        choices,
        clarify_id,
        session_key,
        metadata=None,
    ) -> SendResult:
        return SendResult(success=True, message_id="clarify-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class OrphanPendingContentAgent:
    """Fake AIAgent that leaves captured content behind after its run.

    Class-level knobs let each test configure the scenario before the
    gateway constructs the agent:
      - pending_content: text captured on ``_last_content_with_tools``
      - pre_delivered:   normalized text already delivered via interim rail
      - final_response:  what the fake run returns (defaults to the classic
        "referenced but did not restate" case)
    """

    session_key = "agent:main:telegram:group:-1001:17585"
    pending_content = None
    pre_delivered = None
    interrupted = False
    final_response = "Plan presented above \u2014 awaiting your go-ahead"
    instances = []

    def __init__(self, **kwargs):
        self._last_content_with_tools = None
        self._last_content_tools_all_housekeeping = False
        self._delivered_interim_texts = set()
        type(self).instances.append(self)

    @staticmethod
    def _strip_think_blocks(text):
        return re.sub(r"<thinking>.*?</thinking>", "", str(text or ""), flags=re.S)

    @staticmethod
    def _normalize_interim_visible_text(text):
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _interim_text_was_delivered(self, text):
        normalized = self._normalize_interim_visible_text(text)
        return bool(normalized) and normalized in self._delivered_interim_texts

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if type(self).pending_content is not None:
            self._last_content_with_tools = type(self).pending_content
        if type(self).pre_delivered is not None:
            self._delivered_interim_texts = {type(self).pre_delivered}
        return {
            "final_response": type(self).final_response,
            "messages": [],
            "api_calls": 1,
            "interrupted": type(self).interrupted,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )


@pytest.fixture(autouse=True)
def _reset_agent_knobs():
    OrphanPendingContentAgent.instances = []
    OrphanPendingContentAgent.pending_content = None
    OrphanPendingContentAgent.pre_delivered = None
    OrphanPendingContentAgent.interrupted = False
    OrphanPendingContentAgent.final_response = (
        "Plan presented above \u2014 awaiting your go-ahead"
    )
    yield


def _install_fakes(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = OrphanPendingContentAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    return gateway_run


@pytest.mark.asyncio
async def test_captured_content_delivered_when_final_response_differs(
    monkeypatch, tmp_path
):
    """Captured tool-turn content ships as its own message when the final
    response is non-empty but different."""
    OrphanPendingContentAgent.pending_content = "Here is the full plan."
    OrphanPendingContentAgent.final_response = (
        "Plan presented above \u2014 awaiting your go-ahead"
    )

    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="build it",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == (
        "Plan presented above \u2014 awaiting your go-ahead"
    )
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": "Here is the full plan.",
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]
    # The field is cleared so nothing can be re-sent by later fallbacks.
    assert OrphanPendingContentAgent.instances[0]._last_content_with_tools is None


@pytest.mark.asyncio
async def test_captured_content_not_resent_when_final_response_contains_it(
    monkeypatch, tmp_path
):
    """When the model echoes the captured text in its final response, the
    orphan delivery is skipped so the user does not see a duplicate."""
    OrphanPendingContentAgent.pending_content = "Here is the full plan."
    OrphanPendingContentAgent.final_response = (
        "Here is the full plan.\n\nShall I proceed?"
    )

    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="build it",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == "Here is the full plan.\n\nShall I proceed?"
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_captured_content_not_resent_when_already_delivered(
    monkeypatch, tmp_path
):
    """Content that already went through the interim rail is not duplicated."""
    OrphanPendingContentAgent.pending_content = "Already shown to the user."
    OrphanPendingContentAgent.pre_delivered = "Already shown to the user."
    OrphanPendingContentAgent.final_response = "Proceeding."

    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="build it",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == "Proceeding."
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_thinking_blocks_stripped_before_orphan_delivery(monkeypatch, tmp_path):
    """Reasoning blocks are scrubbed from captured content before delivery."""
    OrphanPendingContentAgent.pending_content = (
        "<thinking>internal reasoning</thinking>Here is the plan."
    )
    OrphanPendingContentAgent.final_response = (
        "Plan presented above \u2014 awaiting your go-ahead"
    )

    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="build it",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == (
        "Plan presented above \u2014 awaiting your go-ahead"
    )
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": "Here is the plan.",
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]


@pytest.mark.asyncio
async def test_no_captured_content_sends_nothing(monkeypatch, tmp_path):
    """A run without captured content behaves exactly as before."""
    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == (
        "Plan presented above \u2014 awaiting your go-ahead"
    )
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_captured_content_not_delivered_when_run_interrupted(
    monkeypatch, tmp_path
):
    """A run killed mid-stream (gate denial, /stop) must not flush its
    partial narration as a standalone message."""
    OrphanPendingContentAgent.pending_content = (
        "**4. Rewrite the procedures in rules/01** \u2014 per the writing skill's pattern:"
    )
    OrphanPendingContentAgent.interrupted = True
    OrphanPendingContentAgent.final_response = (
        "Operation interrupted."
    )

    _install_fakes(monkeypatch, tmp_path)
    adapter = OrphanCaptureAdapter()
    runner = _make_runner(adapter)

    result = await runner._run_agent(
        message="build it",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=OrphanPendingContentAgent.session_key,
    )

    assert result["final_response"] == "Operation interrupted."
    assert adapter.sent == []
