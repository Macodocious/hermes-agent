"""Tests for delivering the agent's pre-clarify message to the chat.

Regression: when ``display.interim_assistant_messages`` is disabled (e.g.
Discord), assistant text written in the same turn as a ``clarify`` tool call
was silently dropped — the user saw only the bare "Hermes needs your input"
prompt. The gateway now delivers that pending content as its own message
before the prompt (see ``_clarify_callback_sync`` in gateway/run.py).
"""

import importlib
import re
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.session import SessionSource


class ClarifyCaptureAdapter(BasePlatformAdapter):
    """Records ``send`` (normal messages) and ``send_clarify`` (prompts)."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.clarify_prompts = []

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
        self.clarify_prompts.append(
            {
                "chat_id": chat_id,
                "question": question,
                "choices": choices,
                "clarify_id": clarify_id,
                "session_key": session_key,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="clarify-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class ClarifyPendingContentAgent:
    """Fake AIAgent that simulates a content+clarify turn.

    Class-level knobs let each test configure the scenario before the
    gateway constructs the agent:
      - pending_content: text captured on ``_last_content_with_tools``
      - pre_delivered:   normalized text already delivered via interim rail
      - instances:       every constructed instance, for post-run asserts
    """

    session_key = "agent:main:telegram:group:-1001:17585"
    pending_content = None
    pre_delivered = None
    instances = []

    def __init__(self, **kwargs):
        self.clarify_callback = kwargs.get("clarify_callback")
        self.tools = []
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

        # Fire the clarify callback on a worker thread (as the agent loop
        # does), then wait until the entry is registered so the gateway's
        # post-run cleanup can resolve it instead of leaking a blocked thread.
        if self.clarify_callback is not None:
            worker = threading.Thread(
                target=self.clarify_callback,
                args=("Build it as designed?", ["Yes", "No"]),
                daemon=True,
            )
            worker.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                from tools import clarify_gateway as cm

                if cm.get_pending_for_session(
                    type(self).session_key, include_choice_prompts=True
                ) is not None:
                    break
                time.sleep(0.02)

        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


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
def _isolate_clarify_state():
    _clear_clarify_state()
    ClarifyPendingContentAgent.instances = []
    ClarifyPendingContentAgent.pending_content = None
    ClarifyPendingContentAgent.pre_delivered = None
    yield
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_pending_content_sent_before_clarify_prompt(monkeypatch, tmp_path):
    """Assistant text from the clarify turn ships as its own message first."""
    ClarifyPendingContentAgent.pending_content = "I'll review the spec first.\n\nProceed?"

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ClarifyPendingContentAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ClarifyCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=ClarifyPendingContentAgent.session_key,
    )

    assert result["final_response"] == "done"
    assert len(adapter.clarify_prompts) == 1
    assert adapter.clarify_prompts[0]["question"] == "Build it as designed?"
    # The pending content must be delivered as a standalone message.
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": "I'll review the spec first.\n\nProceed?",
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]
    # The field is cleared so the post-clarify fallback can't re-send it.
    assert ClarifyPendingContentAgent.instances[0]._last_content_with_tools is None


@pytest.mark.asyncio
async def test_already_delivered_content_not_resent(monkeypatch, tmp_path):
    """Content that already went through the interim rail is not duplicated."""
    ClarifyPendingContentAgent.pending_content = "Already shown to the user."
    ClarifyPendingContentAgent.pre_delivered = "Already shown to the user."

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ClarifyPendingContentAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ClarifyCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=ClarifyPendingContentAgent.session_key,
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []  # no duplicate standalone message
    assert len(adapter.clarify_prompts) == 1


@pytest.mark.asyncio
async def test_no_pending_content_still_sends_prompt(monkeypatch, tmp_path):
    """A clarify without pending content behaves exactly as before."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = ClarifyPendingContentAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ClarifyCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-1",
        session_key=ClarifyPendingContentAgent.session_key,
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert len(adapter.clarify_prompts) == 1
