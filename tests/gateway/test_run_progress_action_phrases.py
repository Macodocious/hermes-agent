"""Gateway end-to-end test: action-aware tool progress bubbles.

Drives the real gateway ``progress_callback`` path (via ``_run_agent`` with a
fake agent, mirroring ``test_run_progress_task_start.py``) and asserts that
action-driven tools render complete standalone phrases — "⏰ Reading scheduled
jobs" for ``cronjob action="list"`` — instead of the misleading fixed-verb
form ("⏰ Scheduling list").
"""

import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

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
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": {"stopped": True}})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class SingleToolAgent:
    """Fake agent that emits exactly one tool.started event for the given call."""

    def __init__(self, tool_name, preview, tool_args, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []
        self.tool_name = tool_name
        self.preview = preview
        self.tool_args = tool_args

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", self.tool_name, self.preview, self.tool_args)
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
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


def _install_fake_agent(monkeypatch, fake_agent):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = fake_agent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    # Register tool emojis used by the bubble renderer.
    import tools.cronjob_tools  # noqa: F401 - register cronjob emoji
    import tools.memory_tool  # noqa: F401 - register memory emoji
    import tools.skill_manager_tool  # noqa: F401 - register skill_manage emoji


async def _run_bubble(monkeypatch, tmp_path, fake_agent_cls):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    _install_fake_agent(monkeypatch, fake_agent_cls)

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="run",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-action-phrases",
        session_key="agent:main:telegram:group:-1001:17585",
    )
    assert result["final_response"] == "done"
    return " ".join(call["content"] for call in adapter.sent)


@pytest.mark.asyncio
async def test_cronjob_list_bubble_reads_jobs(monkeypatch, tmp_path):
    class Agent(SingleToolAgent):
        def __init__(self, **kwargs):
            super().__init__("cronjob", "list", {"action": "list"}, **kwargs)

    rendered = await _run_bubble(monkeypatch, tmp_path, Agent)
    assert "⏰ Reading scheduled jobs" in rendered, f"missing cronjob list bubble in {rendered}"
    assert "Scheduling list" not in rendered


@pytest.mark.asyncio
async def test_cronjob_create_bubble_creates_job(monkeypatch, tmp_path):
    class Agent(SingleToolAgent):
        def __init__(self, **kwargs):
            super().__init__(
                "cronjob", "create",
                {"action": "create", "name": "daily-brief"},
                **kwargs,
            )

    rendered = await _run_bubble(monkeypatch, tmp_path, Agent)
    assert "⏰ Creating a scheduled job" in rendered, f"missing cronjob create bubble in {rendered}"
    assert "Scheduling create" not in rendered


@pytest.mark.asyncio
async def test_skill_manage_delete_bubble_deletes_skill(monkeypatch, tmp_path):
    class Agent(SingleToolAgent):
        def __init__(self, **kwargs):
            super().__init__(
                "skill_manage", "delete",
                {"action": "delete", "name": "my-skill"},
                **kwargs,
            )

    rendered = await _run_bubble(monkeypatch, tmp_path, Agent)
    assert "📝 Deleting skill my-skill" in rendered, f"missing skill_manage bubble in {rendered}"
    assert "Updating skill" not in rendered


@pytest.mark.asyncio
async def test_memory_add_bubble_saves_to_memory(monkeypatch, tmp_path):
    class Agent(SingleToolAgent):
        def __init__(self, **kwargs):
            super().__init__(
                "memory", "add",
                {"action": "add", "target": "user"},
                **kwargs,
            )

    rendered = await _run_bubble(monkeypatch, tmp_path, Agent)
    assert "🧠 Saving to memory" in rendered, f"missing memory bubble in {rendered}"
    assert "Updating memory" not in rendered
