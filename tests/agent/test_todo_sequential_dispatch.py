"""E2E: the SEQUENTIAL todo dispatch path drives the lifecycle.

Regression for the dispatch asymmetry found in the post-merge health
check: ``agent_runtime_helpers.invoke_tool`` (concurrent path) forwarded
``action``/``item_id`` into ``todo_tool`` and ran the persist + lifecycle
hooks, but ``tool_executor.execute_tool_calls_sequential`` (single-call
path — the majority of turns) dropped both. Every single-call
``action=begin``/``complete`` silently degraded to a plain read: the store
never transitioned and nothing persisted.

These tests drive the REAL sequential dispatcher
(``agent.tool_executor.execute_tool_calls_sequential``) with a real
``AIAgent`` and pin two contracts:

    1. ``action``/``item_id`` reach ``todo_tool`` (store transitions).
    2. Lifecycle actions persist the store (write-through).

External seams are pinned by string-path monkeypatch (suite-survivable,
same pattern as ``test_todo_dispatch_e2e``): SessionDB persistence, the
lifecycle config gate, and tool-result persistence.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tools.todo_tool import TodoStore


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "todo tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _make_agent():
    """Real AIAgent driven through the real sequential dispatch surface."""
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=TOOL_DEFINITIONS,
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _mock_tool_call(arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="todo", arguments=arguments),
    )


@pytest.fixture
def dispatched(monkeypatch, tmp_path):
    """Pin the external seams; return (agent, seams).

    String-path patches resolve at fixture time and the re-imports inside
    the dispatch resolve the same module objects, matching the hardening
    established for ``test_todo_dispatch_e2e`` against suites that wipe
    ``sys.modules`` mid-run.
    """
    seams: dict = {"persist": 0}
    monkeypatch.setattr(
        "hermes_cli.tasks.persist_todo_store",
        lambda agent: seams.__setitem__("persist", seams["persist"] + 1),
    )
    monkeypatch.setattr("agent.task_manager._persist", lambda agent: None)
    monkeypatch.setattr(
        "agent.task_manager._lifecycle_config", lambda: {"enabled": True}
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "agent.tool_executor.maybe_persist_tool_result",
        lambda **kwargs: kwargs["content"],
    )
    agent = _make_agent()
    agent._todo_store.write(
        [{"id": "1", "content": "Build the thing", "status": "pending"}]
    )
    return agent, seams


def _dispatch_sequential(agent, arguments: str) -> list:
    """Run one todo tool call through the real sequential dispatcher."""
    from agent.tool_executor import execute_tool_calls_sequential

    assistant_message = SimpleNamespace(
        content="", tool_calls=[_mock_tool_call(arguments)]
    )
    messages: list = []
    execute_tool_calls_sequential(agent, assistant_message, messages, "")
    assert len(messages) == 1, "tool result was never appended"
    return messages


def test_sequential_begin_transitions_store(dispatched) -> None:
    """action=begin must transition the store through the sequential path.

    Regression: the sequential path dropped ``action``/``item_id`` — the
    tool saw a plain read and the item stayed pending.
    """
    agent, seams = dispatched
    _dispatch_sequential(agent, json.dumps({"action": "begin", "item_id": "1"}))

    assert agent._todo_store.read()[0]["status"] == "in_progress"
    assert agent._task_lifecycle_action_issued is True


def test_sequential_begin_persists_store(dispatched) -> None:
    """An action-only call is a mutation and must persist write-through.

    Regression: the persist condition only checked ``todos``/
    ``dispositions``, so action-only transitions were never persisted and
    the state_meta row went stale.
    """
    agent, seams = dispatched
    _dispatch_sequential(agent, json.dumps({"action": "begin", "item_id": "1"}))

    assert seams["persist"] == 1


def test_sequential_complete_transitions_to_completed(dispatched) -> None:
    """action=complete must reach the store: in_progress -> completed."""
    agent, seams = dispatched
    _dispatch_sequential(agent, json.dumps({"action": "begin", "item_id": "1"}))
    _dispatch_sequential(agent, json.dumps({"action": "complete", "item_id": "1"}))

    assert agent._todo_store.read()[0]["status"] == "completed"


def test_sequential_write_mode_still_persists_and_hooks(dispatched) -> None:
    """The legacy write path keeps its existing persist + hook behavior."""
    agent, seams = dispatched
    _dispatch_sequential(
        agent,
        json.dumps(
            {"todos": [{"id": "2", "content": "Second task", "status": "pending"}],
             "merge": True}
        ),
    )

    assert len(agent._todo_store.read()) == 2
    assert seams["persist"] == 1


def test_sequential_plain_read_skips_persist_and_hooks(dispatched) -> None:
    """A bare read (no todos/dispositions/action) stays side-effect free."""
    agent, seams = dispatched
    _dispatch_sequential(agent, json.dumps({}))

    assert seams["persist"] == 0
    assert agent._todo_store.read()[0]["status"] == "pending"
