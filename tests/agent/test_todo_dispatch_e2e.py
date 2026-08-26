"""E2E: the real todo dispatch path drives the lifecycle (P1/P2).

The design doc promised "begin write arms; close write + done verdict
marks done" through the real dispatch — this is that test, added as a
regression from PR review (the two-key close once cleared the goal on
the close write, so the judge never ran and the task stranded in
closing forever).

Exercises ``agent_runtime_helpers.invoke_tool`` for the ``todo`` tool
end to end: todo_tool -> persist_todo_store -> on_todo_write (goal
arming/clearing), then the loop's verdict observation
(``task_manager.observe_verdict``) for the second key.

External seams are pinned: SessionDB persistence and the GoalManager
are faked, so the test is deterministic and needs no real goals
provider.
"""

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import invoke_tool
from tools.todo_tool import TodoStore


def _make_agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="e2e-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store: TodoStore, item_id: str, content: str) -> None:
    store.write([{"id": item_id, "content": content, "status": "pending"}])


class FakeGoalManager:
    """Records arm/clear calls; never touches a real goals provider."""

    def __init__(self, calls: list):
        self.calls = calls

    def set(self, text: str) -> None:
        self.calls.append(f"set:{text}")

    def clear(self) -> None:
        self.calls.append("clear")


@pytest.fixture
def dispatched(monkeypatch):
    """Pin persistence + GoalManager, return (agent, calls).

    The patches resolve ``agent.task_manager`` by *string* at fixture
    time (and the verdict calls re-import the module) so they survive
    suites that delete ``agent.*`` from ``sys.modules`` mid-session
    (e.g. test_empty_tool_name_loop_dampening): the dispatch re-imports
    the module at call time, so patching the collection-time binding
    would silently miss.
    """
    calls: list = []
    monkeypatch.setattr(
        "hermes_cli.tasks.persist_todo_store", lambda agent: calls.append("persist")
    )
    monkeypatch.setattr(
        "agent.task_manager._load_goal_manager",
        lambda agent: FakeGoalManager(calls),
    )
    monkeypatch.setattr("agent.task_manager._persist", lambda agent: None)
    # Pin the lifecycle gate: the E2E must be independent of the host
    # config (a config-touching test earlier in the suite can change what
    # load_config returns and silently disable the gate).
    monkeypatch.setattr(
        "agent.task_manager._lifecycle_config", lambda: {"enabled": True}
    )
    # The post-close review must not fire real LLM calls in E2E tests.
    monkeypatch.setattr(
        "agent.task_manager._review_config", lambda: {"enabled": False}
    )
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    return _make_agent(store), calls


def _task_manager():
    """Re-import at call time so verdict calls hit the live module.

    The same sys.modules wipe that can orphan module-object patches also
    orphans a collection-time ``from agent import task_manager`` binding;
    re-importing here resolves whatever the dispatch would resolve.
    """
    import importlib

    return importlib.import_module("agent.task_manager")


def _invoke(agent, action: str, item_id: str) -> None:
    invoke_tool(
        agent,
        "todo",
        {"action": action, "item_id": item_id},
        effective_task_id="",
        pre_tool_block_checked=True,
        skip_tool_request_middleware=True,
    )


def test_begin_write_arms_goal(dispatched) -> None:
    agent, calls = dispatched
    _invoke(agent, "begin", "1")

    assert "set:Complete the task: Build the thing" in calls
    assert agent._todo_store.read()[0]["status"] == "in_progress"
    assert agent._task_lifecycle_action_issued is True


def test_close_write_keeps_goal_armed_and_judge_done_finalizes(dispatched) -> None:
    """The doc's E2E promise: close write + done verdict marks done.

    Regression: the close write used to clear the goal, so the loop
    stopped before the judge could run and the task stranded in closing.
    """
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "close", "1")

    # The loop must still be armed after the close write (second key).
    assert "clear" not in calls
    assert agent._todo_store.read()[0]["status"] == "closing"

    nudge = _task_manager().observe_verdict(agent, {"verdict": "done"})
    assert nudge is None
    assert agent._todo_store.read()[0]["status"] == "completed"


def test_close_write_with_continue_verdict_returns_to_work(dispatched) -> None:
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "close", "1")

    nudge = _task_manager().observe_verdict(agent, {"verdict": "continue"})
    assert nudge is None
    assert agent._todo_store.read()[0]["status"] == "in_progress"


def test_pause_write_clears_goal(dispatched) -> None:
    agent, calls = dispatched
    _invoke(agent, "begin", "1")
    _invoke(agent, "pause", "1")

    assert "clear" in calls
    assert agent._todo_store.read()[0]["status"] == "paused"
