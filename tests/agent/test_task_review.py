"""Tests for the post-close verification probe (P6) in task_manager.

Covers the mechanical side of the mandatory second set of eyes: the
unconditional probe write on finalization (intent check always present,
import checks derived from the session's changed files, capped), the
module derivation from the git diff (tests/ files excluded), the
fail-open write path, and the no-finalization no-op.

The probe write itself (``_write_probe``) is exercised against a
temporary probes/active/ directory — no real gateway, no LLM calls.
"""

from types import SimpleNamespace

import pytest
import yaml

from agent import task_manager
from tools.todo_tool import TodoStore


def _make_agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="probe-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store: TodoStore, item_id: str, content: str, **extra) -> None:
    item = {"id": item_id, "content": content, "status": "pending"}
    item.update(extra)
    store.write([item])


@pytest.fixture(autouse=True)
def _probe_home(monkeypatch, tmp_path) -> None:
    """Pin the probes home to a temp dir so tests never touch the host."""
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path
    )


def _close_in_flight(store: TodoStore, item_id: str) -> None:
    """Drive a task through begin + close, leaving it in closing.

    The probe fires on the verdict observation that finalizes the task
    (the E2E pattern): begin → close → observe_verdict(done).
    """
    store.transition("begin", item_id)
    store.transition("close", item_id)


def _active_probes(tmp_path) -> list:
    return sorted((tmp_path / "probes" / "active").glob("*.yaml"))


# ── the probe hook (close → probe write) ───────────────────────────────


def test_close_writes_probe_with_intent(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None  # deferred verification — no continuation nudge
    probes = _active_probes(tmp_path)
    assert len(probes) == 1
    probe = yaml.safe_load(probes[0].read_text(encoding="utf-8"))
    assert probe["target"] == "task:1"
    assert probe["activation"] == "gateway_restart"
    assert probe["status"] == "pending"
    assert probe["created_at"]
    types = [check["type"] for check in probe["checks"]]
    assert "intent" in types  # intent is the mandatory default check
    intent = next(c for c in probe["checks"] if c["type"] == "intent")
    assert "Build the thing" in intent["prompt"]


def test_close_writes_probe_with_import_checks(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    monkeypatch.setattr(
        task_manager,
        "_git_diff",
        lambda session_id: (
            "diff --git a/agent/task_manager.py b/agent/task_manager.py\n"
            "diff --git a/tests/agent/test_task_review.py b/tests/agent/test_task_review.py\n"
        ),
    )

    _close_in_flight(store, "1")
    task_manager.observe_verdict(agent, {"verdict": "done"})

    probe = yaml.safe_load(_active_probes(tmp_path)[0].read_text(encoding="utf-8"))
    modules = [
        check["module"]
        for check in probe["checks"]
        if check["type"] == "import"
    ]
    assert modules == ["agent.task_manager"]  # tests/ files never probed


def test_import_checks_capped(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    diff_lines = "\n".join(
        f"diff --git a/mod{i}.py b/mod{i}.py" for i in range(10)
    )
    monkeypatch.setattr(task_manager, "_git_diff", lambda session_id: diff_lines)

    _close_in_flight(store, "1")
    task_manager.observe_verdict(agent, {"verdict": "done"})

    probe = yaml.safe_load(_active_probes(tmp_path)[0].read_text(encoding="utf-8"))
    imports = [c for c in probe["checks"] if c["type"] == "import"]
    assert len(imports) == task_manager.PROBE_MAX_IMPORT_CHECKS


def test_no_finalization_writes_nothing(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    # A continue verdict leaves the task in_progress — no finalization.
    store.transition("begin", "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "continue"})

    assert nudge is None
    assert _active_probes(tmp_path) == []


def test_probe_write_failure_is_logged_not_raised(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    def boom(session_id, item):
        raise OSError("disk full")

    monkeypatch.setattr(task_manager, "_write_probe", boom)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None  # a failed write never blocks the close
    assert store.read()[0]["status"] == "completed"


def test_probe_skipped_without_session(monkeypatch, tmp_path) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing")
    agent = _make_agent(store)
    agent.session_id = ""
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert _active_probes(tmp_path) == []


# ── module derivation from the session diff ──────────────────────────


def test_changed_modules_derives_dotted_names(monkeypatch) -> None:
    monkeypatch.setattr(
        task_manager,
        "_git_diff",
        lambda session_id: (
            "diff --git a/agent/task_manager.py b/agent/task_manager.py\n"
            "diff --git a/hermes_cli/tasks.py b/hermes_cli/tasks.py\n"
        ),
    )
    assert task_manager._changed_modules("s") == [
        "agent.task_manager",
        "hermes_cli.tasks",
    ]


def test_changed_modules_excludes_tests_and_non_python(monkeypatch) -> None:
    monkeypatch.setattr(
        task_manager,
        "_git_diff",
        lambda session_id: (
            "diff --git a/tests/agent/test_task_review.py b/tests/agent/test_task_review.py\n"
            "diff --git a/README.md b/README.md\n"
            "diff --git a/agent/task_manager.py b/agent/task_manager.py\n"
        ),
    )
    assert task_manager._changed_modules("s") == ["agent.task_manager"]


def test_changed_modules_empty_without_diff(monkeypatch) -> None:
    monkeypatch.setattr(task_manager, "_git_diff", lambda session_id: None)
    assert task_manager._changed_modules("s") == []
