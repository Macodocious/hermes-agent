"""Tests for the post-close review step (P6) in task_manager.

Covers the mechanical side of the mandatory second set of eyes: plan
resolution (explicit ref → item content → inline origin window, fail-closed
when none resolves), verdict parsing (fail-open on unusable output), the
fix-task spawn (source=review, lineage review_of), the lineage depth cap
(mechanical escalation past tasks.review.max_rounds), and the config gate.

The review runner itself (``_run_review``) is exercised with a faked
``call_llm`` — no real LLM calls ever fire in these tests.
"""

from types import SimpleNamespace

import pytest

from agent import task_manager
from tools.todo_tool import TodoStore


def _make_agent(store: TodoStore) -> SimpleNamespace:
    return SimpleNamespace(
        _todo_store=store,
        session_id="review-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )


def _seed(store: TodoStore, item_id: str, content: str, **extra) -> None:
    item = {"id": item_id, "content": content, "status": "pending"}
    item.update(extra)
    store.write([item])


@pytest.fixture(autouse=True)
def _review_on(monkeypatch) -> None:
    """Pin lifecycle + review config so tests are independent of the host."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr(task_manager, "_review_config", lambda: {"enabled": True, "max_rounds": 2})


def _PLAN() -> str:
    """A substantial task content that resolves as a self-contained plan."""
    return (
        "Build the thing per the spec. The spec requires a deterministic "
        "task lifecycle with a code-enforced state machine, a two-key close "
        "with a judge verdict, a turn-end audit that refuses clean ends for "
        "work without an open task, and a post-close review that spawns fix "
        "tasks mechanically. The implementation must be modular, testable, "
        "and delivered as a pull request per the house procedure. "
    )


def _close_in_flight(store: TodoStore, item_id: str) -> None:
    """Drive a task through begin + close, leaving it in closing.

    The review fires on the verdict observation that finalizes the task
    (the E2E pattern): begin → close → observe_verdict(done).
    """
    store.transition("begin", item_id)
    store.transition("close", item_id)


# ── plan resolution ───────────────────────────────────────────────────


def test_plan_resolution_explicit_ref_wins(monkeypatch, tmp_path) -> None:
    plan_file = tmp_path / "plan.md"
    plan_file.write_text("# The Plan\nBuild the thing.", encoding="utf-8")
    store = TodoStore()
    _seed(store, "1", "Build the thing", plan=str(plan_file))
    item = store.read()[0]

    plan = task_manager._resolve_plan("review-session", item)

    assert plan is not None
    assert plan["source"] == "ref"
    assert "Build the thing" in plan["content"]


def test_plan_resolution_falls_through_to_item_content(monkeypatch) -> None:
    long_plan = (
        "Build the thing per the spec. The spec requires a deterministic "
        "task lifecycle with a code-enforced state machine, a two-key close "
        "with a judge verdict, a turn-end audit that refuses clean ends for "
        "work without an open task, and a post-close review that spawns fix "
        "tasks mechanically. The implementation must be modular, testable, "
        "and delivered as a pull request per the house procedure. " * 2
    )
    store = TodoStore()
    _seed(store, "1", long_plan)
    item = store.read()[0]

    plan = task_manager._resolve_plan("review-session", item)

    assert plan is not None
    assert plan["source"] == "item"
    assert plan["content"] == long_plan.strip()


def test_plan_resolution_inline_origin_window(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing", origin="msg-42")
    item = store.read()[0]

    class FakeDB:
        def get_messages(self, session_id, limit=None):
            return [
                {"role": "user", "content": "here is the plan", "platform_message_id": "msg-42"},
                {"role": "assistant", "content": "on it", "platform_message_id": "msg-43"},
            ]

    monkeypatch.setattr(
        "hermes_cli.tasks._get_session_db", lambda: FakeDB()
    )

    plan = task_manager._resolve_plan("review-session", item)

    assert plan is not None
    assert plan["source"] == "inline"
    assert "here is the plan" in plan["content"]


def test_plan_resolution_fail_closed_when_none_resolves(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", "Build the thing", origin="msg-42")
    item = store.read()[0]

    class FakeDB:
        def get_messages(self, session_id, limit=None):
            return []  # origin id never appears

    monkeypatch.setattr(
        "hermes_cli.tasks._get_session_db", lambda: FakeDB()
    )

    plan = task_manager._resolve_plan("review-session", item)

    assert plan is None


# ── verdict parsing ───────────────────────────────────────────────────


def test_parse_review_response_pass() -> None:
    verdict, findings = task_manager._parse_review_response(
        '{"verdict": "pass", "findings": []}'
    )
    assert verdict == "pass"
    assert findings == []


def test_parse_review_response_needs_work() -> None:
    verdict, findings = task_manager._parse_review_response(
        '{"verdict": "needs_work", "findings": ["missing tests", "no docs"]}'
    )
    assert verdict == "needs_work"
    assert findings == ["missing tests", "no docs"]


def test_parse_review_response_fenced_json() -> None:
    verdict, findings = task_manager._parse_review_response(
        '```json\n{"verdict": "needs_work", "findings": ["gap"]}\n```'
    )
    assert verdict == "needs_work"
    assert findings == ["gap"]


def test_parse_review_response_fail_open_on_garbage() -> None:
    verdict, findings = task_manager._parse_review_response("sorry, no json here")
    assert verdict == "pass"
    assert findings == []


def test_parse_review_response_empty_findings_downgrades_to_pass() -> None:
    verdict, findings = task_manager._parse_review_response(
        '{"verdict": "needs_work", "findings": []}'
    )
    assert verdict == "pass"
    assert findings == []


# ── the review hook (close → review → fix task) ───────────────────────


def test_close_triggers_review_and_spawns_fix_task(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", _PLAN())
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    def fake_review(packet):
        return "needs_work", ["the spec says X but the code does Y"]

    monkeypatch.setattr(task_manager, "_run_review", fake_review)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is not None
    assert "begin the fix task" in nudge.lower()
    items = store.read()
    fix = next(i for i in items if i.get("review_of") == "1")
    assert fix["status"] == "pending"
    assert fix["source"] == "review"
    assert "the spec says X" in fix["content"]


def test_review_pass_spawns_nothing(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", _PLAN())
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    monkeypatch.setattr(task_manager, "_run_review", lambda packet: ("pass", []))

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert not any(i.get("review_of") == "1" for i in store.read())


def test_review_fail_closed_without_plan(monkeypatch) -> None:
    """No plan resolves → the review does not run; the close is held."""
    store = TodoStore()
    _seed(store, "1", "Build the thing", origin="msg-42")
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    monkeypatch.setattr(
        "hermes_cli.tasks._get_session_db", lambda: FakeEmptyDB()
    )
    called = []

    def fake_review(packet):
        called.append(packet)
        return "needs_work", ["should never fire"]

    monkeypatch.setattr(task_manager, "_run_review", fake_review)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert called == []  # the review never ran
    assert store.read()[0]["status"] == "completed"  # close held


class FakeEmptyDB:
    def get_messages(self, session_id, limit=None):
        return []


def test_review_transport_failure_fails_open(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", _PLAN())
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)

    def boom(**kwargs):
        raise RuntimeError("transport down")

    # The fail-open lives INSIDE _run_review: a transport failure on the
    # review call must never block the close. Patch call_llm (the real
    # layer), not _run_review.
    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert store.read()[0]["status"] == "completed"
    assert not any(i.get("review_of") == "1" for i in store.read())


def test_review_disabled_by_config(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", _PLAN())
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    monkeypatch.setattr(task_manager, "_review_config", lambda: {"enabled": False})
    called = []

    def fake_review(packet):
        called.append(packet)
        return "needs_work", ["x"]

    monkeypatch.setattr(task_manager, "_run_review", fake_review)

    _close_in_flight(store, "1")
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    assert called == []


# ── lineage depth cap ─────────────────────────────────────────────────


def test_review_lineage_cap_escalates(monkeypatch) -> None:
    store = TodoStore()
    _seed(store, "1", _PLAN())
    agent = _make_agent(store)
    monkeypatch.setattr(task_manager, "_persist", lambda a: None)
    monkeypatch.setattr(task_manager, "_run_review", lambda packet: ("needs_work", ["gap"]))

    def _close_and_observe(item_id: str) -> None:
        store.transition("begin", item_id)
        store.transition("close", item_id)
        task_manager.observe_verdict(agent, {"verdict": "done"})

    def _pending_fix() -> str:
        return next(
            i["id"]
            for i in store.read()
            if i.get("review_of") == "1" and i["status"] == "pending"
        )

    # Round 1: the original task closes; the review spawns a pending fix
    # task (lineage review_of -> 1). Depth 0 < cap 2.
    _close_and_observe("1")
    assert _pending_fix() is not None

    # Round 2: the first fix closes and fails review. Depth 1 < cap 2 —
    # another fix task is spawned.
    _close_and_observe(_pending_fix())
    assert _pending_fix() is not None

    # Round 3: the second fix closes and fails review. Depth 2 >= cap 2 —
    # the next fix task is mechanically escalated (no nudge, no new
    # pending fix).
    nudge = None
    fix2 = _pending_fix()
    store.transition("begin", fix2)
    store.transition("close", fix2)
    nudge = task_manager.observe_verdict(agent, {"verdict": "done"})

    assert nudge is None
    escalated = [
        i for i in store.read()
        if i.get("review_of") == "1" and i["status"] == "escalated"
    ]
    assert len(escalated) == 1
    assert "REVIEW ESCALATED" in escalated[0]["content"]
    assert "cap 2" in escalated[0]["content"]


def test_review_lineage_depth_counts_existing_rounds() -> None:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "task", "status": "completed"},
            {"id": "review-1-1", "content": "fix", "status": "pending", "source": "review", "review_of": "1"},
        ]
    )
    assert task_manager._review_lineage_depth(store, "1") == 1


# ── the review runner (faked call_llm) ────────────────────────────────


def test_run_review_parses_verdict(monkeypatch) -> None:
    class FakeResp:
        class FakeChoice:
            class FakeMessage:
                content = '{"verdict": "needs_work", "findings": ["gap"]}'

            message = FakeMessage()

        choices = [FakeChoice()]

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", lambda **kwargs: FakeResp()
    )

    verdict, findings = task_manager._run_review({"impl": "code", "plan": "spec"})

    assert verdict == "needs_work"
    assert findings == ["gap"]


def test_run_review_fails_open_on_transport(monkeypatch) -> None:
    def boom(**kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)

    verdict, findings = task_manager._run_review({"impl": "code", "plan": "spec"})

    assert verdict == "pass"
    assert findings == []
