"""Focused checks for the additive core extension seams.

Covers the five seams that replace the approval-gate plugin's monkey-patches:
guard_decision, tool_schema, the pre_tool_call reasoning kwarg,
pre_batch_dispatch, and approval_presentation. Each seam must be a no-op when
no plugin registers it (fail-closed / byte-for-byte unchanged).
"""
from types import SimpleNamespace

import pytest


def test_new_hooks_are_registered():
    from hermes_cli.plugins import VALID_HOOKS
    assert {
        "guard_decision",
        "tool_schema",
        "pre_batch_dispatch",
        "approval_presentation",
    } <= VALID_HOOKS


# ── guard_decision ────────────────────────────────────────────────────────

def test_guard_decision_skip(monkeypatch):
    import tools.approval as am
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kw: [{"approved": True}],
    )
    assert am._invoke_guard_decision("rm -rf /", "local") == {
        "approved": True, "message": None, "gate_approved": True,
    }


def test_guard_decision_absent_runs_native(monkeypatch):
    import tools.approval as am
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    assert am._invoke_guard_decision("rm -rf /", "local") is None


def test_guard_decision_malformed_runs_native(monkeypatch):
    import tools.approval as am
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kw: [{"approved": "yes"}, None, 5],
    )
    assert am._invoke_guard_decision("x", "local") is None


# ── tool_schema ────────────────────────────────────────────────────────────

def test_merge_schema_fragment_is_additive():
    from tools.registry import _merge_schema_fragment
    schema = {"properties": {"path": {"type": "string"}}, "required": ["path"]}
    _merge_schema_fragment(
        schema,
        {"properties": {"reason": {"type": "string"}}, "required": ["reason"]},
    )
    assert set(schema["properties"]) == {"path", "reason"}
    assert schema["required"] == ["path", "reason"]


def test_tool_schema_contributions_absent(monkeypatch):
    from tools import registry
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    assert registry._get_tool_schema_contributions() == {}


# ── reasoning passthrough ───────────────────────────────────────────────────

def test_extract_assistant_reasoning_precedence():
    from agent.tool_executor import _extract_assistant_reasoning
    msg = SimpleNamespace(reasoning="why", reasoning_content="rc", content="c")
    assert _extract_assistant_reasoning(msg) == "why"
    assert _extract_assistant_reasoning({"reasoning_content": "rc"}) == "rc"
    assert _extract_assistant_reasoning(None) == ""


# ── approval_presentation ───────────────────────────────────────────────────

def test_approval_presentation_spec(monkeypatch):
    import tools.approval as am
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kw: [{"title": "T", "timeout": 30}],
    )
    assert am._invoke_approval_presentation("terminal", {}, "r", "s") == {
        "title": "T", "timeout": 30,
    }


def test_approval_presentation_absent(monkeypatch):
    import tools.approval as am
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    assert am._invoke_approval_presentation("terminal", {}, "r", "s") is None


# ── pre_batch_dispatch ──────────────────────────────────────────────────────

def test_pre_batch_dispatch_block_appends_result_per_call(monkeypatch):
    import run_agent
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook", lambda name: name == "pre_batch_dispatch"
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kw: [{"action": "block", "message": "nope"}],
    )
    calls = [
        SimpleNamespace(function=SimpleNamespace(name="patch"), id="c1"),
        SimpleNamespace(function=SimpleNamespace(name="write_file"), id="c2"),
    ]
    messages = []
    fake_self = SimpleNamespace(session_id="s")
    run_agent.AIAgent._execute_tool_calls(
        fake_self, SimpleNamespace(tool_calls=calls), messages, "task", 0
    )
    assert len(messages) == 2
    assert all("nope" in str(m) for m in messages)


def test_pre_batch_dispatch_absent_dispatches_normally(monkeypatch):
    import run_agent
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: False)
    calls = [SimpleNamespace(function=SimpleNamespace(name="patch"), id="c1")]
    messages = []
    fake_self = SimpleNamespace(session_id="s")
    # No hook → falls through to the real dispatch path, which needs a real
    # agent; assert only that the hook path appended nothing.
    with pytest.raises(Exception):
        run_agent.AIAgent._execute_tool_calls(
            fake_self, SimpleNamespace(tool_calls=calls), messages, "task", 0
        )
    assert messages == []
