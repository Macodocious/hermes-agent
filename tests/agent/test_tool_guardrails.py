"""Pure tool-call guardrail primitive tests."""

import json

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    canonical_tool_args,
    classify_tool_failure,
)


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)


def test_default_config_is_soft_warning_only_with_hard_stop_disabled():
    cfg = ToolCallGuardrailConfig()

    assert cfg.warnings_enabled is True
    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 2
    assert cfg.same_tool_failure_warn_after == 3
    assert cfg.no_progress_warn_after == 2
    assert cfg.exact_failure_block_after == 5
    assert cfg.same_tool_failure_halt_after == 8
    assert cfg.no_progress_block_after == 5


def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_gateway_platform_defaults_to_hard_stop_without_changing_cli_default():
    cli_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cli")
    telegram_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="telegram")
    cron_cfg = ToolCallGuardrailConfig.from_mapping({}, platform="cron")

    assert cli_cfg.hard_stop_enabled is False
    assert telegram_cfg.hard_stop_enabled is True
    assert cron_cfg.hard_stop_enabled is True


def test_non_interactive_hard_stop_can_be_disabled_explicitly():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"non_interactive_hard_stop_enabled": False},
        platform="telegram",
    )

    assert cfg.hard_stop_enabled is False
    assert cfg.non_interactive_hard_stop_enabled is False


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    assert controller.halt_decision is None


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(no_progress_warn_after=2, no_progress_block_after=2)
    )

    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"






def _coarse_controller(**overrides) -> ToolCallGuardrailController:
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig(hard_stop_enabled=True, **overrides)
    )


def _varying_read_calls(path: str, count: int) -> list[dict]:
    """Distinct-args read_file calls that all land inside the first read's range.

    The real reconnaissance loop varied offset/limit on every call, so exact-args
    keying saw a new signature each time and never accumulated a count. These
    calls reproduce that: every signature differs, every range is already
    covered by the first read.
    """
    return [{"path": path, "offset": 1, "limit": 500 - i} for i in range(count)]


def test_read_file_repeat_is_keyed_on_path_and_line_coverage_not_exact_args():
    # Same path + already covered lines must count as one repeat regardless of
    # the args that differ between calls.
    controller = _coarse_controller()

    assert controller.before_call("read_file", {"path": "/tmp/a.py", "offset": 1, "limit": 500}).action == "allow"
    controller.after_call("read_file", {"path": "/tmp/a.py", "offset": 1, "limit": 500}, "body", failed=False)

    # Same range, different limit spelling -> still redundant.
    assert controller.before_call("read_file", {"path": "/tmp/a.py", "offset": 1, "limit": 500}).action == "allow"
    controller.after_call("read_file", {"path": "/tmp/a.py", "offset": 1, "limit": 500}, "body", failed=False)

    # A genuinely new line range is NOT a repeat.
    assert controller.before_call("read_file", {"path": "/tmp/a.py", "offset": 900, "limit": 100}).action == "allow"
    controller.after_call("read_file", {"path": "/tmp/a.py", "offset": 900, "limit": 100}, "more", failed=False)

    # A different file is its own key.
    assert controller.before_call("read_file", {"path": "/tmp/b.py", "offset": 1, "limit": 500}).action == "allow"


def _coarse_signals(controller, tool_name: str, calls: list[dict]) -> list[str]:
    """Effective per-call signal: deny from before_call, warn from after_call.

    Mirrors the runtime split — ``before_call`` refuses a call (deny), while
    ``after_call`` appends guidance to the result (warn).
    """
    signals = []
    for args in calls:
        decision = controller.before_call(tool_name, args)
        if not decision.allows_execution:
            signals.append(decision.action)
            continue
        after = controller.after_call(tool_name, args, "body", failed=False)
        signals.append(after.action)
    return signals


def test_read_file_warns_at_four_and_denies_at_eight_without_halting():
    controller = _coarse_controller()

    signals = _coarse_signals(controller, "read_file", _varying_read_calls("/tmp/hooks.py", 8))

    # warn on the 4th touch, deny on the 8th — Mac's warn-4 / deny-8 spec.
    assert signals == ["allow", "allow", "allow", "warn", "warn", "warn", "warn", "deny"]

    denied = controller.before_call("read_file", {"path": "/tmp/hooks.py", "offset": 1, "limit": 500})
    assert denied.action == "deny"
    assert denied.code == "repeat_target_deny"
    # Deny refuses the call but must NOT halt the turn.
    assert denied.allows_execution is False
    assert denied.should_halt is False
    assert controller.halt_decision is None
    # The reason carries the coverage manifest so the model can self-heal.
    assert "/tmp/hooks.py" in denied.message
    assert "lines 1-500" in denied.message


def test_search_files_repeat_is_keyed_on_pattern_path_target():
    controller = _coarse_controller()
    base = {"pattern": "transform_llm_output", "path": "/usr/local/lib/hermes-agent", "target": "content"}
    calls = [{**base, "limit": 50 - i} for i in range(8)]  # distinct signature, same coarse key

    signals = _coarse_signals(controller, "search_files", calls)

    assert signals == ["allow", "allow", "allow", "warn", "warn", "warn", "warn", "deny"]

    # A different pattern is a different key and stays allowed.
    assert controller.before_call("search_files", {**base, "pattern": "other"}).action == "allow"


def test_file_mutation_clears_read_coverage_so_post_edit_reread_is_allowed():
    # read -> edit -> read is legitimate; the post-edit re-read must not be
    # counted as a redundant repeat of the pre-edit content.
    controller = _coarse_controller()

    for args in _varying_read_calls("/tmp/planner.py", 7):
        controller.before_call("read_file", args)
        controller.after_call("read_file", args, "body", failed=False)

    assert controller.before_call("read_file", {"path": "/tmp/planner.py", "offset": 1, "limit": 500}).action == "deny"

    controller.after_call("patch", {"path": "/tmp/planner.py", "old_string": "a", "new_string": "b"}, "ok", failed=False)

    assert controller.before_call("read_file", {"path": "/tmp/planner.py", "offset": 1, "limit": 500}).action == "allow"


def test_coarse_repeat_state_is_cleared_by_reset_for_turn():
    controller = _coarse_controller()
    for args in _varying_read_calls("/tmp/run.py", 8):
        controller.before_call("read_file", args)
        controller.after_call("read_file", args, "body", failed=False)
    assert controller.before_call("read_file", {"path": "/tmp/run.py", "offset": 1, "limit": 500}).action == "deny"

    controller.reset_for_turn()

    assert controller.before_call("read_file", {"path": "/tmp/run.py", "offset": 1, "limit": 500}).action == "allow"


def test_compaction_reset_clears_read_coverage_so_reread_is_allowed():
    # Context compaction summarises read content out of the window, but the
    # guardrail's own reset runs only at turn start. A mid-turn compaction must
    # clear read coverage so the model can re-read what it no longer holds.
    controller = _coarse_controller()
    for args in _varying_read_calls("/tmp/compacted.py", 8):
        controller.before_call("read_file", args)
        controller.after_call("read_file", args, "body", failed=False)
    assert controller.before_call(
        "read_file", {"path": "/tmp/compacted.py", "offset": 1, "limit": 500}
    ).action == "deny"

    controller.reset_read_coverage()

    assert controller.before_call(
        "read_file", {"path": "/tmp/compacted.py", "offset": 1, "limit": 500}
    ).action == "allow"


def test_compaction_reset_keeps_search_repeat_counts():
    # The reset is read-scoped: compaction does not invalidate search repeats,
    # so their counts must survive it.
    controller = _coarse_controller()
    base = {"pattern": "transform_llm_output", "path": "/usr/local/lib/hermes-agent", "target": "content"}
    calls = [{**base, "limit": 50 - i} for i in range(8)]
    for args in calls:
        controller.before_call("search_files", args)
        controller.after_call("search_files", args, "body", failed=False)
    assert controller.before_call("search_files", base).action == "deny"

    controller.reset_read_coverage()

    assert controller.before_call("search_files", base).action == "deny"
# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True










