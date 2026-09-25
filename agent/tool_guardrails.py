"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed


IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "cronjob",
        "delegate_task",
        "process",
    }
)

# Coarse repeat detection. Exact-args keying (sha256 of canonical args) misses
# reconnaissance loops that vary offset/limit/query slightly: every variation
# is a new signature, so the threshold is never reached. These tools are keyed
# on their *target* instead, so a window of near-identical calls accumulates
# one count.
READ_FILE_TOOL = "read_file"
SEARCH_FILES_TOOL = "search_files"
FILE_MUTATION_TOOLS = frozenset({"write_file", "patch"})
# Mirrors the read_file_tool signature defaults in tools/file_tools.py.
READ_FILE_DEFAULT_OFFSET = 1
READ_FILE_DEFAULT_LIMIT = 500


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    repeat_target_warn_after: int = 4
    repeat_target_deny_after: int = 8
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
            repeat_target_warn_after=_positive_int(
                warn_after.get("repeat_target", data.get("repeat_target_warn_after")),
                defaults.repeat_target_warn_after,
            ),
            repeat_target_deny_after=_positive_int(
                hard_stop_after.get("repeat_target", data.get("repeat_target_deny_after")),
                defaults.repeat_target_deny_after,
            ),
        )


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | deny | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self.reset_for_turn()

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._repeat_counts: dict[tuple[str, ...], int] = {}
        self._read_coverage: dict[str, list[tuple[int, int]]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        # Coarse target repeat: deny (never halt) once a read/search target has
        # been repeated past the deny threshold. Deny refuses this single call
        # and leaves the turn running; the synthetic result carries a coverage
        # manifest so the model can self-heal from what it already has.
        coarse = self._coarse_state(tool_name, args)
        if coarse is not None:
            key, redundant, count = coarse
            if redundant and count >= self.config.repeat_target_deny_after:
                return ToolGuardrailDecision(
                    action="deny",
                    code="repeat_target_deny",
                    message=self._repeat_deny_message(tool_name, key, count),
                    tool_name=tool_name,
                    count=count,
                    signature=signature,
                )

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: this read-only call returned the same "
                            f"result {repeat_count} times. Stop repeating it unchanged; "
                            "use the result already provided or try a different query."
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this turn. "
                        "Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        # A landed file mutation invalidates read coverage for that path:
        # read → edit → read is legitimate, so the post-edit re-read must not
        # count as a redundant repeat.
        if tool_name in FILE_MUTATION_TOOLS:
            self._clear_read_coverage(args)

        coarse_warn = self._record_coarse_repeat(tool_name, args, signature)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return coarse_warn or ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return coarse_warn or ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools

    def _coarse_state(
        self, tool_name: str, args: Mapping[str, Any]
    ) -> tuple[tuple[str, ...], bool, int] | None:
        """Return (key, is_redundant, count_including_this_call) for a target-keyed tool.

        ``count`` is the total number of touches of this target in the window.
        ``read_file`` is redundant when the requested line range is already
        covered; ``search_files`` is redundant when the same
        (pattern, path, target) has already been searched. Returns None for
        every other tool.
        """
        key = _coarse_key(tool_name, args)
        if key is None:
            return None
        count = self._repeat_counts.get(key, 0) + 1
        if tool_name == SEARCH_FILES_TOOL:
            return key, count > 1, count
        start, end = _read_range(args)
        covered = self._read_coverage.get(key[1], [])
        return key, _range_covered(covered, start, end), count

    def _record_coarse_repeat(
        self, tool_name: str, args: Mapping[str, Any], signature: ToolCallSignature
    ) -> ToolGuardrailDecision | None:
        """Record a coarse target touch; warn once the repeat threshold is met."""
        state = self._coarse_state(tool_name, args)
        if state is None:
            return None
        key, redundant, count = state
        self._repeat_counts[key] = count
        if tool_name == READ_FILE_TOOL and not redundant:
            start, end = _read_range(args)
            self._read_coverage[key[1]] = _merge_range(
                self._read_coverage.get(key[1], []), start, end
            )
            return None
        if not redundant:
            return None
        if self.config.warnings_enabled and count >= self.config.repeat_target_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="repeat_target_warning",
                message=self._repeat_warn_message(tool_name, key, count),
                tool_name=tool_name,
                count=count,
                signature=signature,
            )
        return None

    def _clear_read_coverage(self, args: Mapping[str, Any]) -> None:
        """Drop read coverage invalidated by a landed file mutation."""
        path = args.get("path")
        if isinstance(path, str) and path:
            self._read_coverage.pop(path, None)
            self._repeat_counts.pop((READ_FILE_TOOL, path), None)
            return
        # V4A multi-file patches carry their targets inside the patch body, not
        # in ``path``. We cannot attribute the mutation to one file, so drop all
        # read coverage — the safe direction for a nudge (never over-block).
        self._read_coverage.clear()
        self._repeat_counts = {
            key: count
            for key, count in self._repeat_counts.items()
            if key[0] != READ_FILE_TOOL
        }

    def _repeat_warn_message(self, tool_name: str, key: tuple[str, ...], count: int) -> str:
        if tool_name == READ_FILE_TOOL:
            ranges = self._read_coverage.get(key[1], [])
            covered = f" (lines {_format_ranges(ranges)} already covered)" if ranges else ""
            return (
                f"{tool_name} has re-read {key[1]} {count} times in this window"
                f"{covered}. The content is already in your context — use it "
                "instead of reading it again."
            )
        return (
            f"{tool_name} has repeated the search {key[1]!r} in {key[2] or '.'} "
            f"{count} times in this window. Use the results already provided or "
            "narrow the query instead of repeating it."
        )

    def _repeat_deny_message(self, tool_name: str, key: tuple[str, ...], count: int) -> str:
        if tool_name == READ_FILE_TOOL:
            header = (
                f"Denied {tool_name}: {key[1]} has already been read {count} times "
                "in this window. The content is already in your context — use it "
                "instead of re-reading."
            )
        else:
            header = (
                f"Denied {tool_name}: the search {key[1]!r} in {key[2] or '.'} has "
                f"already been run {count} times in this window. Use the results "
                "already provided or narrow the query."
            )
        manifest = self._coverage_manifest()
        if not manifest:
            return header
        return header + "\n\nAlready covered this window:\n" + "\n".join(manifest)

    def _coverage_manifest(self, limit: int = 10) -> list[str]:
        """Compact list of what this window has already read and searched."""
        lines: list[str] = []
        reads = sorted(
            (path, ranges) for path, ranges in self._read_coverage.items() if ranges
        )
        for path, ranges in reads[:limit]:
            lines.append(f"  {path}  lines {_format_ranges(ranges)}")
        if len(reads) > limit:
            lines.append(f"  …and {len(reads) - limit} more files")
        searches = sorted(
            (key, count)
            for key, count in self._repeat_counts.items()
            if key[0] == SEARCH_FILES_TOOL
        )
        for key, count in searches[:limit]:
            lines.append(f"  search {key[1]!r} in {key[2] or '.'} ×{count}")
        if len(searches) > limit:
            lines.append(f"  …and {len(searches) - limit} more searches")
        return lines


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _coarse_key(tool_name: str, args: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Coarse repeat key for a target-keyed tool, else None."""
    if tool_name == READ_FILE_TOOL:
        path = args.get("path")
        if isinstance(path, str) and path:
            return (READ_FILE_TOOL, path)
        return None
    if tool_name == SEARCH_FILES_TOOL:
        pattern = args.get("pattern")
        if isinstance(pattern, str) and pattern:
            return (
                SEARCH_FILES_TOOL,
                pattern,
                str(args.get("path") or ""),
                str(args.get("target") or ""),
            )
        return None
    return None


def _read_range(args: Mapping[str, Any]) -> tuple[int, int]:
    """Half-open line range [start, end) a read_file call will cover."""
    start = _positive_int(args.get("offset"), READ_FILE_DEFAULT_OFFSET)
    span = _positive_int(args.get("limit"), READ_FILE_DEFAULT_LIMIT)
    return start, start + span


def _range_covered(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    """True when [start, end) is fully inside one merged coverage range."""
    return any(s <= start and end <= e for s, e in ranges)


def _merge_range(ranges: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    """Add [start, end) to a sorted, disjoint range list, merging overlaps."""
    merged: list[tuple[int, int]] = []
    for s, e in sorted(ranges + [(start, end)]):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _format_ranges(ranges: list[tuple[int, int]]) -> str:
    """Render half-open ranges as inclusive line spans, e.g. '1-500, 700-900'."""
    return ", ".join(f"{start}-{end - 1}" for start, end in ranges)


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _sha256(value: str) -> str:
    # surrogatepass: tool results scraped from the web can carry unpaired
    # UTF-16 surrogates (e.g. half of a mathematical-bold pair); a strict
    # encode raises and takes down the whole conversation loop. The hash only
    # needs deterministic bytes, not valid UTF-8.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
