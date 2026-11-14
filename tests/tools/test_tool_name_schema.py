"""Guard: core tool names read action-first, and vendor payloads keep their own names.

Two failures this guard exists to catch:

1. A new or changed core tool whose name does not lead with its action word.
2. A vendor API payload key whose tool type was rewritten to a Hermes tool name,
   which sends an unknown tool type to that vendor.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"

# The action words the core surface is allowed to lead with. A new tool whose
# first segment is absent here is a deliberate decision, not an accident.
ACTION_WORDS = {
    "add", "analyze", "attach", "back", "block", "call", "click", "close",
    "comment", "complete", "create", "delegate", "edit", "execute", "extend",
    "extract", "generate", "get", "heartbeat", "link", "list", "manage",
    "navigate", "patch", "press", "query", "read", "reply", "scroll", "search",
    "send", "show", "snapshot", "switch", "type", "unblock", "use", "view",
    "write",
}

# Multi-word names that carry no action word to lead with, so the rule cannot
# reshape them. Recorded deliberately; nothing new may join this set silently.
NO_ACTION_WORD = {
    "browser_cdp", "browser_console", "browser_dialog", "browser_vision",
    "health_report", "kanban_attachments", "text_to_speech",
}

# Tools that ship from a plugin surface and are named by that surface.
PLUGIN_SURFACE = {"discord", "discord_admin"}

# Names a vendor defines for its own server-side tools. A Hermes tool must
# never take one of these, because the same string then means two things.
VENDOR_TOOL_TYPES = {
    "web_search", "web_search_preview", "x_search", "code_interpreter",
    "file_search", "computer_use_preview",
}

SOURCE_DIRS = ("tools", "agent", "plugins", "gateway", "hermes_cli")

PAYLOAD_KEY = re.compile(r"""["']type["']\s*:\s*["']([a-z0-9_]+)["']""")


_COLLECT_NAMES = """
import importlib, json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from tools.registry import registry

for path in sorted(Path(sys.argv[1], "tools").glob("*.py")):
    if path.stem in {"registry", "__init__"}:
        continue
    try:
        importlib.import_module("tools." + path.stem)
    except Exception:
        continue

tools = getattr(registry, "tools", None) or getattr(registry, "_tools", {}) or {}
print(json.dumps(sorted(tools)))
"""


def _registered_names() -> set[str]:
    """Every tool name the registry holds in a clean interpreter.

    Collected in a subprocess because sibling test modules register their own
    fake tools in the shared registry, and the guard judges the shipped surface.
    """
    result = subprocess.run(
        [sys.executable, "-c", _COLLECT_NAMES, str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"could not collect tool names: {result.stderr.strip()[-2000:]}")
    names = json.loads(result.stdout.strip().splitlines()[-1])
    return set(names)


def test_every_tool_name_leads_with_an_action_word():
    names = _registered_names()
    assert names, "no tools registered — the guard cannot judge an empty surface"
    offenders = sorted(
        name for name in names
        if "_" in name
        and name not in NO_ACTION_WORD
        and name not in PLUGIN_SURFACE
        and name.split("_")[0] not in ACTION_WORDS
    )
    assert not offenders, (
        "tool names whose first segment is not a known action word: "
        f"{offenders}. Add the word to ACTION_WORDS only when the name is deliberate."
    )


def test_no_tool_takes_a_vendor_tool_type_as_its_name():
    names = _registered_names()
    collisions = sorted(names & VENDOR_TOOL_TYPES)
    assert not collisions, f"tool names collide with vendor tool types: {collisions}"


def test_no_source_file_sends_a_tool_name_as_a_vendor_payload_type():
    names = _registered_names()
    offenders: list[str] = []
    for directory in SOURCE_DIRS:
        for path in (REPO_ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                for match in PAYLOAD_KEY.finditer(line):
                    if match.group(1) in names:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number} — {match.group(0)}")
    assert not offenders, (
        "a vendor payload carries a Hermes tool name as its tool type: "
        + "; ".join(offenders)
    )
