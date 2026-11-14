"""Shim for tool discovery. Registers `use_computer` with tools.registry.

The real implementation lives in the `tools/use_computer/` package to keep
the file structure clean. This shim exists because tools.registry auto-imports
`tools/*.py` — we need a top-level module to trigger the registration.
"""

from __future__ import annotations

from tools.use_computer.schema import COMPUTER_USE_SCHEMA
from tools.use_computer.tool import (
    check_computer_use_requirements,
    handle_computer_use,
    set_approval_callback,
)
from tools.registry import registry


registry.register(
    name="use_computer",
    toolset="use_computer",
    schema=COMPUTER_USE_SCHEMA,
    handler=lambda args, **kw: handle_computer_use(args, **kw),
    check_fn=check_computer_use_requirements,
    requires_env=[],
    description=(
        "Universal desktop control via cua-driver (macOS, Windows, Linux). Works with any "
        "tool-capable model (Anthropic, OpenAI, OpenRouter, local vLLM, "
        "etc.). Background computer-use: does NOT steal the user's cursor "
        "or keyboard focus."
    ),
)


__all__ = [
    "handle_computer_use",
    "set_approval_callback",
    "check_computer_use_requirements",
]
