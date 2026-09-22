"""Task-card rendering for the /tasks command.

Builds the approved "progress-forward" card as a platform-agnostic spec dict.
The Discord adapter turns the spec into a Components V2 message; the CLI
renderer in ``hermes_cli/cli_commands_mixin`` stays plain text and does not use
this module.

Layout
------
    📋 Current Tasks
    In Progress
    ►  <active task>
    Up Next
    ○  <queued task>
    ...
    Done
    > ✓  <completed task>      <- faded, same size
    ...
    ███████░░░░░   7 / 12 · 3h 12m elapsed

The card renders as three stacked blocks in a single container:

1. A heading text display carrying the title glyph.
2. One text display per non-empty section, each opened by a bold title-case
   label.  Section labels borrow the interactive approval prompt's field-name
   styling (bold, title case, glyph-free); row text is regular weight.
3. A subtext footer carrying the twelve-segment bar and the ``7 / 12`` ratio,
   separated by spacing separators that reproduce the drawn vertical rhythm.

Completed rows are prefixed with ``> `` because a blockquote is the only
Discord primitive that mutes text without shrinking it (``-#`` renders
``.875rem`` and a code fence is monospace at ``.75rem``).  The approved design
fades completed rows by colour alone, at body size.

The bar and the ``7 / 12`` ratio are both derived from the same counts, so they
cannot disagree.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from hermes_constants import (
    TASK_CARD_BAR_EMPTY,
    TASK_CARD_BAR_FILLED,
    TASK_CARD_FADE_PREFIX,
    TASK_CARD_FOOTER_CODE_FENCE,
    TASK_CARD_FOOTER_SEPARATOR,
    TASK_CARD_FOOTER_SUBTEXT_PREFIX,
    TASK_CARD_GLYPH_ACTIVE,
    TASK_CARD_GLYPH_DONE,
    TASK_CARD_GLYPH_PENDING,
    TASK_CARD_MAX_CHARS,
    TASK_CARD_MAX_ROWS_PER_SECTION,
    TASK_CARD_SECTION_DONE,
    TASK_CARD_SECTION_IN_PROGRESS,
    TASK_CARD_SECTION_UP_NEXT,
    TASK_CARD_TITLE_GLYPH,
    TASK_CARD_TITLE_HEADING,
    TASK_LIST_TITLE,
)

logger = logging.getLogger(__name__)

# Text-display budgets.  Discord caps message content at 2000 characters and a
# Components V2 message carries a total text budget across all of its
# components, so each block is clipped below that with a visible ellipsis
# rather than letting the whole send fail.
_ELLIPSIS = "\u2026"
_OVERFLOW_PREFIX = "+"
_OVERFLOW_SUFFIX = " more"

# Budget reserved for the per-section label ("**Up Next**") that heads a block.
_SECTION_LABEL_RESERVE = 32

# Statuses that terminate a task without completing it.  These are counted in
# the denominator (they are not cancelled) but never render a row: showing an
# abandoned task under "Done" would misreport it as finished, and the bar is a
# completion ratio.
_ABANDONED_STATUSES = frozenset({"escalated"})
_CANCELLED_STATUS = "cancelled"

# Status to section.  ``closing`` and ``paused`` sit with the in-progress
# group because the task has been started and is still on the board; only
# ``pending`` is genuinely not yet begun.
_SECTION_BY_STATUS: Dict[str, str] = {
    "in_progress": TASK_CARD_SECTION_IN_PROGRESS,
    "closing": TASK_CARD_SECTION_IN_PROGRESS,
    "paused": TASK_CARD_SECTION_IN_PROGRESS,
    "pending": TASK_CARD_SECTION_UP_NEXT,
    "completed": TASK_CARD_SECTION_DONE,
}

_GLYPH_BY_SECTION: Dict[str, str] = {
    TASK_CARD_SECTION_IN_PROGRESS: TASK_CARD_GLYPH_ACTIVE,
    TASK_CARD_SECTION_UP_NEXT: TASK_CARD_GLYPH_PENDING,
    TASK_CARD_SECTION_DONE: TASK_CARD_GLYPH_DONE,
}

_FADED_SECTIONS = frozenset({TASK_CARD_SECTION_DONE})

_SECTION_ORDER = (
    TASK_CARD_SECTION_IN_PROGRESS,
    TASK_CARD_SECTION_UP_NEXT,
    TASK_CARD_SECTION_DONE,
)


def format_elapsed(seconds: float) -> str:
    """Render an elapsed duration the way the card's footer shows it.

    The card always reports a coarse, human-readable figure — an exact second
    count is noise on a card that is regenerated on demand.
    """
    total = max(0.0, float(seconds))
    if total < 60:
        return "<1m elapsed"
    minutes = int(total // 60)
    if minutes < 60:
        return f"{minutes}m elapsed"
    hours = int(total // 3600)
    remainder_minutes = int((total % 3600) // 60)
    if hours < 24:
        return f"{hours}h {remainder_minutes}m elapsed"
    days = int(total // 86400)
    remainder_hours = int((total % 86400) // 3600)
    return f"{days}d {remainder_hours}h elapsed"


def _row(glyph: str, task: str, faded: bool) -> str:
    """Render one task row, applying the completed-task fade when asked."""
    if faded:
        return f"{TASK_CARD_FADE_PREFIX}{glyph}  {task}"
    return f"{glyph}  {task}"


def _clip_section(label: str, lines: Sequence[str]) -> str:
    """Join a section's label and rows, clipping to the text budget.

    The clip happens at a line boundary so a task row is never cut mid-word
    without the ellipsis marking it.
    """
    budget = TASK_CARD_MAX_CHARS - _SECTION_LABEL_RESERVE
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    if len(kept) < len(lines):
        logger.debug(
            "Task card section %r clipped from %d to %d rows",
            label,
            len(lines),
            len(kept),
        )
        kept.append(_ELLIPSIS)
    return "\n".join([f"**{label}**", *kept])


def _section_lines(glyph: str, tasks: Sequence[str], faded: bool) -> List[str]:
    """Render a section's rows, collapsing overflow into a ``+N more`` line."""
    visible = tasks[:TASK_CARD_MAX_ROWS_PER_SECTION]
    lines = [_row(glyph, task, faded) for task in visible]
    overflow = len(tasks) - len(visible)
    if overflow > 0:
        lines.append(f"{_OVERFLOW_PREFIX}{overflow}{_OVERFLOW_SUFFIX}")
    return lines


def _footer_text(filled: int, total: int, elapsed_label: str) -> str:
    """Bar and ratio in the footer line, elapsed time kept, subtext sized.

    The percentage is deliberately absent: the bar already encodes the same
    proportion segment-for-segment.  The bar is wrapped in backticks so the
    segments keep their even glyph column, exactly as the approved card shows.
    """
    bar = TASK_CARD_BAR_FILLED * filled + TASK_CARD_BAR_EMPTY * max(0, total - filled)
    return (
        f"{TASK_CARD_FOOTER_SUBTEXT_PREFIX}"
        f"{TASK_CARD_FOOTER_CODE_FENCE}{bar}{TASK_CARD_FOOTER_CODE_FENCE}"
        f"   {filled} / {total} "
        f"{TASK_CARD_FOOTER_SEPARATOR} {elapsed_label}"
    )


def build_task_card(
    items: Sequence[Dict[str, Any]],
    elapsed_label: str,
) -> Optional[Dict[str, Any]]:
    """Build the task-card spec for ``items``, or None when nothing renders.

    Returns a dict with ``heading`` (the title text display's content),
    ``sections`` (one entry per non-empty section, each with a ``label`` and its
    rendered ``text``) and ``footer``.  The caller (platform adapter) is
    responsible for turning it into a native message.
    """
    sections: Dict[str, List[str]] = {name: [] for name in _SECTION_ORDER}
    completed = 0
    denominator = 0

    for item in items:
        status = str(item.get("status", "") or "")
        if status == _CANCELLED_STATUS:
            # Cancelled work is not part of the progress ratio and never
            # renders — it would otherwise read as pending work forever.
            continue
        denominator += 1
        if status == "completed":
            completed += 1
        if status in _ABANDONED_STATUSES:
            continue
        section_name = _SECTION_BY_STATUS.get(status)
        if section_name is None:
            # Unknown status: surface it under the queue rather than dropping
            # the task silently, so it can be corrected.
            logger.debug("Task card: unrecognised task status %r", status)
            section_name = TASK_CARD_SECTION_UP_NEXT
        task_text = str(item.get("content", "") or "").strip()
        if not task_text:
            continue
        sections[section_name].append(task_text)

    total = denominator
    if total == 0:
        return None

    rendered: List[Dict[str, Any]] = []
    for name in _SECTION_ORDER:
        tasks = sections[name]
        if not tasks:
            continue
        lines = _section_lines(_GLYPH_BY_SECTION[name], tasks, name in _FADED_SECTIONS)
        rendered.append({"label": name, "text": _clip_section(name, lines)})

    if not rendered:
        return None

    return {
        "heading": f"{TASK_CARD_TITLE_HEADING} {TASK_CARD_TITLE_GLYPH} {TASK_LIST_TITLE}",
        "sections": rendered,
        "footer": _footer_text(completed, total, elapsed_label),
    }
