"""Platform-neutral approval presentation spec.

The ``approval_presentation`` plugin hook lets a plugin own the look and
behaviour of the interactive approval prompt without patching a platform
adapter. This module defines the single spec type every renderer consumes and
the validator that makes the hook fail-closed: a malformed or partial return
never reaches a renderer — it is dropped, and the caller falls back to core's
default presentation.

Kept stdlib-only so both the CLI approval path (``tools/approval.py``) and
platform adapters can import it with no import cycle.
"""

from __future__ import annotations

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class PresentationField(TypedDict, total=False):
    """One embed field: the label, its value, and whether it renders inline."""

    name: str
    value: str
    inline: bool


class PresentationAction(TypedDict, total=False):
    """One declarative approval control.

    ``id`` is the only field core interprets: the built-in ids (``once``,
    ``session``, ``always``, ``deny``) map onto core's own approval choices,
    and any other id is forwarded to the ``approval_action`` hook so the plugin
    decides what the click means. Core never needs to understand a plugin's id.
    """

    id: str
    label: str
    style: str


class PresentationPrePrompt(TypedDict, total=False):
    """A message delivered before the approval prompt (e.g. a batch's diffs)."""

    text: str
    attachments: list[str]


class PresentationSpec(TypedDict, total=False):
    """The presentation spec a plugin supplies for one approval prompt.

    Every key is optional; renderers apply what is present and fall back to
    their defaults for what is not. ``title`` / ``description`` / ``color`` /
    ``fields`` / ``timeout`` mirror the keys the Discord adapter has always
    consumed, so an existing hook return stays valid.
    """

    title: str
    description: str
    body: str
    color: int
    fields: list[PresentationField]
    attachments: list[str]
    pre_prompt: PresentationPrePrompt
    actions: list[PresentationAction]
    timeout: int
    reason: str


# The string keys copied verbatim when they hold a non-empty string.
_STRING_KEYS = ("title", "description", "body", "reason")

# Button styles core knows how to render; an unknown style falls back to the
# neutral default rather than dropping the action.
ACTION_STYLES = ("green", "grey", "blurple", "red")
DEFAULT_ACTION_STYLE = "grey"


def _clean_fields(raw: object) -> list[PresentationField]:
    """Return well-formed fields from ``raw``; drop malformed entries."""
    if not isinstance(raw, list):
        return []
    fields: list[PresentationField] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        field: PresentationField = {"name": name}
        value = item.get("value")
        if isinstance(value, str):
            field["value"] = value
        field["inline"] = bool(item.get("inline", False))
        fields.append(field)
    return fields


def _clean_actions(raw: object) -> list[PresentationAction]:
    """Return well-formed actions from ``raw``; drop entries without an id."""
    if not isinstance(raw, list):
        return []
    actions: list[PresentationAction] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action_id = item.get("id")
        if not isinstance(action_id, str) or not action_id:
            continue
        action: PresentationAction = {"id": action_id}
        label = item.get("label")
        if isinstance(label, str) and label:
            action["label"] = label
        style = item.get("style")
        action["style"] = (
            style if isinstance(style, str) and style in ACTION_STYLES
            else DEFAULT_ACTION_STYLE
        )
        actions.append(action)
    return actions


def _clean_attachments(raw: object) -> list[str]:
    """Return the string paths from ``raw``; drop non-strings and blanks."""
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _clean_pre_prompt(raw: object) -> PresentationPrePrompt | None:
    """Return a well-formed pre-prompt, or ``None`` when nothing usable exists."""
    if not isinstance(raw, dict):
        return None
    pre_prompt: PresentationPrePrompt = {}
    text = raw.get("text")
    if isinstance(text, str) and text:
        pre_prompt["text"] = text
    attachments = _clean_attachments(raw.get("attachments"))
    if attachments:
        pre_prompt["attachments"] = attachments
    return pre_prompt or None


def validate_presentation_spec(raw: object) -> PresentationSpec | None:
    """Validate a hook return into a presentation spec, or ``None``.

    Fail-closed: a non-dict return, or one carrying no usable key, yields
    ``None`` so the caller renders core's default. Malformed entries inside an
    otherwise valid spec are dropped rather than aborting the whole spec, so a
    single bad attachment path never costs the plugin its prompt.
    """
    if not isinstance(raw, dict):
        return None

    spec: PresentationSpec = {}
    for key in _STRING_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            spec[key] = value

    color = raw.get("color")
    if isinstance(color, int) and not isinstance(color, bool):
        spec["color"] = color

    fields = _clean_fields(raw.get("fields"))
    if fields:
        spec["fields"] = fields

    attachments = _clean_attachments(raw.get("attachments"))
    if attachments:
        spec["attachments"] = attachments

    pre_prompt = _clean_pre_prompt(raw.get("pre_prompt"))
    if pre_prompt is not None:
        spec["pre_prompt"] = pre_prompt

    actions = _clean_actions(raw.get("actions"))
    if actions:
        spec["actions"] = actions

    timeout = raw.get("timeout")
    if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0:
        spec["timeout"] = timeout

    return spec or None
