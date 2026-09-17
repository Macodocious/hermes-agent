#!/usr/bin/env python3
"""Tests for the rich /tasks card: builder plus the handler's send branch.

The builder tests pin the approved design's properties — heading glyph, title
case section names, non-emoji row glyphs, the blockquote fade on completed
rows, one progress source for bar and ratio, and the footer's exact shape.

The handler tests pin the two delivery paths: an adapter that can render the
card sends it and suppresses the text reply; an adapter that cannot (or whose
send fails) falls back to the plain-text list, so /tasks never comes back
empty.

The adapter tests pin the wire structure: a Components V2 layout message whose
container matches the approved card component-for-component, because that
structure — not merely the text — is what the user approved.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.task_card import build_task_card, format_elapsed
from hermes_constants import (
    TASK_CARD_BAR_MIN_SEGMENTS,
    TASK_CARD_MAX_CHARS,
    TASK_LIST_TITLE,
    TASK_CARD_TITLE_GLYPH,
)
from hermes_state import AsyncSessionDB
from tools.todo_tool import TodoStore

# Component types on the wire, as the approved card carries them.
_TYPE_CONTAINER = 17
_TYPE_TEXT_DISPLAY = 10
_TYPE_SEPARATOR = 14

# The accent rail's packed RGB integer.
_ACCENT_BLURPLE = 5793266


def _make_source(platform: Platform = Platform.DISCORD) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/tasks", *, platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(platform), message_id="m1")


def _make_session_entry(platform: Platform = Platform.DISCORD) -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source(platform)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
        total_tokens=0,
    )


def _make_store() -> TodoStore:
    store = TodoStore()
    store.write(
        [
            {"id": "1", "content": "Done thing", "status": "completed"},
            {"id": "2", "content": "Working thing", "status": "in_progress"},
            {"id": "3", "content": "Next thing", "status": "pending"},
        ]
    )
    return store


def _sections(card) -> dict:
    """Section label -> text, for the card's rendered sections."""
    return {s["label"]: s["text"] for s in card["sections"]}


def _row_lines(card, label: str) -> list:
    """A section's rows, excluding its bold label line."""
    return _sections(card)[label].splitlines()[1:]


def _make_runner(session_entry: SessionEntry, adapter, *, platform: Platform = Platform.DISCORD):
    """Build a GatewayRunner whose adapter is exactly ``adapter``.

    ``adapter`` is passed in rather than mocked here because the whole point
    of these tests is which methods the adapter does and does not expose.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {platform: adapter}
    runner._running_agents = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner._session_db = AsyncSessionDB(MagicMock())
    runner._session_db._db.get_session.return_value = None
    runner._profile_adapters = {}
    return runner


def _attach_store(runner, platform: Platform = Platform.DISCORD) -> None:
    running_agent = MagicMock()
    running_agent._todo_store = _make_store()
    runner._running_agents[build_session_key(_make_source(platform))] = running_agent


class _CardAdapter:
    """Adapter double that can render the card."""

    def __init__(self, *, success: bool = True):
        self.calls = []
        self._success = success

    async def send_task_card(self, chat_id, card, metadata=None):
        self.calls.append({"chat_id": chat_id, "card": card, "metadata": metadata})
        if self._success:
            return SendResult(success=True, message_id="m-card")
        return SendResult(success=False, error="boom")


# ─── Builder ────────────────────────────────────────────────────────────────


def test_card_heading_carries_the_clipboard_glyph():
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    assert TASK_CARD_TITLE_GLYPH in card["heading"]
    assert TASK_LIST_TITLE in card["heading"]
    # The approved card renders its title as a markdown heading, not an embed
    # title — the heading marker is part of the approved structure.
    assert card["heading"].startswith("### ")


def test_card_sections_are_title_case_and_glyph_free():
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    assert [s["label"] for s in card["sections"]] == ["In Progress", "Up Next", "Done"]
    for section in card["sections"]:
        label_line = section["text"].splitlines()[0]
        assert label_line == f"**{section['label']}**"
        assert label_line.isascii()


def test_card_row_glyphs_avoid_the_emoji_property():
    """U+25B6 carries the Emoji property and Discord swaps in a colour emoji."""
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    body = "\n".join(s["text"] for s in card["sections"])
    assert "\u25b6" not in body
    assert "\u25ba" in body


def test_completed_rows_use_the_blockquote_fade():
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    assert _row_lines(card, "Done") == ["> \u2713  Done thing"]


def test_active_and_queued_rows_are_not_faded_or_bolded():
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    assert _row_lines(card, "In Progress") == ["\u25ba  Working thing"]
    assert _row_lines(card, "Up Next") == ["\u25cb  Next thing"]
    rows = "\n".join(
        line
        for section in card["sections"]
        for line in section["text"].splitlines()[1:]
    )
    assert "**" not in rows
    assert "-#" not in rows


def test_footer_has_bar_ratio_and_elapsed_but_no_percentage():
    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    assert card["footer"] == "-# `\u2588\u2588\u2588\u2588\u2591\u2591\u2591\u2591\u2591\u2591\u2591\u2591`   1 / 3 \u00b7 3h 12m elapsed"
    assert "%" not in card["footer"]


def test_bar_keeps_a_minimum_width_at_a_one_task_list():
    """One task must still draw a full track, not a single stray segment."""
    card = build_task_card([{"content": "only", "status": "pending"}], "1m elapsed")
    bar = card["footer"].split("`")[1]
    assert len(bar) == TASK_CARD_BAR_MIN_SEGMENTS
    assert bar == "\u2591" * TASK_CARD_BAR_MIN_SEGMENTS
    assert "0 / 1" in card["footer"]


def test_bar_fill_tracks_the_ratio_at_every_count():
    """Bar width is stable and its fill is the ratio's proportion."""
    for filled, total in ((0, 1), (1, 3), (5, 8), (7, 12), (12, 12), (1, 40)):
        items = [
            {"content": f"task {i}", "status": "completed" if i < filled else "pending"}
            for i in range(total)
        ]
        bar = build_task_card(items, "1m elapsed")["footer"].split("`")[1]
        width = max(total, TASK_CARD_BAR_MIN_SEGMENTS)
        expected_fill = round(filled / total * width)
        assert len(bar) == width, f"{filled}/{total}: width {len(bar)} != {width}"
        assert bar.count("\u2588") == expected_fill, f"{filled}/{total}: fill mismatch"


def test_bar_and_ratio_share_one_source():
    """The approved twelve-task card is unchanged by the minimum width."""
    items = [
        {"content": f"task {i}", "status": "completed" if i < 7 else "pending"}
        for i in range(12)
    ]
    card = build_task_card(items, "1m elapsed")
    assert card["footer"].split("`")[1] == "\u2588" * 7 + "\u2591" * 5
    assert "7 / 12" in card["footer"]


def test_cancelled_tasks_leave_the_denominator_and_the_rows():
    items = [
        {"content": "done", "status": "completed"},
        {"content": "live", "status": "in_progress"},
        {"content": "dropped", "status": "cancelled"},
    ]
    card = build_task_card(items, "1m elapsed")
    body = "\n".join(s["text"] for s in card["sections"])
    assert "dropped" not in body
    assert "1 / 2" in card["footer"]


def test_escalated_counts_in_the_denominator_but_renders_no_row():
    items = [
        {"content": "ok", "status": "completed"},
        {"content": "stuck", "status": "escalated"},
    ]
    card = build_task_card(items, "1m elapsed")
    assert [s["label"] for s in card["sections"]] == ["Done"]
    assert "stuck" not in card["sections"][0]["text"]
    assert "1 / 2" in card["footer"]


def test_empty_and_all_cancelled_lists_render_no_card():
    assert build_task_card([], "1m elapsed") is None
    assert build_task_card([{"content": "x", "status": "cancelled"}], "1m elapsed") is None


def test_section_overflow_collapses_into_one_line():
    items = [{"content": f"task {i}", "status": "pending"} for i in range(20)]
    card = build_task_card(items, "1m elapsed")
    rows = _row_lines(card, "Up Next")
    assert rows[-1] == "+8 more"
    assert len(rows) == 13


def test_section_text_stays_within_the_display_budget():
    items = [{"content": "x" * 900, "status": "pending"} for _ in range(4)]
    card = build_task_card(items, "1m elapsed")
    for section in card["sections"]:
        assert len(section["text"]) <= TASK_CARD_MAX_CHARS


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "<1m elapsed"),
        (30, "<1m elapsed"),
        (300, "5m elapsed"),
        (11520, "3h 12m elapsed"),
        (90000, "1d 1h elapsed"),
    ],
)
def test_format_elapsed(seconds, expected):
    assert format_elapsed(seconds) == expected


# ─── Handler delivery paths ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capable_adapter_receives_the_card_and_text_is_suppressed():
    session_entry = _make_session_entry()
    adapter = _CardAdapter()
    runner = _make_runner(session_entry, adapter)
    _attach_store(runner)

    result = await runner._handle_task_command(_make_event())

    assert result == ""
    assert len(adapter.calls) == 1
    card = adapter.calls[0]["card"]
    assert TASK_CARD_TITLE_GLYPH in card["heading"]
    assert [s["label"] for s in card["sections"]] == ["In Progress", "Up Next", "Done"]
    assert adapter.calls[0]["chat_id"] == "c1"


@pytest.mark.asyncio
async def test_adapter_without_the_capability_falls_back_to_text():
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry, SimpleNamespace())
    _attach_store(runner)

    result = await runner._handle_task_command(_make_event())

    assert result.splitlines()[0] == "── **Current Tasks** ───────"
    assert "- [>] Working thing" in result


@pytest.mark.asyncio
async def test_failed_card_send_falls_back_to_text():
    session_entry = _make_session_entry()
    adapter = _CardAdapter(success=False)
    runner = _make_runner(session_entry, adapter)
    _attach_store(runner)

    result = await runner._handle_task_command(_make_event())

    assert len(adapter.calls) == 1
    assert "- [>] Working thing" in result


@pytest.mark.asyncio
async def test_mock_adapter_without_the_method_does_not_auto_succeed():
    """A bare MagicMock auto-creates attributes; it must not count as capable."""
    session_entry = _make_session_entry()
    runner = _make_runner(session_entry, MagicMock())
    _attach_store(runner)

    result = await runner._handle_task_command(_make_event())

    assert "- [>] Working thing" in result


# ─── Adapter send ───────────────────────────────────────────────────────────


class _FakeDiscordChannel:
    def __init__(self):
        self.sent = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(id=7777)


def _make_discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    return adapter


def _sent_components(channel) -> list:
    """The serialized component tree of the single sent message."""
    return channel.sent[0]["view"].to_components()


@pytest.mark.asyncio
async def test_send_task_card_renders_a_components_v2_container():
    adapter = _make_discord_adapter()
    channel = _FakeDiscordChannel()
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(),
    )

    card = build_task_card(_make_store().read(), "3h 12m elapsed")
    result = await adapter.send_task_card("4242", card)

    assert result.success is True
    assert result.message_id == "7777"
    assert len(channel.sent) == 1

    components = _sent_components(channel)
    assert len(components) == 1

    container = components[0]
    assert container["type"] == _TYPE_CONTAINER
    assert container["accent_color"] == _ACCENT_BLURPLE

    # The approved card's structure: heading, then one separator + text display
    # per section, then a divider separator + the footer text display.
    children = container["components"]
    assert [child["type"] for child in children] == [
        _TYPE_TEXT_DISPLAY,
        _TYPE_SEPARATOR,
        _TYPE_TEXT_DISPLAY,
        _TYPE_SEPARATOR,
        _TYPE_TEXT_DISPLAY,
        _TYPE_SEPARATOR,
        _TYPE_TEXT_DISPLAY,
        _TYPE_SEPARATOR,
        _TYPE_TEXT_DISPLAY,
    ]
    assert children[0]["content"] == card["heading"]
    assert children[-1]["content"] == card["footer"]

    # Only the footer's separator draws a visible divider; the section
    # separators are pure spacing, exactly as the approved card renders them.
    separators = [child for child in children if child["type"] == _TYPE_SEPARATOR]
    assert [s["divider"] for s in separators] == [False, False, False, True]


@pytest.mark.asyncio
async def test_send_task_card_preserves_the_fade_prefix_in_section_text():
    adapter = _make_discord_adapter()
    channel = _FakeDiscordChannel()
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: channel,
        fetch_channel=AsyncMock(),
    )

    card = build_task_card(_make_store().read(), "1m elapsed")
    await adapter.send_task_card("4242", card)

    texts = [
        child["content"]
        for child in _sent_components(channel)[0]["components"]
        if child["type"] == _TYPE_TEXT_DISPLAY
    ]
    done_text = next(text for text in texts if text.startswith("**Done**"))
    assert done_text.splitlines()[1] == "> \u2713  Done thing"


@pytest.mark.asyncio
async def test_send_task_card_targets_the_thread_when_metadata_carries_one():
    adapter = _make_discord_adapter()
    channel = _FakeDiscordChannel()
    seen = []

    def get_channel(channel_id):
        seen.append(channel_id)
        return channel

    adapter._client = SimpleNamespace(get_channel=get_channel, fetch_channel=AsyncMock())

    card = build_task_card(_make_store().read(), "1m elapsed")
    await adapter.send_task_card("4242", card, metadata={"thread_id": "999"})

    assert seen == [999]


@pytest.mark.asyncio
async def test_send_task_card_fails_closed_without_a_client():
    adapter = _make_discord_adapter()
    adapter._client = None

    card = build_task_card(_make_store().read(), "1m elapsed")
    result = await adapter.send_task_card("4242", card)

    assert result.success is False
    assert result.error


@pytest.mark.asyncio
async def test_send_task_card_rejects_an_empty_card():
    adapter = _make_discord_adapter()
    adapter._client = SimpleNamespace(get_channel=lambda _id: _FakeDiscordChannel())

    result = await adapter.send_task_card("4242", {"sections": []})

    assert result.success is False
