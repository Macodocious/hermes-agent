"""Discord threads created in configured categories are auto-joined."""

import os
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _apply_yaml_config,
)


class _FakeThread:
    def __init__(self, thread_id, type_value, category_id, name="t"):
        self.id = thread_id
        self.type = SimpleNamespace(value=type_value)
        self.parent = SimpleNamespace(category=SimpleNamespace(id=category_id), id=999)
        self.name = name
        self.guild = SimpleNamespace(id=777)
        self.joined = False

    async def join(self):
        self.joined = True


class _FakeTracker:
    def __init__(self):
        self.marked = []

    def mark(self, thread_id):
        self.marked.append(thread_id)


@pytest.mark.asyncio
async def test_auto_join_thread_joins_and_marks_public_thread_in_configured_category(monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_JOIN_CATEGORIES", "111,222")
    thread = _FakeThread(thread_id=42, type_value=11, category_id="222")
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._threads = _FakeTracker()
    adapter.platform = SimpleNamespace(value="discord")

    await adapter._auto_join_thread(thread)

    assert thread.joined is True
    assert adapter._threads.marked == ["42"]


@pytest.mark.asyncio
async def test_auto_join_thread_skips_unconfigured_category(monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_JOIN_CATEGORIES", "111")
    thread = _FakeThread(thread_id=42, type_value=11, category_id="222")
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._threads = _FakeTracker()
    adapter.platform = SimpleNamespace(value="discord")

    await adapter._auto_join_thread(thread)

    assert thread.joined is False
    assert adapter._threads.marked == []


@pytest.mark.asyncio
async def test_auto_join_thread_skips_private_thread(monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_JOIN_CATEGORIES", "111")
    thread = _FakeThread(thread_id=42, type_value=13, category_id="111")
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._threads = _FakeTracker()
    adapter.platform = SimpleNamespace(value="discord")

    await adapter._auto_join_thread(thread)

    assert thread.joined is False
    assert adapter._threads.marked == []


def test_yaml_config_bridges_auto_join_categories_to_env(monkeypatch):
    monkeypatch.delenv("DISCORD_AUTO_JOIN_CATEGORIES", raising=False)

    _apply_yaml_config(
        {"discord": {"auto_join_categories": ["111", "222"]}},
        {"auto_join_categories": ["111", "222"]},
    )
    assert os.environ["DISCORD_AUTO_JOIN_CATEGORIES"] == "111,222"


@pytest.mark.asyncio
async def test_auto_join_thread_fires_on_thread_create_hook(monkeypatch):
    monkeypatch.setenv("DISCORD_AUTO_JOIN_CATEGORIES", "111,222")
    thread = _FakeThread(thread_id=42, type_value=11, category_id="222", name="discussion")
    adapter = object.__new__(DiscordAdapter)
    adapter.config = SimpleNamespace(extra={})
    adapter._threads = _FakeTracker()
    adapter.platform = SimpleNamespace(value="discord")

    captured = {}

    def _fake_invoke_hook(hook_name, **kwargs):
        captured["hook_name"] = hook_name
        captured.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)

    await adapter._auto_join_thread(thread)

    assert thread.joined is True
    assert adapter._threads.marked == ["42"]
    assert captured["hook_name"] == "on_thread_create"
    assert captured["thread_id"] == "42"
    assert captured["parent_channel_id"] == "999"
    assert captured["guild_id"] == "777"
    assert captured["thread_name"] == "discussion"
