"""The thread auto-archive window is a setting, not a hardcoded day."""

import os
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import (
    DEFAULT_THREAD_AUTO_ARCHIVE_MINUTES,
    DiscordAdapter,
    _apply_yaml_config,
)


def _resolve(extra=None) -> int:
    """Call the resolver on a stand-in, since ``name`` is read-only on the adapter."""
    stub = SimpleNamespace(
        name="discord",
        config=SimpleNamespace(extra=dict(extra or {})),
    )
    return DiscordAdapter._discord_thread_auto_archive(stub)


def test_default_is_one_week(monkeypatch):
    monkeypatch.delenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", raising=False)
    assert DEFAULT_THREAD_AUTO_ARCHIVE_MINUTES == 10080
    assert _resolve() == 10080


def test_config_extra_wins_over_env(monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", "1440")
    assert _resolve({"thread_auto_archive_duration": 4320}) == 4320


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", "4320")
    assert _resolve() == 4320


@pytest.mark.parametrize("bad", ["900", "weekly", " "])
def test_unusable_values_keep_default(monkeypatch, bad):
    monkeypatch.delenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", raising=False)
    assert _resolve({"thread_auto_archive_duration": bad}) == 10080


def test_yaml_config_bridges_to_env(monkeypatch):
    monkeypatch.delenv("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", raising=False)
    try:
        _apply_yaml_config({}, {"thread_auto_archive_duration": 4320})
        assert os.environ["DISCORD_THREAD_AUTO_ARCHIVE_DURATION"] == "4320"
    finally:
        # The bridge writes to os.environ directly, so the value outlives
        # monkeypatch's teardown and would leak into other tests.
        os.environ.pop("DISCORD_THREAD_AUTO_ARCHIVE_DURATION", None)
