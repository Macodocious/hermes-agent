"""Test Discord slash command sync skip logic rejects stale success claims.

Regression guard for the bug where an interrupted sync left a new
fingerprint paired with a stale ``last_success_at``, causing the next
startup to skip re-syncing and permanently miss newly added commands
(e.g. ``/tasks`` never appearing after a mid-sync restart).
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    if sys.modules.get("discord") is None:
        discord_mod = MagicMock()
        discord_mod.Intents.default.return_value = MagicMock()
        sys.modules["discord"] = discord_mod
        sys.modules["discord.ext"] = MagicMock()
        sys.modules["discord.ext.commands"] = MagicMock()


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def adapter():
    """Create a Discord adapter with mocked Discord client."""
    _ensure_discord_mock()
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._client.tree = MagicMock()
    adapter._client.http = AsyncMock()
    adapter._client.application_id = "test_app_id"
    return adapter


def _state_entry(fingerprint: str, last_success_at: float, last_attempt_at: float) -> dict:
    return {
        "test_app_id": {
            "fingerprint": fingerprint,
            "last_success_at": last_success_at,
            "last_attempt_at": last_attempt_at,
        }
    }


def test_skip_reason_accepts_success_newer_than_attempt(adapter):
    """A completed sync (success >= attempt) with matching fingerprint skips."""
    state = _state_entry("fp-1", last_success_at=200.0, last_attempt_at=100.0)
    with patch.object(adapter, "_read_command_sync_state", return_value=state):
        reason = adapter._command_sync_skip_reason("test_app_id", "fp-1")
    assert reason == "same slash-command fingerprint already synced"


def test_skip_reason_rejects_stale_success(adapter):
    """An interrupted sync (success < attempt) must NOT skip re-syncing.

    This is the exact corruption observed in production: the July 30
    success timestamp survived a mid-sync restart, so the new fingerprint
    was treated as already synced and /tasks never registered.
    """
    state = _state_entry("fp-2", last_success_at=100.0, last_attempt_at=200.0)
    with patch.object(adapter, "_read_command_sync_state", return_value=state):
        reason = adapter._command_sync_skip_reason("test_app_id", "fp-2")
    assert reason is None


def test_skip_reason_rejects_missing_success(adapter):
    """No success record at all must never skip."""
    state = _state_entry("fp-3", last_success_at=0.0, last_attempt_at=200.0)
    with patch.object(adapter, "_read_command_sync_state", return_value=state):
        reason = adapter._command_sync_skip_reason("test_app_id", "fp-3")
    assert reason is None


def test_skip_reason_rejects_fingerprint_mismatch(adapter):
    """A different fingerprint must never skip, even with a fresh success."""
    state = _state_entry("fp-old", last_success_at=200.0, last_attempt_at=100.0)
    with patch.object(adapter, "_read_command_sync_state", return_value=state):
        reason = adapter._command_sync_skip_reason("test_app_id", "fp-new")
    assert reason is None


def test_record_attempt_clears_stale_success(adapter):
    """Recording a new attempt must invalidate any prior success claim."""
    prior = _state_entry("fp-old", last_success_at=100.0, last_attempt_at=50.0)
    written = {}

    def _fake_write(state):
        written.update(state)

    with patch.object(adapter, "_read_command_sync_state", return_value=prior), patch.object(
        adapter, "_write_command_sync_state", side_effect=_fake_write
    ):
        adapter._record_command_sync_attempt("test_app_id", "fp-new")

    entry = written["test_app_id"]
    assert entry["fingerprint"] == "fp-new"
    assert entry["last_success_at"] == 0
    assert entry["summary"] is None
    assert entry["last_attempt_at"] > 0


def test_record_rate_limit_clears_stale_success(adapter):
    """A rate-limited attempt must also invalidate any prior success claim."""
    prior = _state_entry("fp-old", last_success_at=100.0, last_attempt_at=50.0)
    written = {}

    def _fake_write(state):
        written.update(state)

    with patch.object(adapter, "_read_command_sync_state", return_value=prior), patch.object(
        adapter, "_write_command_sync_state", side_effect=_fake_write
    ):
        adapter._record_command_sync_rate_limit("test_app_id", "fp-new", retry_after=30.0)

    entry = written["test_app_id"]
    assert entry["fingerprint"] == "fp-new"
    assert entry["last_success_at"] == 0
    assert entry["summary"] is None
    assert entry["retry_after_until"] > 0
