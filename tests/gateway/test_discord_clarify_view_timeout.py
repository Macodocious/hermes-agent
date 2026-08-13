"""Tests for the clarify button view timeout wiring.

``ClarifyChoiceView`` must honor ``agent.clarify_timeout`` — the same
deadline the agent thread blocks on in ``wait_for_response`` — so the
buttons stay live exactly as long as the prompt is answerable. The view
timeout is deliberately NOT clamped to the approval-family
``[30, 900]`` range: the clarify prompt is sent as a regular channel
message, so every button click is a fresh interaction with a fresh token
(Discord's ~15-minute token expiry applies to the initial interaction
response and to ephemeral messages, not to component buttons on normal
messages).

The invariant asserted throughout: ``view.timeout == get_clarify_timeout()``
for any configured value — the buttons and the agent's blocking wait can
never drift apart.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Repo root importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Triggers the shared discord mock from tests/gateway/conftest.py before
# importing the production module.
from plugins.platforms.discord.adapter import (  # noqa: E402
    ClarifyChoiceView,
    _DISCORD_PROMPT_TIMEOUT_DEFAULT,
    _read_clarify_view_timeout,
)


@pytest.fixture
def clarify_timeout(monkeypatch):
    """Stub ``tools.clarify_gateway.get_clarify_timeout``.

    Returns a setter so each test controls the configured value (or makes
    the read explode to exercise the fallback).
    """
    import tools.clarify_gateway

    def _set(value):
        if value is None:
            def _boom():
                raise RuntimeError("config read failed")
            monkeypatch.setattr(tools.clarify_gateway, "get_clarify_timeout", _boom)
        else:
            monkeypatch.setattr(
                tools.clarify_gateway, "get_clarify_timeout", lambda: value
            )

    return _set


def _make_view():
    return ClarifyChoiceView(
        choices=["apple", "banana"],
        clarify_id="cidT",
        allowed_user_ids={"42"},
    )


class TestReadClarifyViewTimeout:
    """The reader must pass the configured value through unclamped."""

    def test_returns_configured_clarify_timeout(self, clarify_timeout):
        clarify_timeout(43200)  # 12h — the user's configured value
        assert _read_clarify_view_timeout() == 43200

    def test_no_upper_clamp_beyond_approval_range(self, clarify_timeout):
        """Values above the approval-family 900s ceiling must pass through —
        regular-message buttons are not bound by the 15-min interaction
        token window."""
        clarify_timeout(14400)  # 4h — same as approvals.gateway_timeout
        assert _read_clarify_view_timeout() == 14400

    def test_small_values_pass_through(self, clarify_timeout):
        """A short clarify_timeout must not be inflated — the buttons should
        die exactly when the agent thread unblocks."""
        clarify_timeout(45)
        assert _read_clarify_view_timeout() == 45

    def test_default_when_config_absent(self, clarify_timeout):
        clarify_timeout(3600)  # get_clarify_timeout's own default
        assert _read_clarify_view_timeout() == 3600

    def test_config_read_explosion_falls_back_to_historical_default(
        self, clarify_timeout
    ):
        """A crashing config read must not bring down view construction."""
        clarify_timeout(None)
        assert _read_clarify_view_timeout() == _DISCORD_PROMPT_TIMEOUT_DEFAULT


class TestClarifyChoiceViewTimeout:
    """The view must be constructed with exactly the clarify timeout."""

    def test_view_timeout_matches_clarify_timeout(self, clarify_timeout):
        clarify_timeout(43200)
        assert _make_view().timeout == 43200

    def test_view_timeout_matches_default(self, clarify_timeout):
        clarify_timeout(3600)
        assert _make_view().timeout == 3600

    def test_view_timeout_matches_short_value(self, clarify_timeout):
        clarify_timeout(45)
        assert _make_view().timeout == 45

    def test_view_timeout_falls_back_on_config_error(self, clarify_timeout):
        clarify_timeout(None)
        assert _make_view().timeout == _DISCORD_PROMPT_TIMEOUT_DEFAULT
