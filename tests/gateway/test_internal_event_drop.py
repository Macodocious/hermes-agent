"""A queued internal event must never start a run of its own.

An internal event (watch/process notification, picker turn, auto-resume) that
arrives while a turn is running is merged into ``_pending_messages`` like any
other message.  Before this fix the drain popped it and started a fresh run,
so the user got an extra assistant message after the turn's final one.

The drop lives where the queue is consumed, so it covers every producer and
both drains (the adapter's in-band/late drains and the gateway's turn-end
drain, which consumes through ``get_pending_message``).
"""

import asyncio
import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    return _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)


def _make_event(text="hi", chat_id="42", internal=False):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
        internal=internal,
    )


def _sk(chat_id="42"):
    return build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    )


@pytest.mark.asyncio
async def test_in_band_drain_drops_internal_event(caplog):
    """An internal event queued behind a running turn starts no second run."""
    adapter = _make_adapter()
    sk = _sk()
    runs: list[str] = []

    async def handler(event):
        runs.append(event.text)
        if len(runs) == 1:
            adapter._pending_messages[sk] = _make_event(
                text="[IMPORTANT: Background process proc_1 completed normally]",
                internal=True,
            )
        return "ok"

    adapter._message_handler = handler

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
        await adapter.handle_message(_make_event(text="M0"))
        for _ in range(400):
            if sk not in adapter._active_sessions:
                break
            await asyncio.sleep(0.01)
        await adapter.cancel_background_tasks()

    assert runs == ["M0"], f"internal event started a run: {runs!r}"
    assert any(
        "Dropping queued internal event" in record.message for record in caplog.records
    ), "the drop was not recorded"


@pytest.mark.asyncio
async def test_in_band_drain_still_runs_user_event():
    """A user message queued behind a running turn still runs."""
    adapter = _make_adapter()
    sk = _sk()
    runs: list[str] = []

    async def handler(event):
        runs.append(event.text)
        if len(runs) == 1:
            adapter._pending_messages[sk] = _make_event(text="M1")
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="M0"))
    for _ in range(400):
        if len(runs) >= 2 and sk not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)
    await adapter.cancel_background_tasks()

    assert runs == ["M0", "M1"], f"user follow-up did not run: {runs!r}"


def test_get_pending_message_drops_internal_event(caplog):
    """The gateway's turn-end drain consumes through this pop."""
    adapter = _make_adapter()
    sk = _sk()
    adapter._pending_messages[sk] = _make_event(text="notice", internal=True)

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
        assert adapter.get_pending_message(sk) is None
    assert sk not in adapter._pending_messages
    assert any(
        "Dropping queued internal event" in record.message for record in caplog.records
    )


def test_get_pending_message_returns_user_event():
    adapter = _make_adapter()
    sk = _sk()
    event = _make_event(text="user text")
    adapter._pending_messages[sk] = event

    assert adapter.get_pending_message(sk) is event


def test_user_message_replaces_queued_internal_event():
    """A user message must not inherit the internal marker and be dropped."""
    pending: dict = {}
    sk = _sk()
    merge_pending_message_event(pending, sk, _make_event(text="notice", internal=True))
    merge_pending_message_event(pending, sk, _make_event(text="user text"), merge_text=True)

    assert pending[sk].text == "user text"
    assert pending[sk].internal is False


def test_internal_event_does_not_displace_user_message():
    pending: dict = {}
    sk = _sk()
    merge_pending_message_event(pending, sk, _make_event(text="user text"))
    merge_pending_message_event(pending, sk, _make_event(text="notice", internal=True))

    assert pending[sk].text == "user text"
    assert pending[sk].internal is False
