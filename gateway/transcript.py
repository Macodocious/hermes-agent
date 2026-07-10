"""Per-thread conversation transcript persistence.

Writes append-only JSONL transcript files for each message thread.
One file per thread at ``<directory>/<thread_id>.jsonl``.

Inbound messages (from users) are captured via ``pre_gateway_dispatch``.
Outbound messages (assistant responses) are captured via ``post_gateway_delivery``.
Both use real platform message IDs (e.g. Discord snowflakes).

The writer is safe to use from multiple concurrent tasks: each append
uses ``os.open(O_APPEND)`` and ``os.fsync()``. A half-written line is
invalid JSON and is skipped by the reader.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def install_transcript_hooks(manager: Any, config: Any) -> None:
    """Register transcript persistence hooks when enabled in *config*.

    Called by :meth:`GatewayRunner.__init__` when ``transcripts.enabled`` is
    ``True``.  Hooks are appended directly to the :class:`PluginManager`'s
    internal dict so no synthetic manifest is required.
    """
    _tx_cfg = getattr(config, "transcripts", None)
    if _tx_cfg is None or not getattr(_tx_cfg, "enabled", False):
        return
    _dir = getattr(_tx_cfg, "directory", None)
    if _dir is None:
        _dir = Path.home() / ".hermes" / "transcripts"
    writer = ThreadTranscriptWriter(Path(_dir))

    manager._hooks.setdefault("pre_gateway_dispatch", []).append(
        lambda **kw: _on_inbound(writer, **kw)
    )
    manager._hooks.setdefault("post_gateway_delivery", []).append(
        lambda **kw: _on_outbound(writer, **kw)
    )
    logger.info("Transcript hooks installed for %s", _dir)


def _on_inbound(writer: "ThreadTranscriptWriter", *, event, **kw: Any) -> None:
    """Handler for ``pre_gateway_dispatch`` — writes user messages."""
    source = getattr(event, "source", None)
    if source is None:
        return
    writer.write(
        role="user",
        user_id=getattr(source, "user_id", None),
        message_id=getattr(event, "message_id", None),
        thread_id=getattr(source, "thread_id", None),
        chat_id=getattr(source, "chat_id", None),
        platform=getattr(getattr(source, "platform", None), "value", None),
        content=getattr(event, "text", None),
        timestamp=getattr(event, "timestamp", None),
    )


def _on_outbound(writer: "ThreadTranscriptWriter", *, event, result, content, **kw: Any) -> None:
    """Handler for ``post_gateway_delivery`` — writes assistant messages."""
    source = getattr(event, "source", None)
    if source is None:
        return
    writer.write(
        role="assistant",
        user_id=getattr(source, "user_id", None),
        message_id=getattr(result, "message_id", None),
        thread_id=getattr(source, "thread_id", None),
        chat_id=getattr(source, "chat_id", None),
        platform=getattr(getattr(source, "platform", None), "value", None),
        content=content,
        timestamp=time.time(),
    )


class ThreadTranscriptWriter:
    """Thread-safe append-only JSONL transcript writer.

    One file per thread_id.  Each line is a single JSON object
    representing one message turn (user or assistant).
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    # ── public API ───────────────────────────────────────────────

    def write(
        self,
        *,
        role: str,
        user_id: Optional[str],
        message_id: Optional[str],
        thread_id: Optional[str],
        chat_id: Optional[str],
        platform: Optional[str],
        content: Optional[str],
        timestamp: Optional[float] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Append a single message record to its thread file.

        All fields are optional so the writer degrades gracefully
        when platform metadata is missing.
        """
        record: Dict[str, Any] = {}
        if role:
            record["role"] = role
        if user_id is not None:
            record["user_id"] = str(user_id)
        if message_id is not None:
            record["message_id"] = str(message_id)
        if thread_id is not None:
            record["thread_id"] = str(thread_id)
        if chat_id is not None:
            record["chat_id"] = str(chat_id)
        if platform:
            record["platform"] = platform
        if content is not None:
            record["content"] = content
        record["timestamp"] = timestamp if timestamp is not None else time.time()
        if model:
            record["model"] = model
        if session_id is not None:
            record["session_id"] = session_id

        _target_id = thread_id or chat_id or "unknown"
        _file = self._directory / f"{_target_id}.jsonl"

        line = json.dumps(record, ensure_ascii=False) + "\n"
        _atomic_append(_file, line)


# ── helpers ────────────────────────────────────────────────────

def _atomic_append(path: Path, text: str) -> None:
    """Append *text* to *path* using O_APPEND and fsync.

    Creates the file if it does not exist.  On any error the
    exception is logged but not raised so transcript writes can
    never break message delivery.
    """
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, text.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception as exc:
        logger.warning("Transcript append failed for %s: %s", path, exc)


# ── convenience: build a writer from config ────────────────────

def build_transcript_writer(config: Any) -> Optional[ThreadTranscriptWriter]:
    """Return a writer when transcripts are enabled in *config*, else None.

    Looks for ``transcripts.enabled`` and ``transcripts.directory``
    on the GatewayConfig object.
    """
    if config is None:
        return None
    _tx_cfg = getattr(config, "transcripts", None)
    if _tx_cfg is None:
        return None
    _enabled = getattr(_tx_cfg, "enabled", False)
    if not _enabled:
        return None
    _dir = getattr(_tx_cfg, "directory", None)
    if _dir is None:
        _dir = Path.home() / ".hermes" / "transcripts"
    return ThreadTranscriptWriter(Path(_dir))
