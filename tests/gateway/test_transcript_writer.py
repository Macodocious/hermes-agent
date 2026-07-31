"""Unit tests for the gateway transcript writer."""
import json
from datetime import datetime, timezone
from pathlib import Path

from gateway.transcript import ThreadTranscriptWriter


def test_write_serializes_datetime_timestamp(tmp_path):
    """Records carrying datetime timestamps must serialize to JSONL.

    Regression: inbound events pass ``event.timestamp`` (a datetime) to
    ``write()``; ``json.dumps`` without ``default=str`` raised
    ``TypeError: Object of type datetime is not JSON serializable``,
    crashing the pre_gateway_dispatch transcript hook and silently
    stopping transcript recording.
    """
    writer = ThreadTranscriptWriter(Path(tmp_path))
    writer.write(
        role="user",
        user_id="u1",
        message_id="m1",
        thread_id="t1",
        chat_id="t1",
        platform="discord",
        content="hello",
        timestamp=datetime(2026, 7, 31, 13, 40, 31, tzinfo=timezone.utc),
    )

    lines = (
        (Path(tmp_path) / "t1.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["role"] == "user"
    assert record["timestamp"] == "2026-07-31 13:40:31+00:00"
