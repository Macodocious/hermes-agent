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


def test_clarify_question_and_answer_are_recorded(tmp_path, monkeypatch):
    """A clarify prompt and its answer each get a record, tagged with origin.

    Neither reaches the delivered-reply path — the prompt is a dedicated send
    and a button answer is not a message at all — so without these records a
    clarify exchange is absent from the transcript.
    """
    from gateway import transcript

    monkeypatch.setattr(transcript, "_writer", ThreadTranscriptWriter(Path(tmp_path)))

    transcript.record_clarify_question(
        clarify_id="c1",
        question="Which store holds it?",
        choices=["state.db", "jsonl"],
        thread_id="t9",
        chat_id="c9",
        platform="discord",
    )
    transcript.record_clarify_answer(
        clarify_id="c1",
        answer="state.db",
        source=transcript.CLARIFY_SOURCE_BUTTON,
        thread_id="t9",
        chat_id="c9",
        platform="discord",
    )

    lines = (Path(tmp_path) / "t9.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

    assert [record["role"] for record in records] == ["clarify", "clarify"]
    assert [record["kind"] for record in records] == ["question", "answer"]
    assert records[0]["choices"] == ["state.db", "jsonl"]
    assert records[1]["content"] == "state.db"
    assert records[1]["source"] == "button"
    assert records[0]["clarify_id"] == records[1]["clarify_id"] == "c1"
