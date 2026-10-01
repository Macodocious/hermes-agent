"""Tests for the async post-implementation review (the review-phase replacement).

Covers the whole contract: the finding-line parse, the recursion guard that
stops a review from launching a review, the append of findings to the end of
the todo list, and the launch itself — that a finalized task queues a review
and that the review never blocks the finalization it was launched from.
"""

from types import SimpleNamespace
import queue

from agent import task_manager, task_review


# ── Finding parse ──────────────────────────────────────────────────────


def test_parse_findings_reads_the_agreed_line_format() -> None:
    stdout = (
        "Some prose the model wrote.\n"
        "REVIEW_FINDING: high | the retry loop never bounds its attempts in agent/x.py\n"
        "REVIEW_FINDING: low | naming: `tmp` shadows the module-level helper\n"
    )
    findings = task_review._parse_findings(stdout)
    assert len(findings) == 2
    assert findings[0]["severity"] == "high"
    assert "never bounds its attempts" in findings[0]["content"]
    assert findings[1]["severity"] == "low"


def test_parse_findings_drops_lines_that_are_not_findings() -> None:
    stdout = (
        "I reviewed the change and it looks reasonable.\n"
        "REVIEW_FINDING: no separator or body here\n"
        "REVIEW_FINDING: high |\n"
    )
    # A line that does not match the agreed format is never guessed at.
    assert task_review._parse_findings(stdout) == []


def test_parse_findings_caps_the_number_returned() -> None:
    stdout = "\n".join(
        f"REVIEW_FINDING: low | finding number {n}"
        for n in range(task_review.REVIEW_MAX_ITEMS * 3)
    )
    findings = task_review._parse_findings(stdout)
    assert len(findings) == task_review.REVIEW_MAX_ITEMS


# ── Recursion guard ────────────────────────────────────────────────────


def test_recursion_guard_refuses_to_launch_from_a_review_process(monkeypatch) -> None:
    monkeypatch.setenv(task_review.REVIEW_ENV_FLAG, "1")
    assert task_review.is_review_process() is True
    launched = task_review.launch(
        session_id="s", item_id="1", task="t", evidence=""
    )
    assert launched is False


def test_spawn_returns_none_inside_a_review_process(monkeypatch) -> None:
    monkeypatch.setenv(task_review.REVIEW_ENV_FLAG, "1")
    # The guard fires before any binary lookup or subprocess spawn.
    assert task_review._spawn("prompt") is None


# ── Launch from finalization ───────────────────────────────────────────


def test_done_verdict_launches_the_review(monkeypatch, tmp_path) -> None:
    """The judge's done verdict finalizes the task and launches the review."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    launched: list[dict] = []

    def fake_launch(**kwargs):
        launched.append(kwargs)
        return True

    monkeypatch.setattr(task_review, "launch", fake_launch)

    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "1", "content": "Build the thing", "status": "pending"}])
    agent = SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
        _turn_file_mutation_paths={"/repo/agent/x.py"},
    )
    store.transition("begin", "1")
    store.transition("close", "1")
    before = task_manager._open_item_ids(store)

    nudge = task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "complete", "bound_task_id": "1"}
    )

    assert store.read()[0]["status"] == "completed"
    assert len(launched) == 1
    assert launched[0]["item_id"] == "1"
    assert launched[0]["session_id"] == "test-session"
    assert launched[0]["task"] == "Build the thing"
    assert nudge is None or isinstance(nudge, str)
    assert before  # sanity: the task was open before the verdict


def test_read_only_task_skips_the_review(monkeypatch, tmp_path) -> None:
    """A task whose turn changed no file finalizes with no review launched."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    launched: list[dict] = []
    monkeypatch.setattr(
        task_review, "launch", lambda **kwargs: launched.append(kwargs) or True
    )

    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "1", "content": "Research the thing", "status": "pending"}])
    agent = SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
        _turn_file_mutation_paths=set(),  # read-only turn
    )
    store.transition("begin", "1")
    store.transition("close", "1")

    task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "complete", "bound_task_id": "1"}
    )

    assert store.read()[0]["status"] == "completed"  # finalization still stands
    assert launched == []  # but no review is spent on read-only work


def test_missing_mutation_record_fails_toward_not_reviewing(monkeypatch, tmp_path) -> None:
    """An absent mutation record is non-mutative — the check never raises."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    launched: list[dict] = []
    monkeypatch.setattr(
        task_review, "launch", lambda **kwargs: launched.append(kwargs) or True
    )

    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "1", "content": "Build the thing", "status": "pending"}])
    # A path that finalizes without a turn context: no mutation attribute.
    agent = SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
    )
    store.transition("begin", "1")
    store.transition("close", "1")

    task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "complete", "bound_task_id": "1"}
    )

    assert store.read()[0]["status"] == "completed"
    assert launched == []


def test_review_launch_never_blocks_finalization(monkeypatch, tmp_path) -> None:
    """A launch that raises must not stop the task finalizing."""
    monkeypatch.setattr(task_manager, "_lifecycle_config", lambda: {"enabled": True})
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    def exploding_launch(**kwargs):
        raise RuntimeError("review subsystem unavailable")

    monkeypatch.setattr(task_review, "launch", exploding_launch)

    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "1", "content": "Build the thing", "status": "pending"}])
    agent = SimpleNamespace(
        _todo_store=store,
        session_id="test-session",
        _task_lifecycle_action_issued=False,
        _task_lifecycle_nudge="",
        _turn_file_mutation_paths={"/repo/agent/x.py"},
    )
    store.transition("begin", "1")
    store.transition("close", "1")

    task_manager.observe_verdict(
        agent, {"verdict": "done", "reason": "complete", "bound_task_id": "1"}
    )

    # The finalization stands even though the review could not launch.
    assert store.read()[0]["status"] == "completed"


# ── Append to the todo list ────────────────────────────────────────────


def test_append_findings_appends_rows_to_the_end(monkeypatch) -> None:
    from tools.todo_tool import TodoStore

    store = TodoStore()
    store.write([{"id": "1", "content": "Existing work", "status": "pending"}])

    monkeypatch.setattr("hermes_cli.tasks.load_todo", lambda session_id: store)
    monkeypatch.setattr("hermes_cli.tasks.save_todo", lambda session_id, s: None)

    appended = task_review._append_findings(
        "test-session",
        "1",
        [
            {"severity": "high", "content": "the guard is not wired in"},
            {"severity": "low", "content": "rename `tmp`"},
        ],
    )

    assert appended == 2
    rows = store.read()
    assert len(rows) == 3
    # Existing row keeps priority at the head; findings land at the end.
    assert rows[0]["id"] == "1"
    assert rows[1]["source"] == "review"
    assert "Review finding (high):" in rows[1]["content"]
    assert rows[2]["source"] == "review"
    assert "Review finding (low):" in rows[2]["content"]


def test_append_findings_with_nothing_to_report_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr(
        "hermes_cli.tasks.load_todo",
        lambda session_id: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    # No findings means no store access at all.
    assert task_review._append_findings("s", "1", []) == 0
