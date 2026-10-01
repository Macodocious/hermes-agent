"""Async post-implementation review — the reviewer that replaces the review phase.

When the judge decides a task is done, it launches this instance: a separate,
asynchronous, read-only pass that checks the implementation's adherence to the
approved plan and specifications.
Its findings are appended to the end of the ``todo`` list so the agent picks
them up on the next turn.

Why a separate process: a full quality pass has to read the implementation,
the plan and the specs and then disagree with the work where it falls short.
A bare text completion cannot contradict a claim it cannot inspect, so the
reviewer runs as a read-only ``hermes`` subprocess (the same containment the
probe-runner verifier uses): ``REVIEW_TOOLSET`` is the entire ``-t`` argument,
so core resolves exactly read_file/search_files/skill_view and no write,
patch, terminal or code-execution tool can resolve at all.

Everything here is deterministic plumbing — the worker, the queue, the spawn
and the parse. The judgement is the subprocess's.

Severity: the reviewer assigns each finding its own severity and reports it;
this module carries that label through verbatim. No severity ladder is defined
here, deliberately — the levels were never specified, and inventing them would
put this code's vocabulary in the user's mouth.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Subprocess contract ────────────────────────────────────────────────────
REVIEW_TOOLSET = "read"
REVIEW_BINARY_NAME = "hermes"
# A duration, so deliberately not a power of two. The reviewer reads a task's
# implementation, its plan and its specs before judging — the same read-bound
# shape as the probe-runner verifier, whose measured runs reached 82.6s with
# one over 120s. Sized from that measurement, not from a guess.
REVIEW_TIMEOUT_SECONDS = 300
REVIEW_STDOUT_MAX_CHARS = 65536

# ── Queue ──────────────────────────────────────────────────────────────────
# Reviews run one at a time: a burst of finalized tasks spawns one read-only
# subprocess at a time instead of N simultaneously.
REVIEW_MAX_CONCURRENT = 1
REVIEW_QUEUE_MAXSIZE = 1024
REVIEW_DRAIN_POLL_SECONDS = 0.25

# Cap on findings appended to the todo list from one review. The todo list is
# a planning aid the model re-reads after every compression event, so a
# review that returns a hundred findings must not inflate it.
REVIEW_MAX_ITEMS = 16

# Marked on the spawned child's environment. The child is a full hermes
# process, so it fires the plugins_loaded hook on startup and would re-enter
# the very finalization that launched it; the guard makes the launch a no-op
# inside a review process.
REVIEW_ENV_FLAG = "HERMES_TASK_REVIEW"

# The findings cross a process boundary, so each is a line whose format both
# sides agree in advance — never prose to be interpreted.
FINDING_SENTINEL_PREFIX = "REVIEW_FINDING:"
FINDING_FIELD_SEPARATOR = "|"

# Provenance tag for reviewer-authored todo rows: code-owned, never
# model-authorable.
REVIEW_SOURCE = "review"

_PROMPT_TEMPLATE = (
    "You are the post-implementation reviewer. A task was just finalized as "
    "done. Review the implementation in this repository and report where it "
    "does not match the approved plan and specifications.\n\n"
    "Inspect the implementation directly. Test results are never evidence, "
    "and never delegate to test runs.\n\n"
    "Report every finding on its own line, in exactly this format:\n"
    "{sentinel} <severity> {separator} <how the implementation departs from "
    "the approved plan and specifications>\n\n"
    "Give each finding the severity you judge it to carry. Put nothing else "
    "on a finding line. If the implementation matches, output no finding "
    "lines at all.\n\n"
    "Task: {task}\n"
    "{evidence}"
)

_worker_lock = threading.Lock()
_work_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=REVIEW_QUEUE_MAXSIZE)
_worker_started = False


def is_review_process() -> bool:
    """True when this process IS a spawned review (the recursion guard).

    The reviewer is a full hermes process, so it fires the same startup hooks
    as the parent. Without this guard, finalizing inside the child would
    launch another review, which would finalize and launch another — an
    unbounded process tree. The guard makes the launch site a no-op.
    """
    return str(os.environ.get(REVIEW_ENV_FLAG) or "").strip() == "1"


def _build_prompt(*, task: str, evidence: str) -> str:
    """The reviewer's instruction. The format is the process-boundary contract."""
    return _PROMPT_TEMPLATE.format(
        sentinel=FINDING_SENTINEL_PREFIX,
        separator=FINDING_FIELD_SEPARATOR,
        task=task or "(no description)",
        evidence=evidence or "",
    )


def _spawn(prompt: str) -> Optional[str]:
    """Run one review subprocess and return its stdout, or None on failure.

    The child is read-only by toolset construction, not by instruction.
    Failure is logged at error level — a review that never ran must not look
    like a review that found nothing.
    """
    if is_review_process():
        # Recursion guard: never spawn a reviewer from a reviewer.
        return None
    binary = shutil.which(REVIEW_BINARY_NAME)
    if not binary:
        logger.error(
            "task_review: %s not found on PATH — review not run",
            REVIEW_BINARY_NAME,
        )
        return None
    # -z takes the prompt as its value, so the flags must precede it.
    argv = [binary, "-t", REVIEW_TOOLSET, "-z", prompt]
    env = dict(os.environ)
    env[REVIEW_ENV_FLAG] = "1"
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=REVIEW_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "task_review: review subprocess timed out after %ss",
            REVIEW_TIMEOUT_SECONDS,
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("task_review: review subprocess failed: %s", exc)
        return None
    if result.returncode != 0:
        logger.error(
            "task_review: review subprocess exited %s: %s",
            result.returncode,
            (result.stderr or "").strip()[:500],
        )
        return None
    return (result.stdout or "")[:REVIEW_STDOUT_MAX_CHARS]


def _parse_findings(stdout: str) -> List[Dict[str, str]]:
    """Parse the reviewer's finding lines into ``{severity, content}`` rows.

    A line that does not match the agreed format is not a finding and is
    dropped — the reviewer's prose is never guessed at.
    """
    findings: List[Dict[str, str]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith(FINDING_SENTINEL_PREFIX):
            continue
        body = line[len(FINDING_SENTINEL_PREFIX):].strip()
        severity, separator, content = body.partition(FINDING_FIELD_SEPARATOR)
        severity = severity.strip()
        content = content.strip()
        if not separator or not content:
            continue
        findings.append({"severity": severity or "unspecified", "content": content})
        if len(findings) >= REVIEW_MAX_ITEMS:
            break
    return findings


def _append_findings(session_id: str, item_id: str, findings: List[Dict[str, str]]) -> int:
    """Append the review's findings to the end of the session's todo list.

    Appending is the whole contract: the findings land after the existing
    rows so the agent reaches them once the current plan work is done. Rows
    are tagged as reviewer-authored so they are never mistaken for the
    agent's own plan items. Returns the number appended (0 on any failure),
    never raising — this runs on a worker thread.
    """
    if not findings:
        return 0
    try:
        from hermes_cli.tasks import load_todo, save_todo

        store = load_todo(session_id)
        if store is None:
            logger.error(
                "task_review: no todo store for session %s — %s finding(s) dropped",
                session_id,
                len(findings),
            )
            return 0
        # Merge mode drops id-less rows, so each finding carries the next
        # sequential id (mirrors TodoStore._next_item_id).
        numeric_ids = [
            int(row["id"]) for row in store.read() if str(row.get("id", "")).isdigit()
        ]
        next_id = (max(numeric_ids) + 1) if numeric_ids else 1
        rows = []
        for offset, finding in enumerate(findings):
            rows.append(
                {
                    "id": str(next_id + offset),
                    "content": (
                        f"Review finding ({finding['severity']}): {finding['content']}"
                    ),
                    "status": "pending",
                    "source": REVIEW_SOURCE,
                }
            )
        store.write(rows, merge=True)
        save_todo(session_id, store)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "task_review: appending findings for task %s failed: %s", item_id, exc
        )
        return 0
    logger.info(
        "task_review: appended %s finding(s) for task %s to session %s",
        len(findings),
        item_id,
        session_id,
    )
    return len(findings)


def _run_one(work: Dict[str, Any]) -> None:
    """Run one queued review and append its findings."""
    prompt = _build_prompt(
        task=str(work.get("task") or ""),
        evidence=str(work.get("evidence") or ""),
    )
    stdout = _spawn(prompt)
    if stdout is None:
        return
    findings = _parse_findings(stdout)
    if not findings:
        logger.info(
            "task_review: review of task %s reported no findings", work.get("item_id")
        )
        return
    _append_findings(str(work.get("session_id") or ""), str(work.get("item_id") or ""), findings)


def _worker_loop() -> None:
    """Drain the review queue serially, one subprocess at a time."""
    while True:
        work = _work_queue.get()
        try:
            _run_one(work)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("task_review: worker failed on %s: %s", work.get("item_id"), exc)
        finally:
            _work_queue.task_done()


def _ensure_worker() -> None:
    """Start the single worker thread on first use."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        for _ in range(REVIEW_MAX_CONCURRENT):
            thread = threading.Thread(
                target=_worker_loop, name="task-review-worker", daemon=True
            )
            thread.start()
        _worker_started = True


def launch(
    *,
    session_id: str,
    item_id: str,
    task: str,
    evidence: str = "",
) -> bool:
    """Queue an async review of a just-finalized task.

    Returns True when the review was queued. Never blocks on the review: the
    caller is the finalization path, and the whole point is that the review
    runs asynchronously. A full queue (never in practice) drops the review
    with an error log rather than stalling the verdict path.
    """
    if is_review_process():
        # Recursion guard: a review process never launches another.
        return False
    if not session_id:
        return False
    _ensure_worker()
    try:
        _work_queue.put_nowait(
            {
                "session_id": session_id,
                "item_id": item_id,
                "task": task,
                "evidence": evidence,
            }
        )
    except queue.Full:  # pragma: no cover - defensive
        logger.error(
            "task_review: queue full — review of task %s dropped", item_id
        )
        return False
    return True


def drain(timeout: Optional[float] = None) -> bool:
    """Block until the queue is empty (used by tests and shutdown paths).

    ``queue.join()`` is unbounded, so poll the unfinished count to honour the
    caller's timeout. Returns True when drained, False on timeout.
    """
    import time

    deadline = None if timeout is None else time.monotonic() + timeout
    while _work_queue.unfinished_tasks:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(REVIEW_DRAIN_POLL_SECONDS)
    return True
