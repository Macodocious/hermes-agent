"""Tests for #60432: cron jobs must not be silently invisible to gateway
shutdown, and a job whose tool subprocess got killed by shutdown must
never be reported as a successful run.

Covers the cron/scheduler.py primitives directly:
  - get_running_job_ids() -- thread-safe snapshot the gateway drain reads
  - mark_running_jobs_interrupted() -- called by the gateway right after
    it force-kills tool subprocesses
  - the interrupted-flag race guard in run_one_job(), which must win over
    the job's own thread finishing normally with a plausible-looking
    result AFTER its tool was already killed out from under it
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Every test starts from a clean slate and leaves one behind, since
    these sets are module-level globals shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._job_agents.clear()
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._job_agents.clear()


class TestGetRunningJobIds:
    def test_empty_when_nothing_running(self):
        import cron.scheduler as sched

        assert sched.get_running_job_ids() == frozenset()

    def test_reflects_in_flight_jobs(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        sched._running_job_ids.add("job-2")

        result = sched.get_running_job_ids()

        assert result == frozenset({"job-1", "job-2"})

    def test_snapshot_is_immutable_and_independent(self):
        """Mutating _running_job_ids after the call must not change the
        already-returned snapshot -- callers (the gateway drain loop) rely
        on this to safely count in a tight polling loop."""
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        snapshot = sched.get_running_job_ids()
        sched._running_job_ids.add("job-2")

        assert snapshot == frozenset({"job-1"})


class TestMarkRunningJobsInterrupted:
    def test_no_op_when_nothing_running(self):
        import cron.scheduler as sched

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == []
        mock_mark.assert_not_called()

    def test_marks_every_in_flight_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (final-cleanup)")

        assert sorted(marked) == ["job-1", "job-2"]
        assert mock_mark.call_count == 2
        called_ids = {c.args[0] for c in mock_mark.call_args_list}
        assert called_ids == {"job-1", "job-2"}
        for c in mock_mark.call_args_list:
            # success must be False -- an interrupted run is never "ok".
            assert c.args[1] is False
            assert "gateway shutdown" in c.args[2]

    def test_sets_interrupted_flag_for_consumption_by_run_one_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")

        with patch("cron.scheduler.mark_job_run"):
            sched.mark_running_jobs_interrupted("shutdown")

        assert "job-1" in sched._interrupted_job_ids

    def test_one_job_marking_failure_does_not_block_the_others(self):
        """mark_job_run raising for one job (e.g. a jobs.json write race)
        must not prevent the rest from being marked -- this runs during
        shutdown, there's no retry window."""
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        def _side_effect(job_id, success, reason, **kwargs):
            if job_id == "job-1":
                raise OSError("disk full")

        with patch("cron.scheduler.mark_job_run", side_effect=_side_effect):
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == ["job-2"]


class TestIsInterrupted:
    """Peek-only check used at the delivery gate -- must NOT clear the
    flag, unlike _consume_interrupted_flag."""

    def test_false_when_not_marked(self):
        import cron.scheduler as sched

        assert sched._is_interrupted("job-1") is False

    def test_true_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._is_interrupted("job-1") is True

    def test_does_not_clear_the_flag(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        sched._is_interrupted("job-1")

        # Still set -- the later, authoritative check before mark_job_run
        # must still see it.
        assert "job-1" in sched._interrupted_job_ids
        assert sched._is_interrupted("job-1") is True


class TestConsumeInterruptedFlag:
    def test_false_when_not_marked(self):
        import cron.scheduler as sched

        assert sched._consume_interrupted_flag("job-1") is False

    def test_true_and_clears_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._consume_interrupted_flag("job-1") is True
        # Consumed -- a second check (e.g. a later, unrelated fire of the
        # same recurring job ID) must not still read as interrupted.
        assert sched._consume_interrupted_flag("job-1") is False


class TestRunOneJobHonoursInterruptedFlag:
    """run_one_job() must not let a job's own completion overwrite a
    status the shutdown path already wrote for the same run."""

    def _make_job(self, job_id="job-1"):
        return {"id": job_id, "name": "test job", "prompt": "do work"}

    def test_success_path_skipped_when_interrupted(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        # The would-be "success" write must NOT happen -- the shutdown
        # path already wrote the authoritative interrupted status.
        mock_mark.assert_not_called()
        # Flag is consumed so a later, unrelated fire of the same job ID
        # isn't permanently silenced.
        assert job["id"] not in sched._interrupted_job_ids

    def test_interrupted_job_delivers_failure_summary_not_raw_response(self):
        """The status-write guard alone isn't enough: delivery happens
        BEFORE mark_job_run in run_one_job's own flow, so a job that kept
        running post-kill and produced a plausible-looking final_response
        must not have that response sent to the user just because the
        eventual status write gets suppressed. Interrupted jobs must route
        through the same failure-summary delivery path a real failure
        would."""
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "a plausible final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch(
                 "cron.scheduler._summarize_cron_failure_for_delivery",
                 return_value="This run was interrupted.",
             ) as mock_summarize, \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None) as mock_deliver, \
             patch("cron.scheduler.mark_job_run"):
            result = sched.run_one_job(job)

        assert result is True
        mock_summarize.assert_called_once()
        # The summarizer's error argument must mention the interruption,
        # not be silently None / the agent's own (possibly absent) error.
        assert "interrupt" in mock_summarize.call_args.args[1].lower()
        delivered_content = mock_deliver.call_args.args[1]
        assert delivered_content == "This run was interrupted."
        assert "plausible final response" not in delivered_content

    def test_success_path_writes_normally_when_not_interrupted(self):
        """Control case: the guard must not swallow ordinary, un-interrupted
        completions -- only ones the shutdown path explicitly flagged."""
        import cron.scheduler as sched

        job = self._make_job()

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[0] == job["id"]
        assert mock_mark.call_args.args[1] is True  # success

    def test_exception_path_also_honours_interrupted_flag(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler.run_job", side_effect=RuntimeError("boom")), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is False
        mock_mark.assert_not_called()


class TestMarkRunningJobsInterruptedIdempotent:
    """The drain-start mark (new) plus the final-cleanup mark (existing)
    both fire during one shutdown. Re-marking a job already marked by the
    first call would double-increment ``completed`` and double-advance
    ``next_run_at`` inside ``mark_job_run``, so the second call must be a
    no-op for jobs it already saw."""

    def test_second_call_does_not_remark_already_marked_jobs(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            first = sched.mark_running_jobs_interrupted("drain started")
            second = sched.mark_running_jobs_interrupted("final cleanup")

        # First call marks everything in flight; second call marks nothing.
        assert sorted(first) == ["job-1", "job-2"]
        assert second == []
        assert mock_mark.call_count == 2  # only the first call's writes

    def test_jobs_dispatched_after_drain_start_are_still_captured(self):
        """A tick that dispatches a job mid-drain (before the final sweep)
        races the first mark. The second call must catch it — it wasn't in
        the first call's snapshot, so it is not protected by idempotency."""
        import cron.scheduler as sched

        sched._running_job_ids.add("early-job")

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            sched.mark_running_jobs_interrupted("drain started")
            sched._running_job_ids.add("late-job")  # dispatched mid-drain
            second = sched.mark_running_jobs_interrupted("final cleanup")

        assert second == ["late-job"]
        called_ids = {c.args[0] for c in mock_mark.call_args_list}
        assert called_ids == {"early-job", "late-job"}


class TestJobAgentRegistry:
    """Drain-start wedged-agent interrupt: the gateway needs the live agent
    handle per in-flight job. Exercises register/unregister/get/
    interrupt_running_job_agents."""

    def test_register_then_get_returns_agent(self):
        import cron.scheduler as sched

        agent = MagicMock()
        sched.register_cron_agent("job-1", agent)

        assert sched.get_running_job_agents() == {"job-1": agent}

    def test_unregister_drops_agent(self):
        import cron.scheduler as sched

        sched.register_cron_agent("job-1", MagicMock())
        sched.unregister_cron_agent("job-1")

        assert sched.get_running_job_agents() == {}

    def test_unregister_unknown_id_is_safe(self):
        import cron.scheduler as sched

        sched.unregister_cron_agent("never-registered")  # must not raise

    def test_get_returns_independent_snapshot(self):
        import cron.scheduler as sched

        sched.register_cron_agent("job-1", MagicMock())
        snapshot = sched.get_running_job_agents()
        sched.register_cron_agent("job-2", MagicMock())

        assert snapshot == {"job-1": snapshot["job-1"]}

    def test_interrupt_hits_every_registered_agent(self):
        import cron.scheduler as sched

        agent_1 = MagicMock()
        agent_2 = MagicMock()
        sched.register_cron_agent("job-1", agent_1)
        sched.register_cron_agent("job-2", agent_2)

        interrupted = sched.interrupt_running_job_agents("drain started")

        assert sorted(interrupted) == ["job-1", "job-2"]
        agent_1.interrupt.assert_called_once_with("drain started")
        agent_2.interrupt.assert_called_once_with("drain started")

    def test_interrupt_never_raises_when_nothing_registered(self):
        import cron.scheduler as sched

        assert sched.interrupt_running_job_agents("drain started") == []

    def test_interrupt_one_failure_does_not_block_others(self):
        import cron.scheduler as sched

        agent_1 = MagicMock()
        agent_1.interrupt.side_effect = RuntimeError("boom")
        agent_2 = MagicMock()
        sched.register_cron_agent("job-1", agent_1)
        sched.register_cron_agent("job-2", agent_2)

        interrupted = sched.interrupt_running_job_agents("drain started")

        assert interrupted == ["job-2"]
        agent_2.interrupt.assert_called_once_with("drain started")
