"""Tests for #60432: the gateway shutdown drain was structurally blind to
in-flight cron work. Cron jobs run through cron/scheduler.py's own thread
pool, entirely outside ``GatewayRunner._running_agents`` -- the dict every
other active-work check on this class reads. A shutdown (``/update``,
``/restart``, SIGUSR1 -- they all funnel through the same ``stop()``) could
report ``active_at_start=0`` and immediately kill tool subprocesses while a
cron job's terminal command was still running.

These tests cover the gateway side of the fix:
  - _active_cron_job_count() reads cron.scheduler's in-flight job set
  - _drain_active_agents() waits for cron work the same way it already
    waits for chat sessions
  - the final tool-subprocess kill marks any still-in-flight cron job
    interrupted
  - the drain-START mark + agent interrupt: as soon as the drain begins,
    in-flight cron jobs are marked interrupted and their live agents are
    asked to interrupt, so a run that finishes during the drain can no
    longer escape as a false "ok" and a wedged job cannot pin the drain.

See tests/cron/test_shutdown_interrupt.py for the cron-side primitives
this relies on (get_running_job_ids, mark_running_jobs_interrupted,
interrupt_running_job_agents)."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture(autouse=True)
def _reset_cron_running_set():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._job_agents.clear()
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    sched._job_agents.clear()


def _make_async_noop():
    async def _noop(*args, **kwargs):
        return None

    return _noop


class TestActiveCronJobCount:
    def test_zero_when_no_cron_jobs_running(self):
        runner, _adapter = make_restart_runner()
        assert runner._active_cron_job_count() == 0

    def test_reflects_cron_scheduler_state(self):
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("job-1")

        assert runner._active_cron_job_count() == 1

    def test_never_raises_if_cron_module_unavailable(self):
        """Best-effort: a broken/absent import must not take shutdown
        counting down with it."""
        runner, _adapter = make_restart_runner()

        with patch(
            "cron.scheduler.get_running_job_ids", side_effect=ImportError("boom")
        ):
            assert runner._active_cron_job_count() == 0


class TestDrainWaitsForCronWork:
    @pytest.mark.asyncio
    async def test_drain_returns_immediately_when_nothing_active(self):
        runner, _adapter = make_restart_runner()

        _snapshot, timed_out = await runner._drain_active_agents(5.0)

        assert timed_out is False

    @pytest.mark.asyncio
    async def test_drain_waits_for_in_flight_cron_job(self):
        """Before this fix, a cron-only workload made active_at_start=0
        and the drain returned instantly -- this is the exact repro from
        the issue (a `sleep 1800` cron job in flight during /update)."""
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("job-1")

        async def finish_job():
            await asyncio.sleep(0.12)
            sched._running_job_ids.discard("job-1")

        task = asyncio.create_task(finish_job())
        _snapshot, timed_out = await runner._drain_active_agents(2.0)
        await task

        assert timed_out is False, (
            "drain must wait for the cron job to finish, not report "
            "active_at_start=0 and return instantly"
        )

    @pytest.mark.asyncio
    async def test_drain_times_out_if_cron_job_outlives_the_window(self):
        import cron.scheduler as sched

        runner, _adapter = make_restart_runner()
        sched._running_job_ids.add("job-1")  # never removed within the window

        _snapshot, timed_out = await runner._drain_active_agents(0.1)

        assert timed_out is True

    @pytest.mark.asyncio
    async def test_drain_still_waits_for_chat_sessions_unchanged(self):
        """Regression guard: folding cron into the check must not break
        the pre-existing chat-session drain behavior."""
        runner, _adapter = make_restart_runner()
        runner._running_agents = {"session-1": MagicMock()}

        async def finish_agent():
            await asyncio.sleep(0.12)
            runner._running_agents.clear()

        task = asyncio.create_task(finish_agent())
        _snapshot, timed_out = await runner._drain_active_agents(2.0)
        await task

        assert timed_out is False


class TestKillToolSubprocessesMarksCronInterrupted:
    @pytest.mark.asyncio
    async def test_in_flight_cron_job_marked_interrupted_on_forced_kill(self, monkeypatch):
        import cron.scheduler as sched
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt
        import tools.browser_tool as _bt

        runner, adapter = make_restart_runner()
        runner._restart_drain_timeout = 0.01  # force the timeout path
        adapter.disconnect = _make_async_noop()

        sched._running_job_ids.add("job-1")

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        marked_calls = []
        real_mark = sched.mark_running_jobs_interrupted

        def _spy(reason):
            result = real_mark(reason)
            marked_calls.append((reason, result))
            return result

        monkeypatch.setattr(sched, "mark_running_jobs_interrupted", _spy)

        with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"):
            await runner.stop()

        assert marked_calls, "mark_running_jobs_interrupted was never called during shutdown"
        assert any(result == ["job-1"] for _reason, result in marked_calls)

    @pytest.mark.asyncio
    async def test_no_cron_jobs_running_is_a_silent_no_op(self, monkeypatch):
        """Graceful shutdown with nothing in flight must not spuriously
        mark or log anything cron-related."""
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt
        import tools.browser_tool as _bt

        runner, adapter = make_restart_runner()
        adapter.disconnect = _make_async_noop()

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 0)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            await runner.stop()

        mock_mark.assert_not_called()


class TestDrainStartMarksCronInterrupted:
    """The drain-start mark closes the false-"ok" race: a cron job that
    finishes its response DURING the drain (after the mark, before the
    final tool sweep) must still be reported interrupted — Friday's
    post-market run escaped as "ok" by ~0.4 s precisely because the mark
    only fired at the END of teardown."""

    @pytest.mark.asyncio
    async def test_in_flight_cron_job_marked_at_drain_start(self, monkeypatch):
        import cron.scheduler as sched
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt
        import tools.browser_tool as _bt

        runner, adapter = make_restart_runner()
        adapter.disconnect = _make_async_noop()

        sched._running_job_ids.add("job-1")
        sched.register_cron_agent("job-1", MagicMock())

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        # Spy on the mark so we can prove it happened at drain START —
        # before the final tool sweep, while the job is still running.
        marks = []
        real_mark = sched.mark_running_jobs_interrupted

        def _spy(reason):
            result = real_mark(reason)
            marks.append(reason)
            return result

        monkeypatch.setattr(sched, "mark_running_jobs_interrupted", _spy)
        interrupts = []
        real_interrupt = sched.interrupt_running_job_agents

        def _spy_interrupt(reason):
            result = real_interrupt(reason)
            interrupts.append(reason)
            return result

        monkeypatch.setattr(sched, "interrupt_running_job_agents", _spy_interrupt)

        with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"):
            await runner.stop()

        assert marks, "drain-start mark must fire"
        assert "drain began" in marks[0]
        assert interrupts, "drain-start agent interrupt must fire"
        assert "drain started" in interrupts[0]

    @pytest.mark.asyncio
    async def test_drain_start_mark_fires_while_job_still_running(self, monkeypatch):
        """The exact regression shape: the mark must happen at drain
        START, not after the job leaves _running_job_ids — a job that
        finishes during the drain (like Friday's ~0.4 s escape) must be
        caught."""
        import cron.scheduler as sched
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt
        import tools.browser_tool as _bt

        runner, agent = make_restart_runner()
        agent.disconnect = _make_async_noop()

        sched._running_job_ids.add("job-1")

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        marked_while_running = []

        def _spy(reason):
            # The mark fires at drain start — the job is still in flight.
            marked_while_running.append("job-1" in sched._running_job_ids)
            return ["job-1"]

        monkeypatch.setattr(sched, "mark_running_jobs_interrupted", _spy)

        with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"):
            await runner.stop()

        # The drain-start mark must fire while the job is STILL in
        # _running_job_ids. (The mark also fires on the later shutdown
        # phases in this test because the job never leaves the set here —
        # every one of those must ALSO see it still in flight; the job
        # only leaves the set when run_one_job's completion path runs.)
        assert marked_while_running and all(marked_while_running), (
            "the mark must fire while the job is still in _running_job_ids "
            "(drain start), not after it completed"
        )

    @pytest.mark.asyncio
    async def test_registered_agents_interrupted_at_drain_start(self, monkeypatch):
        """A wedged cron agent gets interrupted as the drain begins, not
        after a 600 s inactivity timeout."""
        import cron.scheduler as sched
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt
        import tools.browser_tool as _bt

        runner, agent = make_restart_runner()
        agent.disconnect = _make_async_noop()

        sched._running_job_ids.add("job-1")
        fake_agent = MagicMock()
        sched.register_cron_agent("job-1", fake_agent)

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"):
            await runner.stop()

        fake_agent.interrupt.assert_called_once()
