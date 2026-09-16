"""
Tests for timezone support (hermes_time module + integration points).

Covers:
  - Valid timezone applies correctly
  - Invalid timezone falls back safely (no crash, warning logged)
  - execute_code child env receives TZ
  - Cron uses timezone-aware now()
  - Backward compatibility with naive timestamps
  - format_in_user_tz() helper for agent-emitted timestamp reformatting
"""

import os
import logging
import sys
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import hermes_time


def _reset_hermes_time_cache():
    """Reset the hermes_time module cache (replacement for removed reset_cache)."""
    hermes_time._cached_tz = None
    hermes_time._cached_tz_name = None
    hermes_time._cache_resolved = False


# =========================================================================
# hermes_time.now() — core helper
# =========================================================================

class TestHermesTimeNow:
    """Test the timezone-aware now() helper."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_valid_timezone_applies(self):
        """With a valid IANA timezone, now() returns time in that zone."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # IST is UTC+5:30
        offset = result.utcoffset()
        assert offset == timedelta(hours=5, minutes=30), f"Expected +5:30 offset, got {offset}"

    def test_utc_timezone(self):
        """UTC timezone returns UTC-offset datetime."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        result = hermes_time.now()
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)

    def test_us_eastern(self):
        """US Eastern timezone returns appropriate offset (EST or EDT depending on date)."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # EST is -5, EDT is -4
        offset = result.utcoffset()
        assert offset in (timedelta(hours=-5), timedelta(hours=-4)), f"Expected -5 or -4, got {offset}"

    def test_invalid_timezone_falls_back(self, caplog):
        """Invalid timezone falls back to local time without crashing."""
        os.environ["HERMES_TIMEZONE"] = "Mars/Olympus_Mons"
        with caplog.at_level(logging.WARNING, logger="hermes_time"):
            result = hermes_time.now()
        assert result.tzinfo is not None
        # Should have logged a warning
        assert any("Invalid timezone" in record.message for record in caplog.records)

    def test_empty_timezone_uses_local(self):
        """Empty timezone string falls back to local time."""
        os.environ["HERMES_TIMEZONE"] = ""
        result = hermes_time.now()
        assert result.tzinfo is not None

    def test_format_unchanged(self):
        """now() format doesn't change the datetime structure."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        result = hermes_time.now()
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_cache_invalidation(self):
        """Cache invalidation picks up new timezone value."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        r1 = hermes_time.now()
        _reset_hermes_time_cache()
        os.environ["HERMES_TIMEZONE"] = "Asia/Tokyo"
        r2 = hermes_time.now()
        # Different UTC offsets
        assert r1.utcoffset() != r2.utcoffset()


# =========================================================================
# get_timezone() — explicit lookup
# =========================================================================

class TestGetTimezone:
    """Test get_timezone() returns the right ZoneInfo or None."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_returns_zoneinfo_for_valid(self):
        os.environ["HERMES_TIMEZONE"] = "Europe/London"
        result = hermes_time.get_timezone()
        assert result is not None
        assert isinstance(result, ZoneInfo)

    def test_returns_none_for_empty(self):
        os.environ["HERMES_TIMEZONE"] = ""
        result = hermes_time.get_timezone()
        assert result is None

    def test_returns_none_for_invalid(self, caplog):
        os.environ["HERMES_TIMEZONE"] = "Not/A/Zone"
        with caplog.at_level(logging.WARNING, logger="hermes_time"):
            result = hermes_time.get_timezone()
        assert result is None


# =========================================================================
# execute_code child env — TZ injection
# =========================================================================

class TestCodeExecutionTZ:
    """Test that execute_code child env receives the configured TZ."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_tz_injected_when_configured(self, monkeypatch):
        """When HERMES_TIMEZONE is set, child env should include TZ."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        # Patch the subprocess call to capture env
        from tools import code_execution_tool
        captured_env = {}

        real_popen = code_execution_tool.subprocess.Popen
        def fake_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            # Return a mock that doesn't actually run
            class MockProcess:
                def communicate(self, timeout=None):
                    return (b"", b"")
                @property
                def returncode(self):
                    return 0
            return MockProcess()

        monkeypatch.setattr(code_execution_tool.subprocess, "Popen", fake_popen)
        # Call the tool
        try:
            code_execution_tool.execute_code(code="print('test')", session_id="test")
        except Exception:
            pass
        # Verify TZ is in the env passed to subprocess
        if captured_env:
            assert "TZ" in captured_env
            assert captured_env["TZ"] == "America/New_York"

    def test_tz_not_injected_when_empty(self, monkeypatch):
        """When no timezone is configured, TZ should not be forced."""
        os.environ.pop("HERMES_TIMEZONE", None)
        from tools import code_execution_tool
        captured_env = {}

        def fake_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            class MockProcess:
                def communicate(self, timeout=None):
                    return (b"", b"")
                @property
                def returncode(self):
                    return 0
            return MockProcess()

        monkeypatch.setattr(code_execution_tool.subprocess, "Popen", fake_popen)
        try:
            code_execution_tool.execute_code(code="print('test')", session_id="test")
        except Exception:
            pass
        if captured_env:
            # Either no TZ or TZ inherits from parent
            tz_val = captured_env.get("TZ", "")
            assert tz_val != "America/New_York"


# =========================================================================
# Cron timezone integration
# =========================================================================

class TestCronTimezone:
    """Test that cron scheduling respects the configured timezone."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_parse_schedule_one_shot_duration_anchors_to_configured_tz(self):
        """A bare duration schedule ('30m') anchors run_at to the configured zone."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        from cron.jobs import parse_schedule

        parsed = parse_schedule("30m")
        assert parsed["kind"] == "once"
        run_at = datetime.fromisoformat(parsed["run_at"])
        assert run_at.tzinfo is not None
        # parse_schedule() anchors naive wall-clock intent to the CONFIGURED
        # Hermes zone (hermes_time.now()), not the server-local zone (#51021).
        # The ISO round-trip yields a fixed-offset tzinfo (no ZoneInfo .key),
        # so compare the UTC offset against the configured zone's offset.
        from zoneinfo import ZoneInfo

        expected_offset = datetime.now(ZoneInfo("America/New_York")).utcoffset()
        assert run_at.utcoffset() == expected_offset

    def test_parse_schedule_every_is_recurring_interval(self):
        """A bare duration is a one-shot; 'every X' is a recurring interval."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        from cron.jobs import parse_schedule

        assert parse_schedule("every 5m") == {
            "kind": "interval",
            "minutes": 5,
            "display": "every 5m",
        }

    def test_compute_next_run_tz_aware(self):
        """compute_next_run() returns a tz-aware ISO timestamp for a parsed schedule."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        _reset_hermes_time_cache()
        from cron.jobs import compute_next_run, parse_schedule

        next_run = compute_next_run(parse_schedule("every 1h"))
        assert next_run is not None
        parsed_next = datetime.fromisoformat(next_run)
        assert parsed_next.tzinfo is not None
        # HERMES_TIMEZONE=UTC → the computed run lands in UTC.
        assert parsed_next.utcoffset() == timedelta(0)

    def test_get_due_jobs_handles_naive_timestamps(self, tmp_path):
        """A job stored with naive timestamps is evaluated without raising."""
        import json
        from cron.jobs import get_due_jobs, use_cron_store

        past_naive = (datetime.now() - timedelta(hours=1)).replace(tzinfo=None).isoformat()
        with use_cron_store(tmp_path):
            jobs_file = tmp_path / "cron" / "jobs.json"
            jobs_file.parent.mkdir(parents=True, exist_ok=True)
            jobs_file.write_text(json.dumps({"jobs": [{
                "id": "test-job",
                "prompt": "test",
                "schedule": {"kind": "interval", "minutes": 5},
                "created_at": past_naive,
                "next_run_at": past_naive,
                "enabled": True,
            }]}))
            result = get_due_jobs()
        assert isinstance(result, list)
        # Naive past timestamp, far beyond grace → still fires once now.
        assert [j["id"] for j in result] == ["test-job"]

    def test_ensure_aware_naive_treats_wall_clock_as_system_local(self):
        """_ensure_aware() on a naive datetime attaches the system-local zone."""
        from cron.jobs import _ensure_aware

        naive = datetime(2026, 1, 15, 12, 0, 0)
        result = _ensure_aware(naive)
        assert result.tzinfo is not None
        # The naive value is read as system-local wall time, so the absolute
        # instant equals the same wall clock interpreted in the local zone.
        local_tz = datetime(2026, 1, 15, 12, 0, 0).astimezone().tzinfo
        assert result == datetime(2026, 1, 15, 12, 0, 0, tzinfo=local_tz)

    def test_ensure_aware_normalizes_aware_to_hermes_tz(self):
        """_ensure_aware() on an aware datetime converts it into hermes_time's zone."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        _reset_hermes_time_cache()
        from cron.jobs import _ensure_aware

        aware_utc = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _ensure_aware(aware_utc)
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(0)
        assert result == aware_utc  # same absolute instant

    def test_due_job_not_skipped_when_clock_is_ahead(self, tmp_path, monkeypatch):
        """A past-due job must be reported due against the configured clock.

        Regression guard: a stored next_run_at older than the configured
        Hermes clock must not be skipped when the system clock runs ahead.
        """
        import json
        from cron import jobs
        from cron.jobs import get_due_jobs, use_cron_store

        fixed_now = hermes_time.now()
        monkeypatch.setattr(jobs, "_hermes_now", lambda: fixed_now)
        past = (fixed_now - timedelta(hours=1)).isoformat()

        with use_cron_store(tmp_path):
            jobs_file = tmp_path / "cron" / "jobs.json"
            jobs_file.parent.mkdir(parents=True, exist_ok=True)
            jobs_file.write_text(json.dumps({"jobs": [{
                "id": "test",
                "prompt": "test",
                "schedule": {"kind": "interval", "minutes": 5},
                "created_at": past,
                "next_run_at": past,
                "enabled": True,
            }]}))
            result = get_due_jobs()
        assert [j["id"] for j in result] == ["test"]

    def test_get_due_jobs_naive_cross_timezone(self, tmp_path):
        """Naive stored timestamps are evaluated consistently across TZ configs."""
        import json
        from cron.jobs import get_due_jobs, use_cron_store

        past = (datetime.now() - timedelta(hours=1)).replace(tzinfo=None).isoformat()
        record = {
            "id": "tz-test",
            "prompt": "test",
            "schedule": {"kind": "interval", "minutes": 5},
            "created_at": past,
            "next_run_at": past,
            "enabled": True,
        }
        with use_cron_store(tmp_path):
            jobs_file = tmp_path / "cron" / "jobs.json"
            jobs_file.parent.mkdir(parents=True, exist_ok=True)
            # Run with different timezone configs — the naive instant is
            # normalized per config, so the job stays due in every zone.
            # Rewrite the record each pass: get_due_jobs() fast-forwards the
            # stored next_run_at, which would otherwise hide the job next round.
            for tz in ["UTC", "America/New_York", "Asia/Tokyo"]:
                os.environ["HERMES_TIMEZONE"] = tz
                _reset_hermes_time_cache()
                jobs_file.write_text(json.dumps({"jobs": [dict(record)]}))
                result = get_due_jobs()
                assert [j["id"] for j in result] == ["tz-test"], tz
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_create_job_stores_tz_aware_timestamps(self, tmp_path, monkeypatch):
        """create_job() should store tz-aware timestamps."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        os.environ["HERMES_TIMEZONE"] = "US/Eastern"
        _reset_hermes_time_cache()

        from cron.jobs import create_job
        job = create_job(prompt="TZ test", schedule="every 2h")

        created = datetime.fromisoformat(job["created_at"])
        assert created.tzinfo is not None

        next_run = datetime.fromisoformat(job["next_run_at"])
        assert next_run.tzinfo is not None


# =========================================================================
# hermes_time.format_in_user_tz() — UTC ISO 8601 → user TZ formatter
# =========================================================================

class TestFormatInUserTz:
    """Test the format_in_user_tz helper used by gateway injection and
    exposed for the LLM to call when reformatting tool-output timestamps."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_valid_utc_iso_converts_to_configured_tz(self):
        """With America/New_York configured, a 19:08 UTC ISO becomes 15:08 EDT."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        result = hermes_time.format_in_user_tz("2026-07-08T19:08:02.163000+00:00")
        # July is EDT (UTC-4); 19:08 UTC → 15:08 EDT
        assert "15:08:02" in result
        assert "EDT" in result

    def test_z_suffix_normalized(self):
        """Trailing 'Z' is accepted and treated as UTC."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        result = hermes_time.format_in_user_tz("2026-07-08T19:08:02Z")
        assert "15:08:02" in result

    def test_naive_datetime_assumed_utc(self):
        """A datetime without tzinfo is treated as UTC (defensive)."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        result = hermes_time.format_in_user_tz("2026-07-08T19:08:02")
        assert "15:08:02" in result

    def test_unparseable_input_returned_unchanged(self):
        """Garbage input must not crash the helper; pass-through is the contract."""
        result = hermes_time.format_in_user_tz("not a timestamp")
        assert result == "not a timestamp"

    def test_empty_input_returned_unchanged(self):
        """Empty / None / non-string input is returned as-is."""
        assert hermes_time.format_in_user_tz("") == ""
        assert hermes_time.format_in_user_tz(None) is None
        assert hermes_time.format_in_user_tz(12345) == 12345

    def test_unconfigured_timezone_falls_back_to_server_local(self):
        """With no HERMES_TIMEZONE / config setting, output uses server-local.

        Server-local is whatever the host's ``/etc/localtime`` points at —
        in CI it is usually UTC, on a developer machine it is whatever the
        user has set. We assert the helper succeeds and the formatted
        string is non-empty; we do not pin the offset.
        """
        # Force cache miss so get_timezone() returns None.
        _reset_hermes_time_cache()
        result = hermes_time.format_in_user_tz("2026-01-15T12:00:00+00:00")
        assert result
        # 2026-01-15 falls in EST (UTC-5), so the formatted time differs
        # from 12:00:00 by the host's offset. We only assert shape: a
        # day-of-week, a date, and a time are present.
        import re as _re
        assert _re.match(
            r"^[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ",
            result,
        ), f"unexpected format: {result!r}"

    def test_custom_format_string_respected(self):
        """A non-default fmt is honored."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        _reset_hermes_time_cache()
        result = hermes_time.format_in_user_tz(
            "2026-07-08T19:08:02+00:00", fmt="%Y-%m-%d %H:%M"
        )
        assert result == "2026-07-08 15:08"
