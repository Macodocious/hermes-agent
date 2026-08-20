"""Tests for fallback-cycle state persistence across agent rebuilds.

The gateway evicts and rebuilds cached agents mid-cycle (interrupt path,
/clear, compression splits). The in-memory ``_fallback_cycle_armed`` flag
died with the evicted instance, so the "Primary model restored" notice
could never fire after a rebuild.  The fix persists both cycle flags into
``model_config.gateway_runtime`` via the per-turn runtime sync and
rehydrates them at agent build time.

These tests pin the three halves of the contract:
1. ``_sync_session_model_from_agent`` persists the cycle flag.
2. ``_rehydrate_fallback_cycle_state`` restores it onto a fresh agent.
3. ``switch_model`` clears it on a deliberate model switch.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from run_agent import AIAgent
from agent.agent_init import _rehydrate_fallback_cycle_state
from agent.context_compressor import ContextCompressor


def _make_session_db(tmp_path, session_id, model_config=None, model="primary-model"):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(
        session_id,
        source="test",
        model=model,
        model_config=model_config or {},
    )
    return db


def _make_runner_with_db(db):
    """Minimal GatewayRunner with just the sync + session-db surface.

    The production gateway holds the AsyncSessionDB facade (the sync method
    reads ``self._session_db._db`` for the synchronous SessionDB), so the
    harness must mirror that shape.
    """
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_db = AsyncSessionDB(db)
    return runner


def _make_bare_agent():
    """AIAgent without running __init__ — only the attributes the sync /
    rehydrate helpers read."""
    agent = object.__new__(AIAgent)
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.session_id = "session-1"
    agent._persist_disabled = False
    agent._session_db = None
    return agent


def _runtime_config(db, session_id):
    row = db.get_session(session_id)
    config = json.loads(row["model_config"]) if row and row.get("model_config") else {}
    return config.get("gateway_runtime") or {}


# ── Persistence half (gateway per-turn sync) ──

def test_sync_persists_fallback_cycle_flag(tmp_path):
    db = _make_session_db(tmp_path, "session-1")
    runner = _make_runner_with_db(db)
    agent = _make_bare_agent()
    agent._fallback_cycle_armed = True
    agent._fallback_activated = True

    runner._sync_session_model_from_agent("session-1", agent)

    runtime = _runtime_config(db, "session-1")
    assert runtime["fallback_cycle_active"] is True
    assert runtime["fallback_active"] is True


def test_sync_updates_flag_when_cycle_disarms(tmp_path):
    db = _make_session_db(tmp_path, "session-1")
    runner = _make_runner_with_db(db)
    agent = _make_bare_agent()
    agent._fallback_cycle_armed = True
    runner._sync_session_model_from_agent("session-1", agent)
    assert _runtime_config(db, "session-1")["fallback_cycle_active"] is True

    # Restore-success path disarms the cycle in-memory; the next per-turn
    # sync must persist the disarmed state (no stale True survives).
    agent._fallback_cycle_armed = False
    runner._sync_session_model_from_agent("session-1", agent)
    assert _runtime_config(db, "session-1")["fallback_cycle_active"] is False


def test_sync_noop_without_session_db():
    agent = _make_bare_agent()
    agent._fallback_cycle_armed = True
    runner = _make_runner_with_db(None)

    # Must not raise.
    runner._sync_session_model_from_agent("session-1", agent)


# ── Rehydration half (agent build time) ──

def test_rehydrate_restores_both_flags(tmp_path):
    db = _make_session_db(
        tmp_path,
        "session-1",
        model_config={
            "gateway_runtime": {
                "fallback_cycle_active": True,
                "fallback_active": True,
            }
        },
    )
    agent = _make_bare_agent()
    agent._session_db = db

    _rehydrate_fallback_cycle_state(agent)

    assert agent._fallback_cycle_armed is True
    assert agent._fallback_activated is True


def test_rehydrate_restores_only_cycle_flag(tmp_path):
    db = _make_session_db(
        tmp_path,
        "session-1",
        model_config={
            "gateway_runtime": {
                "fallback_cycle_active": True,
                "fallback_active": False,
            }
        },
    )
    agent = _make_bare_agent()
    agent._session_db = db

    _rehydrate_fallback_cycle_state(agent)

    assert agent._fallback_cycle_armed is True
    # fallback_active is False → the helper must not set the flag (real
    # init_agent initializes _fallback_activated=False separately).
    assert getattr(agent, "_fallback_activated", False) is False


def test_rehydrate_noop_when_no_runtime_persisted(tmp_path):
    db = _make_session_db(tmp_path, "session-1", model_config={})
    agent = _make_bare_agent()
    agent._session_db = db

    _rehydrate_fallback_cycle_state(agent)

    assert getattr(agent, "_fallback_cycle_armed", False) is False
    assert getattr(agent, "_fallback_activated", False) is False


def test_rehydrate_noop_when_persist_disabled(tmp_path):
    db = _make_session_db(
        tmp_path,
        "session-1",
        model_config={
            "gateway_runtime": {"fallback_cycle_active": True},
        },
    )
    agent = _make_bare_agent()
    agent._session_db = db
    agent._persist_disabled = True

    _rehydrate_fallback_cycle_state(agent)

    assert getattr(agent, "_fallback_cycle_armed", False) is False


def test_rehydrate_noop_without_session_db():
    agent = _make_bare_agent()
    agent._session_db = None

    # Must not raise.
    _rehydrate_fallback_cycle_state(agent)

    assert getattr(agent, "_fallback_cycle_armed", False) is False


# ── Clear half (deliberate /model switch) ──

def _make_switchable_agent():
    """Minimal AIAgent that survives a full switch_model() (mirrors the
    harness in test_switch_model_context.py)."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = None
    agent._fallback_chain = []
    agent._fallback_index = 0
    agent._fallback_activated = False
    agent._fallback_cycle_armed = True
    agent._primary_runtime = {}
    agent._credential_pool = None
    agent._session_db = None
    agent._client_kwargs = {}
    agent.context_compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=None,
    )
    return agent


def test_switch_model_clears_fallback_cycle_flag():
    agent = _make_switchable_agent()
    # Stub the client rebuild so no network/config IO happens.
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()

    agent.switch_model(
        "new-model", "openrouter",
        api_key="sk-new", base_url="https://openrouter.ai/api/v1",
    )

    assert agent._fallback_cycle_armed is False
    assert agent._fallback_activated is False
