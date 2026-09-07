from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from conftest import FakeEmbedder, load_provider_module


def test_save_config_expands_hermes_home_and_initialize_uses_agent_identity(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider()
    p.save_config({"db_path": "$HERMES_HOME/shared.db", "workspace_id": "ws", "agent_id": ""}, tmp_path)
    cfg = json.loads((tmp_path / "agent-recall.json").read_text())
    assert cfg["db_path"] == str(tmp_path / "shared.db")
    if os.name == "posix":
        assert (tmp_path / "agent-recall.json").stat().st_mode & 0o777 == 0o600

    p.initialize("s", hermes_home=tmp_path, agent_identity="other-agent")
    assert p._agent_id == "other-agent"
    assert p._workspace_id == "ws"


def test_pre_compress_saves_session_checkpoint(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider(
        {
            "db_path": str(tmp_path / "agent-recall.db"),
            "workspace_id": "ws",
            "agent_id": "hermes",
            "embedding_base_url": "",
            "embedding_model": "fake",
            "auto_capture_compression_checkpoints": True,
        }
    )
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    msg = [
        {"role": "user", "content": "Important detail about the current debugging session."},
        {"role": "assistant", "content": "Acknowledged and used it."},
    ]
    result = p.on_pre_compress(msg)
    assert "checkpoint id=" in result
    found = json.loads(
        p.handle_tool_call("agent_recall_search", {"query": "debugging session", "include_shared": False})
    )
    assert found["count"] == 1
    assert found["results"][0]["visibility"] == "session"


def test_session_switch_rotates_core_identity_for_session_scoped_operations(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider(
        {
            "db_path": str(tmp_path / "agent-recall.db"),
            "workspace_id": "ws",
            "agent_id": "hermes",
            "embedding_base_url": "",
            "embedding_model": "fake",
        }
    )
    p.initialize("old-session", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    assert (
        json.loads(
            p.handle_tool_call(
                "agent_recall_remember",
                {"content": "old session memory", "visibility": "session"},
            )
        )["success"]
        is True
    )

    p.on_session_switch("new-session", parent_session_id="old-session")
    assert p._session_id == "new-session"
    assert p._core.identity.session_id == "new-session"
    new = json.loads(
        p.handle_tool_call(
            "agent_recall_remember",
            {"content": "new session memory", "visibility": "session"},
        )
    )

    visible = json.loads(
        p.handle_tool_call(
            "agent_recall_search",
            {"query": "session memory", "include_shared": False},
        )
    )
    assert [row["id"] for row in visible["results"]] == [new["id"]]
    p.shutdown()


def test_reinitialize_closes_the_previous_core_connection(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider(
        {
            "db_path": str(tmp_path / "agent-recall.db"),
            "workspace_id": "ws",
            "agent_id": "hermes",
            "embedding_base_url": "",
        }
    )
    p.initialize("first", hermes_home=tmp_path, agent_identity="hermes")
    previous = p._core

    p.initialize("second", hermes_home=tmp_path, agent_identity="hermes")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        previous.store.conn.execute("SELECT 1")
    p.shutdown()


def _provider(tmp_path, **overrides):
    mod = load_provider_module()
    provider = mod.AgentRecallProvider(
        {
            "db_path": str(tmp_path / "agent-recall.db"),
            "embedding_base_url": "",
            **overrides,
        }
    )
    provider.initialize("session-one", hermes_home=tmp_path, agent_identity="hermes")
    return provider


















def test_archive_prompt_remains_separate_without_learning_preview(tmp_path):
    provider = _provider(tmp_path, session_archive_enabled=True)
    try:
        prompt = provider.system_prompt_block()
        assert "SessionArchive is a separate, read-only source owned by the current Hermes profile." in prompt
        assert "learning" not in prompt.lower()
        assert "preview" not in prompt.lower()
    finally:
        provider.shutdown()


def test_config_schema_uses_only_hermes_supported_types():
    mod = load_provider_module()
    supported_types = {"text", "integer", "number", "boolean"}
    invalid_types = {
        field["key"]: field["type"]
        for field in mod.AgentRecallProvider().get_config_schema()
        if "type" in field and field["type"] not in supported_types
    }

    assert invalid_types == {}






def test_both_hermes_manifests_omit_unsupported_session_end_hook():
    root = Path(__file__).resolve().parents[1]
    for manifest in (root / "plugin.yaml", root / "hermes_plugin" / "plugin.yaml"):
        hooks = {
            line.removeprefix("  - ").strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.startswith("  - ")
        }
        assert "on_session_end" not in hooks
