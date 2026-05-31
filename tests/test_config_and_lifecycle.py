from __future__ import annotations

import json
import os

from conftest import FakeEmbedder, load_provider_module


def test_save_config_expands_hermes_home_and_initialize_uses_agent_identity(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider()
    p.save_config({"db_path": "$HERMES_HOME/shared.db", "workspace_id": "ws", "agent_id": ""}, tmp_path)
    cfg = json.loads((tmp_path / "agent-recall.json").read_text())
    assert cfg["db_path"] == str(tmp_path / "shared.db")
    if os.name == "posix":
        assert (tmp_path / "agent-recall.json").stat().st_mode & 0o777 == 0o600

    p.initialize("s", hermes_home=tmp_path, agent_identity="agentforge")
    assert p._agent_id == "agentforge"
    assert p._workspace_id == "ws"


def test_pre_compress_saves_session_checkpoint(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes", "embedding_base_url": "", "embedding_model": "fake", "auto_capture_compression_checkpoints": True})
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    msg = [{"role": "user", "content": "Important detail about the current debugging session."}, {"role": "assistant", "content": "Acknowledged and used it."}]
    result = p.on_pre_compress(msg)
    assert "checkpoint id=" in result
    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "debugging session", "include_shared": False}))
    assert found["count"] == 1
    assert found["results"][0]["visibility"] == "session"
