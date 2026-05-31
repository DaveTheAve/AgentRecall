from __future__ import annotations

from conftest import load_provider_module


def test_string_booleans_from_setup_are_normalized(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        "shared_recall": "false",
        "auto_capture_turns": "true",
        "allow_any_agent_to_mutate_shared": "false",
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    assert p._config["shared_recall"] is False
    assert p._config["auto_capture_turns"] is True
    assert p._config["allow_any_agent_to_mutate_shared"] is False


def test_invalid_embedding_dimensions_env_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_RECALL_EMBEDDING_DIMENSIONS", "not-an-int")
    mod = load_provider_module()
    assert mod._default_config(tmp_path)["embedding_dimensions"] == 0