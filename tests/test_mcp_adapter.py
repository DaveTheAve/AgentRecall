from __future__ import annotations

import json

import pytest
from conftest import FakeEmbedder

import agent_recall_mcp
from agent_recall_core import AgentIdentity, AgentRecallCore
from agent_recall_mcp import MCPAccessError, MCPAdapter


def make_adapter(tmp_path, access: str = "read-write") -> MCPAdapter:
    core = AgentRecallCore(
        {
            "db_path": str(tmp_path / "mcp.db"),
            "workspace_id": "ws",
            "agent_id": "mcp-client",
            "embedding_base_url": "",
            "embedding_model": "fake",
        },
        AgentIdentity("ws", "mcp-client", "mcp-session"),
    )
    core.embedder = FakeEmbedder()
    return MCPAdapter(core, access=access)


def test_read_only_mcp_client_can_recall_but_cannot_mutate(tmp_path):
    writer = make_adapter(tmp_path)
    memory = writer.call("remember", {"content": "Shared MCP fact", "visibility": "shared"})
    writer.close()

    reader = make_adapter(tmp_path, access="read-only")
    before = reader.core.store.conn.execute(
        "SELECT access_count, last_accessed_at FROM memories WHERE id = ?", (memory["id"],)
    ).fetchone()
    result = reader.call("search", {"query": "MCP fact", "explain": True})
    assert result["results"][0]["id"] == memory["id"]
    reader.call("prefetch_context", {"query": "MCP fact"})
    reader.call("profile", {"focus": "MCP fact"})
    after = reader.core.store.conn.execute(
        "SELECT access_count, last_accessed_at FROM memories WHERE id = ?", (memory["id"],)
    ).fetchone()
    assert tuple(after) == tuple(before)
    with pytest.raises(MCPAccessError, match="read-only"):
        reader.call("remember", {"content": "must not write"})
    reader.close()


def test_mcp_exposes_curated_public_capabilities_not_internal_methods(tmp_path):
    adapter = make_adapter(tmp_path)
    capabilities = adapter.call("capabilities", {})

    assert capabilities["access"] == "read-write"
    assert "prefetch_context" in capabilities["tools"]
    assert "get_memory" in capabilities["tools"]
    assert "health" in capabilities["tools"]
    assert "_embedding" not in capabilities["tools"]
    adapter.close()


def test_mcp_prefetch_returns_bounded_context_with_identity(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.call("remember", {"content": "Use API lifecycle for task work", "visibility": "agent"})

    result = adapter.call("prefetch_context", {"query": "How should task work run?", "max_chars": 180, "explain": True})

    assert len(result["context"]) <= 180
    assert result["results"] == []
    assert result["count"] == 1
    assert result["identity"] == {"workspace_id": "ws", "agent_id": "mcp-client"}
    adapter.close()


def test_configured_mcp_adapter_constructs_the_host_neutral_curator(tmp_path, monkeypatch):
    class FakeCurator:
        def curate(self, text, *, default_visibility="agent"):
            return [{"content": f"curated: {text}", "visibility": default_visibility}]

    config_path = tmp_path / "agent-recall.json"
    config_path.write_text(
        json.dumps(
            {
                "db_path": str(tmp_path / "mcp-curation.db"),
                "embedding_base_url": "",
                "llm_curator_enabled": True,
                "curated_memories_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_recall_mcp, "build_curator", lambda _config: FakeCurator())
    adapter = agent_recall_mcp.create_adapter_from_config(
        config_path=str(config_path),
        workspace_id="ws",
        agent_id="mcp-client",
    )

    result = adapter.call("curate", {"text": "durable MCP fact", "dry_run": False})
    assert result["stored"] == 1
    found = adapter.call("search", {"query": "durable MCP fact"})
    assert found["results"][0]["content"] == "curated: durable MCP fact"
    adapter.close()


def test_mcp_does_not_disclose_server_filesystem_paths_or_session_ids(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter.call(
        "remember",
        {
            "content": "memory with private provenance",
            "metadata": {"source_path": "/home/service/private/secret.md"},
        },
    )
    found = adapter.call("search", {"query": "private provenance", "include_results": True})["results"][0]
    assert "session_id" not in found
    assert found["metadata"]["source_path"] == "[redacted]"
    assert "db_path" not in adapter.call("profile", {})
    assert "db_path" not in adapter.call("stats", {})["stats"]
    assert "db_path" not in adapter.call("health", {})["sqlite"]
    assert "session_id" not in adapter.call("capabilities", {}).get("identity", {})
    adapter.close()
