from __future__ import annotations

import json

from conftest import FakeEmbedder, load_provider_module


def make_provider(tmp_path, agent="hermes", workspace="ws"):
    mod = load_provider_module()
    provider = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": workspace,
        "agent_id": agent,
        "embedding_base_url": "",
        "embedding_model": "fake",
        "excluded_terms": ["blocked-project"],
        "prefetch_limit": 4,
    })
    provider.initialize("session-1", hermes_home=tmp_path, agent_identity=agent)
    provider._embedder = FakeEmbedder()
    return provider


def test_remember_search_profile_stats_and_forget(tmp_path):
    p = make_provider(tmp_path)
    added = json.loads(p.handle_tool_call("agent_recall_remember", {"content": "User prefers concise responses", "visibility": "shared", "category": "user_pref", "tags": ["preference"]}))
    assert added["success"] is True

    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "concise responses", "limit": 5}))
    assert found["count"] == 1
    assert found["results"][0]["visibility"] == "shared"

    profile = json.loads(p.handle_tool_call("agent_recall_profile", {"focus": "concise", "limit": 3}))
    assert profile["workspace_id"] == "ws"
    assert profile["agent_id"] == "hermes"
    assert profile["recall"]

    stats = json.loads(p.handle_tool_call("agent_recall_stats", {}))
    assert stats["success"] is True
    assert stats["stats"]["workspace_id"] == "ws"

    deleted = json.loads(p.handle_tool_call("agent_recall_forget", {"id": added["id"]}))
    assert deleted["deleted"] is True
    assert json.loads(p.handle_tool_call("agent_recall_search", {"query": "concise"}))["count"] == 0


def test_excluded_terms_block_storage(tmp_path):
    p = make_provider(tmp_path)
    result = json.loads(p.handle_tool_call("agent_recall_remember", {"content": "Do not store blocked-project residue"}))
    assert result["success"] is False
    assert "excluded term" in result["error"]


def test_prefetch_formats_recalled_context(tmp_path):
    p = make_provider(tmp_path)
    json.loads(p.handle_tool_call("agent_recall_remember", {"content": "The coding agent uses a local model for main reasoning", "visibility": "shared"}))
    block = p.prefetch("What does the coding agent use?", session_id="session-1")
    assert "AgentRecall Recalled Context" in block
    assert "coding agent uses a local model" in block


def test_prefetch_honors_the_explicit_session_after_the_provider_switches(tmp_path):
    p = make_provider(tmp_path)
    stored = json.loads(
        p.handle_tool_call(
            "agent_recall_remember",
            {"content": "Session one private deployment note", "visibility": "session"},
        )
    )
    assert stored["success"] is True
    p.on_session_switch("session-2")

    assert p.prefetch("private deployment note", session_id="session-2") == ""
    original = p.prefetch("private deployment note", session_id="session-1")
    assert "Session one private deployment note" in original


class FailingEmbedder:
    def embed(self, text: str):
        raise RuntimeError("embedding down")


def test_embedding_failure_degrades_to_lexical_search(tmp_path):
    p = make_provider(tmp_path)
    p._embedder = FailingEmbedder()
    added = json.loads(p.handle_tool_call("agent_recall_remember", {"content": "Lexical fallback fact", "visibility": "agent"}))
    assert added["success"] is True
    assert "embedding_warning" in added

    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "fallback fact"}))
    assert found["success"] is True
    assert found["count"] == 1
    assert found["results"][0]["content"] == "Lexical fallback fact"
    assert "embedding_warning" in found


def test_import_markdown_chunks_file(tmp_path):
    p = make_provider(tmp_path)
    note = tmp_path / "Memory.md"
    note.write_text("# Title\n" + "A useful shared note.\n" * 200, encoding="utf-8")
    result = json.loads(p.handle_tool_call("agent_recall_import_markdown", {"path": str(note), "visibility": "shared", "chunk_chars": 1000}))
    assert result["success"] is True
    assert result["imported"] >= 2


def test_builtin_memory_write_mirrors_without_throwing(tmp_path):
    p = make_provider(tmp_path)
    p.on_memory_write("add", "user", "User prefers CLI-friendly output", metadata={"source": "test"})
    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "CLI output"}))
    assert found["count"] == 1
    assert found["results"][0]["visibility"] == "shared"
