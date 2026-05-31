from __future__ import annotations

from agent_recall_store import AgentRecallStore


def test_shared_memory_visible_but_private_memory_isolated(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(workspace_id="ws", agent_id="hermes", source_agent_id="hermes", visibility="shared", session_id="s1", content="shared user preference", embedding=[1.0, 0.0], embedding_model="fake")
    store.add_memory(workspace_id="ws", agent_id="hermes", source_agent_id="hermes", visibility="agent", session_id="s1", content="hermes private note", embedding=[0.0, 1.0], embedding_model="fake")

    results = store.search(workspace_id="ws", agent_id="agentforge", session_id="s2", query="preference private", query_embedding=[1.0, 0.0], limit=10)
    contents = [r["content"] for r in results]

    assert "shared user preference" in contents
    assert "hermes private note" not in contents


def test_session_memory_requires_same_agent_and_session(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(workspace_id="ws", agent_id="hermes", source_agent_id="hermes", visibility="session", session_id="s1", content="session only", embedding=[1.0], embedding_model="fake")

    wrong_session = store.search(workspace_id="ws", agent_id="hermes", session_id="s2", query="session", query_embedding=[1.0], limit=10)
    right_session = store.search(workspace_id="ws", agent_id="hermes", session_id="s1", query="session", query_embedding=[1.0], limit=10)

    assert wrong_session == []
    assert [r["content"] for r in right_session] == ["session only"]


def test_workspace_boundary_blocks_cross_workspace_recall(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(workspace_id="ws-a", agent_id="hermes", source_agent_id="hermes", visibility="shared", content="workspace a fact", embedding=[1.0], embedding_model="fake")

    results = store.search(workspace_id="ws-b", agent_id="hermes", session_id="", query="workspace", query_embedding=[1.0], limit=10)

    assert results == []


def test_other_agent_private_memory_cannot_be_updated_or_deleted(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    mem_id = store.add_memory(workspace_id="ws", agent_id="hermes", source_agent_id="hermes", visibility="agent", content="private", embedding=[1.0], embedding_model="fake")

    assert store.update_memory(mem_id, "ws", "agentforge", "", content="hacked") is False
    assert store.delete_memory(mem_id, "ws", "agentforge", "") is False
    visible = store.search(workspace_id="ws", agent_id="hermes", session_id="", query="private", query_embedding=[1.0], limit=5)
    assert visible[0]["content"] == "private"
