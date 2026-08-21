from __future__ import annotations

from agent_recall_store import AgentRecallStore


def test_shared_memory_visible_but_private_memory_isolated(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="shared",
        session_id="s1",
        content="shared user preference",
        embedding=[1.0, 0.0],
        embedding_model="fake",
    )
    store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        session_id="s1",
        content="hermes private note",
        embedding=[0.0, 1.0],
        embedding_model="fake",
    )

    results = store.search(
        workspace_id="ws",
        agent_id="agentforge",
        session_id="s2",
        query="preference private",
        query_embedding=[1.0, 0.0],
        limit=10,
    )
    contents = [r["content"] for r in results]

    assert "shared user preference" in contents
    assert "hermes private note" not in contents


def test_session_memory_requires_same_agent_and_session(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="session",
        session_id="s1",
        content="session only",
        embedding=[1.0],
        embedding_model="fake",
    )

    wrong_session = store.search(
        workspace_id="ws", agent_id="hermes", session_id="s2", query="session", query_embedding=[1.0], limit=10
    )
    right_session = store.search(
        workspace_id="ws", agent_id="hermes", session_id="s1", query="session", query_embedding=[1.0], limit=10
    )

    assert wrong_session == []
    assert [r["content"] for r in right_session] == ["session only"]


def test_empty_session_identity_cannot_authorize_session_scoped_rows(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    memory_id = store.add_memory(
        workspace_id="ws",
        agent_id="openclaw",
        source_agent_id="openclaw",
        visibility="session",
        session_id="",
        content="must not become agent-wide",
        embedding=[1.0],
        embedding_model="fake",
    )

    results = store.search(
        workspace_id="ws",
        agent_id="openclaw",
        session_id="",
        query="agent-wide",
        query_embedding=[1.0],
        limit=10,
    )

    assert results == []
    assert store.get_memory(memory_id, "ws", "openclaw", "") is None
    store.close()


def test_workspace_boundary_blocks_cross_workspace_recall(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    store.add_memory(
        workspace_id="ws-a",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="shared",
        content="workspace a fact",
        embedding=[1.0],
        embedding_model="fake",
    )

    results = store.search(
        workspace_id="ws-b", agent_id="hermes", session_id="", query="workspace", query_embedding=[1.0], limit=10
    )

    assert results == []


def test_other_agent_private_memory_cannot_be_updated_or_deleted(tmp_path):
    store = AgentRecallStore(tmp_path / "agent-recall.db")
    mem_id = store.add_memory(
        workspace_id="ws",
        agent_id="hermes",
        source_agent_id="hermes",
        visibility="agent",
        content="private",
        embedding=[1.0],
        embedding_model="fake",
    )

    assert store.update_memory(mem_id, "ws", "agentforge", "", content="hacked") is False
    assert store.delete_memory(mem_id, "ws", "agentforge", "") is False
    visible = store.search(
        workspace_id="ws", agent_id="hermes", session_id="", query="private", query_embedding=[1.0], limit=5
    )
    assert visible[0]["content"] == "private"


def test_shared_delete_rechecks_acl_after_concurrent_visibility_change(tmp_path, monkeypatch):
    owner = AgentRecallStore(tmp_path / "delete-race.db")
    other = AgentRecallStore(tmp_path / "delete-race.db")
    memory_id = owner.add_memory(
        workspace_id="ws",
        agent_id="owner",
        source_agent_id="owner",
        visibility="shared",
        content="shared then private",
        embedding=[1.0],
        embedding_model="fake",
    )
    original_get_visible = other.get_visible

    def get_visible_then_privatize(*args, **kwargs):
        row = original_get_visible(*args, **kwargs)
        assert owner.update_memory(memory_id, "ws", "owner", "", visibility="agent") is True
        return row

    monkeypatch.setattr(other, "get_visible", get_visible_then_privatize)

    assert other.delete_memory(memory_id, "ws", "other", "", allow_shared_mutation=True) is False
    row = owner.get_memory(memory_id, "ws", "owner", "")
    assert row is not None
    assert row["visibility"] == "agent"
    owner.close()
    other.close()


def test_shared_update_rechecks_acl_after_concurrent_visibility_change(tmp_path, monkeypatch):
    owner = AgentRecallStore(tmp_path / "update-race.db")
    other = AgentRecallStore(tmp_path / "update-race.db")
    memory_id = owner.add_memory(
        workspace_id="ws",
        agent_id="owner",
        source_agent_id="owner",
        visibility="shared",
        category="original",
        content="shared then private",
        embedding=[1.0],
        embedding_model="fake",
    )
    original_get_visible = other.get_visible

    def get_visible_then_privatize(*args, **kwargs):
        row = original_get_visible(*args, **kwargs)
        assert owner.update_memory(memory_id, "ws", "owner", "", visibility="agent") is True
        return row

    monkeypatch.setattr(other, "get_visible", get_visible_then_privatize)

    assert (
        other.update_memory(
            memory_id,
            "ws",
            "other",
            "",
            allow_shared_mutation=True,
            category="unauthorized",
        )
        is False
    )
    row = owner.get_memory(memory_id, "ws", "owner", "")
    assert row is not None
    assert row["visibility"] == "agent"
    assert row["category"] == "original"
    owner.close()
    other.close()


def test_cross_agent_shared_mutator_cannot_change_owner_visibility_scope(tmp_path):
    owner = AgentRecallStore(tmp_path / "shared-scope.db")
    other = AgentRecallStore(tmp_path / "shared-scope.db")
    memory_id = owner.add_memory(
        workspace_id="ws",
        agent_id="owner",
        source_agent_id="owner",
        visibility="shared",
        category="original",
        content="shared scope remains owner-controlled",
        embedding=[1.0],
        embedding_model="fake",
    )

    assert (
        other.update_memory(
            memory_id,
            "ws",
            "other",
            "other-session",
            allow_shared_mutation=True,
            visibility="session",
            target_session_id="other-session",
        )
        is False
    )
    assert (
        other.update_memory(
            memory_id,
            "ws",
            "other",
            "other-session",
            allow_shared_mutation=True,
            category="allowed-shared-edit",
        )
        is True
    )
    row = owner.get_memory(memory_id, "ws", "owner", "")
    assert row is not None
    assert row["visibility"] == "shared"
    assert row["session_id"] == ""
    assert row["category"] == "allowed-shared-edit"
    owner.close()
    other.close()
