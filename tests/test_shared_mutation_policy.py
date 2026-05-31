from __future__ import annotations

import json

from conftest import FakeEmbedder, load_provider_module


def make_provider(tmp_path, agent, allow_any=False):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": agent,
        "embedding_base_url": "",
        "embedding_model": "fake",
        "allow_any_agent_to_mutate_shared": allow_any,
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity=agent)
    p._embedder = FakeEmbedder()
    return p


def test_shared_memories_are_readable_but_owner_mutable_by_default(tmp_path):
    owner = make_provider(tmp_path, "hermes")
    added = json.loads(owner.handle_tool_call("agent_recall_remember", {"content": "Shared durable fact", "visibility": "shared"}))
    owner.shutdown()

    other = make_provider(tmp_path, "agentforge")
    found = json.loads(other.handle_tool_call("agent_recall_search", {"query": "durable fact"}))
    assert found["count"] == 1

    update = json.loads(other.handle_tool_call("agent_recall_update", {"id": added["id"], "content": "Tampered"}))
    delete = json.loads(other.handle_tool_call("agent_recall_forget", {"id": added["id"]}))
    assert update["updated"] is False
    assert delete["deleted"] is False


def test_shared_mutation_can_be_explicitly_relaxed(tmp_path):
    owner = make_provider(tmp_path, "hermes", allow_any=True)
    added = json.loads(owner.handle_tool_call("agent_recall_remember", {"content": "Shared mutable fact", "visibility": "shared"}))
    owner.shutdown()

    other = make_provider(tmp_path, "agentforge", allow_any=True)
    update = json.loads(other.handle_tool_call("agent_recall_update", {"id": added["id"], "content": "Collaboratively updated"}))
    assert update["updated"] is True
    found = json.loads(other.handle_tool_call("agent_recall_search", {"query": "Collaboratively"}))
    assert found["results"][0]["content"] == "Collaboratively updated"
