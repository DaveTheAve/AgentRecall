"""Retained real-SQL explicit-memory contracts; automatic extraction is removed."""
from __future__ import annotations

from conftest import FakeEmbedder

from agent_recall_core import AgentIdentity, AgentRecallCore


def _core(tmp_path):
    core = AgentRecallCore({"db_path": str(tmp_path / "memories.db")}, AgentIdentity("workspace-a", "agent-a", "session"))
    core.embedder = FakeEmbedder()
    return core


def _rows(core):
    return [dict(row) for row in core.store.conn.execute("SELECT * FROM memories ORDER BY id")]


def test_short_fact_explicit_remember_recalls_source(tmp_path):
    core = _core(tmp_path)
    try:
        assert core.remember({"content": "I use Go.", "visibility": "agent"})["action"] == "added"
        assert [row["content"] for row in _rows(core)] == ["I use Go."]
        assert [row["content"] for row in core.search({"query": "Go", "min_score": 0})["results"]] == ["I use Go."]
    finally:
        core.close()


def test_manual_remember_canonical_upsert_does_not_duplicate_source(tmp_path):
    core = _core(tmp_path)
    try:
        args = {"content": "I prefer concise release summaries.", "visibility": "agent", "canonical_key": "release-style"}
        first = core.remember(args)
        second = core.remember(args)
        assert first["id"] == second["id"]
        assert [row["content"] for row in _rows(core)] == [args["content"]]
    finally:
        core.close()


def test_explicit_manual_memory_survives_reopen_and_private_acl(tmp_path):
    source = "I prefer concise release summaries."
    core = _core(tmp_path)
    try:
        assert core.remember({"content": source, "visibility": "agent"})["action"] == "added"
        memory_id = _rows(core)[0]["id"]
        config = dict(core.config)
    finally:
        core.close()
    owner = AgentRecallCore(config, AgentIdentity("workspace-a", "agent-a", "later-session"))
    peer = AgentRecallCore(config, AgentIdentity("workspace-a", "other-agent", "later-session"))
    owner.embedder = FakeEmbedder()
    peer.embedder = FakeEmbedder()
    try:
        assert peer.get_memory(memory_id)["success"] is False
        assert peer.search({"query": "concise release summaries", "min_score": 0})["results"] == []
        assert owner.get_memory(memory_id)["memory"]["content"] == source
        assert [row["content"] for row in owner.search({"query": "concise release summaries", "min_score": 0})["results"]] == [source]
    finally:
        owner.close()
        peer.close()
