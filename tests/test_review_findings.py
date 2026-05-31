from __future__ import annotations

import json

from conftest import FakeEmbedder, load_provider_module


def test_default_embedding_base_url_ignores_remote_openai_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://remote.example/v1")
    mod = load_provider_module()
    p = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes"})
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    assert p._config["embedding_base_url"] == "http://127.0.0.1:6660/v1"


def test_pre_compress_checkpoint_is_opt_in(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes", "embedding_base_url": "", "embedding_model": "fake"})
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    msg = [{"role": "user", "content": "Important detail about the current debugging session."}]
    assert p.on_pre_compress(msg) == ""
    assert json.loads(p.handle_tool_call("agent_recall_search", {"query": "debugging session", "include_shared": False}))["count"] == 0


def test_stats_only_exposes_visible_buckets(tmp_path):
    mod = load_provider_module()
    owner = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes", "embedding_base_url": "", "embedding_model": "fake"})
    owner.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    owner._embedder = FakeEmbedder()
    json.loads(owner.handle_tool_call("agent_recall_remember", {"content": "private", "visibility": "agent", "category": "secret_category"}))
    json.loads(owner.handle_tool_call("agent_recall_remember", {"content": "shared", "visibility": "shared", "category": "shared_category"}))
    owner.shutdown()

    other = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "agentforge", "embedding_base_url": "", "embedding_model": "fake"})
    other.initialize("s2", hermes_home=tmp_path, agent_identity="agentforge")
    stats = json.loads(other.handle_tool_call("agent_recall_stats", {}))["stats"]
    categories = {b["category"] for b in stats["buckets"]}
    assert "shared_category" in categories
    assert "secret_category" not in categories


def test_import_markdown_rejects_non_markdown_outside_roots_symlinks_and_excluded_terms(tmp_path):
    mod = load_provider_module()
    vault = tmp_path / "vault"
    p = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes", "embedding_base_url": "", "embedding_model": "fake", "import_roots": [str(vault)], "excluded_terms": ["blocked-project"]})
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    txt = vault / "secret.txt"
    txt.parent.mkdir()
    txt.write_text("secret", encoding="utf-8")
    target = vault / "target.md"
    target.write_text("target", encoding="utf-8")
    symlink = vault / "link.md"
    symlink.symlink_to(target)
    blocked = vault / "blocked.md"
    blocked.write_text("contains blocked-project content", encoding="utf-8")

    assert json.loads(p.handle_tool_call("agent_recall_import_markdown", {"path": str(outside)}))["success"] is False
    assert json.loads(p.handle_tool_call("agent_recall_import_markdown", {"path": str(txt)}))["success"] is False
    symlink_result = json.loads(p.handle_tool_call("agent_recall_import_markdown", {"path": str(symlink)}))
    assert symlink_result["success"] is False
    assert "symlink" in symlink_result["error"]
    excluded_result = json.loads(p.handle_tool_call("agent_recall_import_markdown", {"path": str(blocked)}))
    assert excluded_result["success"] is False
    assert "excluded term" in excluded_result["error"]


def test_tags_string_is_rejected(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db"), "workspace_id": "ws", "agent_id": "hermes", "embedding_base_url": "", "embedding_model": "fake"})
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    result = json.loads(p.handle_tool_call("agent_recall_remember", {"content": "fact", "tags": "not-a-list"}))
    assert result["success"] is False
    assert "tags" in result["error"]
