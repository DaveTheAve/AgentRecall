from __future__ import annotations

import json

from conftest import FakeEmbedder, load_provider_module


def make_provider(tmp_path, config=None):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        **(config or {}),
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    return mod, p


def test_advanced_modules_are_config_toggles(tmp_path):
    mod = load_provider_module()
    cfg = mod._default_config(tmp_path)
    assert cfg["raw_memories_enabled"] is True
    assert cfg["curated_memories_enabled"] is True
    assert cfg["conclusions_enabled"] is False
    assert cfg["peer_profiles_enabled"] is False
    assert cfg["workspace_profiles_enabled"] is False
    assert cfg["agent_profiles_enabled"] is False
    assert cfg["dialectic_review_enabled"] is False
    assert cfg["conflict_detection_enabled"] is False
    assert cfg["staleness_detection_enabled"] is False
    assert cfg["promotion_rules_enabled"] is False
    assert cfg["demotion_rules_enabled"] is False

    normalized = mod._normalize_config({**cfg, "conclusions_enabled": "true", "peer_profiles_enabled": "true"})
    assert normalized["conclusions_enabled"] is True
    assert normalized["peer_profiles_enabled"] is True


def test_config_schema_exposes_every_production_toggle():
    mod = load_provider_module()
    keys = {item["key"] for item in mod.AgentRecallProvider().get_config_schema()}
    for key in {
        "raw_memories_enabled",
        "curated_memories_enabled",
        "conclusions_enabled",
        "peer_profiles_enabled",
        "workspace_profiles_enabled",
        "agent_profiles_enabled",
        "dialectic_review_enabled",
        "conflict_detection_enabled",
        "staleness_detection_enabled",
        "promotion_rules_enabled",
        "demotion_rules_enabled",
        "auto_capture_turns",
        "auto_capture_compression_checkpoints",
        "allow_any_agent_to_mutate_shared",
        "llm_curator_model",
    }:
        assert key in keys


def test_conclusions_are_disabled_unless_enabled(tmp_path):
    _, p = make_provider(tmp_path)
    result = json.loads(p.handle_tool_call("agent_recall_conclude", {"content": "User prefers explicit provenance"}))
    assert result["success"] is False
    assert "conclusions_enabled" in result["error"]


def test_conclusions_store_inspectable_provenance_when_enabled(tmp_path):
    _, p = make_provider(tmp_path, {"conclusions_enabled": True})
    result = json.loads(p.handle_tool_call("agent_recall_conclude", {
        "content": "User prefers explicit provenance",
        "scope": "peer",
        "subject": "peer-1",
        "source_ids": [1, 2],
        "visibility": "shared",
        "confidence": 0.91,
    }))
    assert result["success"] is True
    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "explicit provenance", "category": "conclusion"}))
    row = found["results"][0]
    assert row["content"] == "User prefers explicit provenance"
    assert row["metadata"]["module"] == "conclusions"
    assert row["metadata"]["source_ids"] == [1, 2]
    assert row["metadata"]["scope"] == "peer"
    assert row["metadata"]["subject"] == "peer-1"


def test_profile_synthesis_uses_configured_curator_model_not_an_unrelated_model(monkeypatch, tmp_path):
    _, p = make_provider(tmp_path, {
        "peer_profiles_enabled": True,
        "llm_curator_model": "custom-chat-model",
        "llm_curator_backend": "codex-cli",
    })
    seen = {}

    class FakeCurator:
        def __init__(self, command, model, timeout):
            seen["model"] = model
            seen["command"] = command

        def curate(self, text, *, default_visibility):
            seen["text"] = text
            return [{"content": "Synthesized peer profile: user likes provenance", "visibility": default_visibility, "category": "peer_profile"}]

    monkeypatch.setattr(__import__(p.__class__.__module__, fromlist=["CodexCliCurator"]), "CodexCliCurator", FakeCurator)
    result = json.loads(p.handle_tool_call("agent_recall_profile_synthesize", {"scope": "peer", "subject": "peer-1", "dry_run": True}))
    assert result["success"] is True
    assert seen["model"] == "custom-chat-model"
    assert seen["model"] != "unrelated-chat-model"
    assert result["stored"] == 0
    assert result["candidates"][0]["content"].startswith("Synthesized peer profile")


def test_profile_scope_must_be_enabled(tmp_path):
    _, p = make_provider(tmp_path, {"peer_profiles_enabled": False})
    result = json.loads(p.handle_tool_call("agent_recall_profile_synthesize", {"scope": "peer", "subject": "peer-1"}))
    assert result["success"] is False
    assert "peer_profiles_enabled" in result["error"]


def test_dialectic_review_is_disabled_unless_enabled(tmp_path):
    _, p = make_provider(tmp_path)
    result = json.loads(p.handle_tool_call("agent_recall_review", {"dry_run": True}))
    assert result["success"] is False
    assert "dialectic_review_enabled" in result["error"]


def test_dialectic_review_respects_llm_curator_model_and_dry_run(monkeypatch, tmp_path):
    _, p = make_provider(tmp_path, {
        "dialectic_review_enabled": True,
        "conflict_detection_enabled": True,
        "staleness_detection_enabled": True,
        "promotion_rules_enabled": True,
        "demotion_rules_enabled": True,
        "llm_curator_model": "review-model",
    })
    p.handle_tool_call("agent_recall_remember", {"content": "User likes clear source provenance", "visibility": "agent"})
    seen = {}

    class FakeCurator:
        def __init__(self, command, model, timeout):
            seen["model"] = model

        def curate(self, text, *, default_visibility):
            seen["text"] = text
            return [{"content": "Promote source provenance preference", "visibility": "agent", "category": "review_recommendation"}]

    monkeypatch.setattr(__import__(p.__class__.__module__, fromlist=["CodexCliCurator"]), "CodexCliCurator", FakeCurator)
    result = json.loads(p.handle_tool_call("agent_recall_review", {"dry_run": True, "focus": "provenance"}))
    assert result["success"] is True
    assert result["stored"] == 0
    assert seen["model"] == "review-model"
    assert "conflict_detection" in result["enabled_modules"]
    assert "staleness_detection" in result["enabled_modules"]
    assert result["recommendations"][0]["content"] == "Promote source provenance preference"
