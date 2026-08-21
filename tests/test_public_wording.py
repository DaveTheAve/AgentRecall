from __future__ import annotations

from pathlib import Path

from conftest import load_provider_module

ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_does_not_frame_project_as_embedding_only():
    combined = "\n".join(
        [
            (ROOT / "plugin.yaml").read_text(encoding="utf-8"),
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        ]
    ).lower()
    forbidden = ["embedding-only", "no chat", "no background chat", "chat llm load"]
    assert not any(term in combined for term in forbidden)


def test_public_tree_does_not_contain_a_developer_home_path():
    generated = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "dist", "build"}
    private_path = "/home/" + "david"
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in generated for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if private_path in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_hermes_tool_schema_keeps_the_gold_standard_search_and_update_surface():
    mod = load_provider_module()
    schemas = {item["name"]: item for item in mod.AgentRecallProvider().get_tool_schemas()}

    assert set(schemas["agent_recall_search"]["parameters"]["properties"]) == {
        "query",
        "limit",
        "include_shared",
        "category",
        "tags",
    }
    assert set(schemas["agent_recall_update"]["parameters"]["properties"]) == {
        "id",
        "content",
        "title",
        "summary",
        "visibility",
        "category",
        "tags",
        "importance",
        "confidence",
        "archived",
    }


def test_tool_descriptions_do_not_hardcode_a_specific_chat_model_or_non_chat_positioning():
    mod = load_provider_module()
    schemas = mod.AgentRecallProvider().get_tool_schemas()
    descriptions = "\n".join(schema.get("description", "") for schema in schemas).lower()
    forbidden = ["gpt-5.3-mini", "non-local-qwen", "embedding-only", "disabled unless", "disabled by default"]
    assert not any(term in descriptions for term in forbidden)


def test_default_config_enables_chat_curation_and_keeps_it_disableable(tmp_path):
    mod = load_provider_module()
    cfg = mod._default_config(tmp_path)
    assert cfg["llm_curator_enabled"] is True
    assert cfg["llm_curator_backend"] == "codex-cli"
    assert cfg["llm_curator_model"]
    assert cfg["llm_curator_command"]

    normalized = mod._normalize_config({**cfg, "llm_curator_enabled": "false"})
    assert normalized["llm_curator_enabled"] is False
