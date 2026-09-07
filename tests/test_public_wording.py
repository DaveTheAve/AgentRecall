from __future__ import annotations

import json
import re
import subprocess
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


def test_public_tree_does_not_ship_private_paths_identities_or_workstation_embedding_defaults():
    generated = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "dist", "build"}
    home_path = re.compile("/" + r"home/([^/\s\"']+)")
    mac_home_path = re.compile("/" + r"Users/([^/\s\"']+)")
    windows_home_path = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+")
    desktop_path = re.compile(r"(?:/|\\)Desktop(?:/|\\)")
    checkout_label = re.compile(r"AgentRecall_[A-Za-z0-9]")
    embedding_artifact = re.compile(r"(?i)\b[a-z0-9.]+-embedding-\d+b(?:-[a-z0-9_.-]+)?")
    mixture_model = re.compile(r"(?i)\b[a-z0-9.]+-\d+b-a\d+b\b")
    private_workspace = re.compile(r"\b[a-z0-9]+-hermes-shared\b")
    embedding_loopback = re.compile(r'"embedding_base_url"\s*:\s*"http://127\.0\.0\.1:(\d+)')
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in generated for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        reasons = []
        home_users = {match.group(1) for match in home_path.finditer(text)} - {"service"}
        if home_users or mac_home_path.search(text) or windows_home_path.search(text):
            reasons.append("personal home path")
        if desktop_path.search(text) or checkout_label.search(text):
            reasons.append("workstation checkout path")
        if embedding_artifact.search(text) or mixture_model.search(text):
            reasons.append("workstation model identifier")
        if private_workspace.search(text):
            reasons.append("private workspace identity")
        unsupported_ports = {match.group(1) for match in embedding_loopback.finditer(text)} - {"8000"}
        if unsupported_ports:
            reasons.append("workstation embedding endpoint")
        if reasons:
            offenders.append((str(path.relative_to(ROOT)), reasons))
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
        "expires_at",
    }


def test_tool_descriptions_do_not_hardcode_a_specific_chat_model_or_non_chat_positioning():
    mod = load_provider_module()
    schemas = mod.AgentRecallProvider().get_tool_schemas()
    descriptions = "\n".join(schema.get("description", "") for schema in schemas).lower()
    forbidden = ["gpt-5.3-mini", "non-local-model", "embedding-only", "disabled unless", "disabled by default"]
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


def test_release_metadata_is_consistently_versioned_as_0_3_0():
    expected = "0.3.0"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    openclaw = json.loads((ROOT / "openclaw.plugin.json").read_text(encoding="utf-8"))
    plugin_yaml = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    uv_lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    project_version = re.search(r'(?ms)^\[project\].*?^version = "([^"]+)"', pyproject)
    assert project_version and project_version.group(1) == expected
    assert package["version"] == expected
    assert openclaw["version"] == expected
    assert f"version: {expected}" in plugin_yaml
    assert 'name = "agent-recall-hermes-plugin"\nversion = "0.3.0"' in uv_lock


def test_python_distribution_packages_the_native_provider_and_manifest():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'packages = ["hermes_plugin"]' in pyproject
    assert '"agent_recall_session_archive"' in pyproject
    assert '"agent_recall_session_learning"' not in pyproject
    assert 'hermes_plugin = ["plugin.yaml"]' in pyproject
    packaged_manifest = ROOT / "hermes_plugin" / "plugin.yaml"
    assert (ROOT / "hermes_plugin" / "__init__.py").is_file()
    assert not (ROOT / "agent_recall_plugin").exists()
    assert packaged_manifest.read_text(encoding="utf-8") == (ROOT / "plugin.yaml").read_text(encoding="utf-8")


def test_readme_qualifies_markdown_import_as_a_hermes_only_tool():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Hermes additionally exposes controlled Markdown import" in readme


def test_v0_3_public_docs_cover_upgrade_and_new_lifecycle_fields():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    mcp = (ROOT / "docs" / "MCP.md").read_text(encoding="utf-8")
    openclaw = (ROOT / "docs" / "OPENCLAW.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    for term in ("v0.3.0", "canonical_key", "expires_at", "physical cleanup", "online backup"):
        assert term.casefold() in readme.casefold()
    for term in ("canonical_key", "expires_at"):
        assert term in mcp
    for term in ("canonicalKey", "expiresAt"):
        assert term in openclaw
    for term in ("0.3.0", "canonical", "expiration", "BM25"):
        assert term in changelog


def test_public_docs_preserve_supported_boundaries():
    documents = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "INSTALL.md", "SECURITY.md", "CHANGELOG.md")
    }
    documents.update(
        {
            str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "docs").glob("*.md"))
        }
    )

    readme = documents["README.md"]
    install = documents["INSTALL.md"]
    architecture = documents["docs/ARCHITECTURE.md"]
    security = documents["SECURITY.md"]
    changelog = documents["CHANGELOG.md"]
    mcp = documents["docs/MCP.md"]
    flat_mcp = " ".join(mcp.split())
    for term in ("SessionArchive", "current Hermes profile", "auto_capture_turns",
                 "auto_capture_compression_checkpoints", "agent_recall_remember"):
        assert term in readme
    for term in ("SessionArchive", "state.db FTS5", "host-neutral", "optional twelfth tool",
                 "never duplicates, migrates, or writes"):
        assert term in architecture
    for term in ("fail closed", "untrusted historical data", "caller-supplied database path",
                 "cross-profile session ID", "standard input rather than command-line arguments",
                 "bounded"):
        assert term in security
    for term in ('"session_archive_enabled": false', "raw Hermes conversation history",
                 "current profile", "untrusted historical data"):
        assert term in install
    for term in ("SessionArchive", "current Hermes profile", "untrusted data",
                 "MCP now defaults to read-only", "strict request schemas",
                 "stable public errors"):
        assert term in changelog
    for term in ("default is **read-only**", "fixed identity", "execution allowlist",
                 "65,536 serialized argument bytes", "1 MiB response budget",
                 "requires a bearer token", "Host and Origin", "Operation failed."):
        assert term in flat_mcp


def test_public_examples_and_benchmark_defaults_use_neutral_identities():
    public_examples = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in ("README.md", "INSTALL.md")
    )
    benchmark_script = (ROOT / "scripts" / "benchmark_retrieval.py").read_text(encoding="utf-8")
    assert "coding-agent" in public_examples.lower()
    assert re.search(r"\b[a-z0-9]+-hermes-shared\b", benchmark_script) is None


def test_release_candidate_excludes_local_environments_dependencies_and_private_backlog():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        release_files = set(result.stdout.split("\0"))
    else:
        sources = next(ROOT.glob("*.egg-info/SOURCES.txt"))
        release_files = set(sources.read_text(encoding="utf-8").splitlines())
    for excluded in (".venv", "node_modules", "IDEAS.md"):
        assert not any(path == excluded or path.startswith(f"{excluded}/") for path in release_files)
