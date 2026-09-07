from __future__ import annotations

import json
import subprocess

from conftest import FakeEmbedder, load_provider_module

from agent_recall_curator import ChatCompletionsCurator, CodexCliCurator, parse_curation_json


def test_parse_curation_json_accepts_fenced_json():
    text = 'Here you go:\n```json\n{"memories":[{"content":"User prefers concise output"}]}\n```'
    assert parse_curation_json(text) == [{
        "content": "User prefers concise output",
        "visibility": "agent",
        "tags": [],
        "importance": 0.5,
        "confidence": 0.8,
    }]


def test_parse_curation_json_normalizes_bad_optional_fields():
    text = json.dumps({"memories": [{
        "content": "  Durable fact  ",
        "visibility": "public",
        "tags": "bad",
        "importance": 7,
        "confidence": "bad",
    }]})
    assert parse_curation_json(text) == [{
        "content": "Durable fact",
        "visibility": "agent",
        "tags": [],
        "importance": 1.0,
        "confidence": 0.8,
    }]


def test_chat_completions_curator_calls_openai_compatible_endpoint(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"memories":[{"content":"User prefers durable memory"}]}'}}]
            }).encode("utf-8")

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    curator = ChatCompletionsCurator(
        base_url="https://example.invalid/v1",
        model="custom-chat-model",
        api_key="x",
        timeout=33,
    )

    result = curator.curate("Remember that the user prefers durable memory", default_visibility="agent")

    assert result == [{
        "content": "User prefers durable memory",
        "visibility": "agent",
        "tags": [],
        "importance": 0.5,
        "confidence": 0.8,
    }]
    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["timeout"] == 33.0
    assert seen["body"]["model"] == "custom-chat-model"
    assert seen["body"]["messages"][-1]["content"].startswith("You extract durable AI-agent memories")
    assert seen["auth"] == "Bearer x"


def test_codex_curator_failure_does_not_echo_prompt_or_process_output(monkeypatch):
    class FailedProcess:
        returncode = 2
        stdout = "sensitive stdout prompt echo"
        stderr = "sensitive stderr prompt echo"

    def fake_run(*args, **kwargs):
        return FailedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    curator = CodexCliCurator(command="codex", model="model")
    try:
        curator.curate("sensitive source text")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert message == "codex curator failed with exit code 2"
    assert "sensitive" not in message


def test_codex_curator_timeout_does_not_echo_prompt_or_command(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=[*cmd, "sensitive source text"], timeout=3)

    monkeypatch.setattr("subprocess.run", fake_run)
    curator = CodexCliCurator(command="codex", model="model", timeout=3)
    try:
        curator.curate("sensitive source text")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert message == "codex curator timed out after 3 seconds"
    assert "sensitive" not in message
    assert "cmd" not in message


def test_curator_tool_is_enabled_by_default_and_uses_configured_model(monkeypatch, tmp_path):
    mod = load_provider_module()

    seen = {}

    class FakeCurator:
        def __init__(self, command, model, timeout):
            seen["command"] = command
            seen["model"] = model
            seen["timeout"] = timeout

        def curate(self, text, *, default_visibility):
            seen["text"] = text
            return [{"content": "Curated durable fact", "visibility": default_visibility, "category": "decision"}]

    implementation = __import__(mod.AgentRecallProvider.__module__, fromlist=["CodexCliCurator"])
    monkeypatch.setattr(implementation, "CodexCliCurator", FakeCurator)
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        "llm_curator_model": "custom-chat-model",
        "llm_curator_command": "custom-codex",
        "llm_curator_timeout": 77,
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()

    result = json.loads(p.handle_tool_call("agent_recall_curate", {"text": "Remember that user prefers concise output", "dry_run": True}))

    assert result["success"] is True
    assert result["candidates"][0]["content"] == "Curated durable fact"
    assert seen == {
        "command": "custom-codex",
        "model": "custom-chat-model",
        "timeout": 77.0,
        "text": "Remember that user prefers concise output",
    }


def test_curator_tool_can_use_openai_compatible_backend(monkeypatch, tmp_path):
    mod = load_provider_module()
    seen = {}

    class FakeChatCurator:
        def __init__(self, base_url, model, api_key, timeout):
            seen["base_url"] = base_url
            seen["model"] = model
            seen["api_key"] = api_key
            seen["timeout"] = timeout

        def curate(self, text, *, default_visibility):
            seen["text"] = text
            return [{"content": "HTTP curated durable fact", "visibility": default_visibility, "category": "decision"}]

    implementation = __import__(mod.AgentRecallProvider.__module__, fromlist=["ChatCompletionsCurator"])
    monkeypatch.setattr(implementation, "ChatCompletionsCurator", FakeChatCurator)
    monkeypatch.setenv("AGENT_RECALL_TEST_CHAT_KEY", "test-key")
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        "llm_curator_backend": "openai-compatible",
        "llm_curator_base_url": "https://chat.example/v1",
        "llm_curator_api_key_env": "AGENT_RECALL_TEST_CHAT_KEY",
        "llm_curator_model": "custom-chat-model",
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()

    result = json.loads(p.handle_tool_call("agent_recall_curate", {"text": "Please extract memories", "default_visibility": "shared"}))

    assert result["success"] is True
    assert result["stored"] == 1
    assert seen == {
        "base_url": "https://chat.example/v1",
        "model": "custom-chat-model",
        "api_key": "test-key",
        "timeout": 120.0,
        "text": "Please extract memories",
    }


def test_curator_tool_can_be_disabled(tmp_path):
    mod = load_provider_module()
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        "llm_curator_enabled": False,
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()
    result = json.loads(p.handle_tool_call("agent_recall_curate", {"text": "Remember that user prefers concise output"}))
    assert result["success"] is False
    assert "disabled" in result["error"]


def test_curator_tool_can_store_candidates_from_fake_codex(monkeypatch, tmp_path):
    mod = load_provider_module()

    class FakeCurator:
        def __init__(self, command, model, timeout):
            assert model == "gpt-5.3-mini"

        def curate(self, text, *, default_visibility):
            return [{"content": "Curated durable fact", "visibility": default_visibility, "category": "decision"}]

    implementation = __import__(mod.AgentRecallProvider.__module__, fromlist=["CodexCliCurator"])
    monkeypatch.setattr(implementation, "CodexCliCurator", FakeCurator)
    p = mod.AgentRecallProvider({
        "db_path": str(tmp_path / "agent-recall.db"),
        "workspace_id": "ws",
        "agent_id": "hermes",
        "embedding_base_url": "",
        "embedding_model": "fake",
        "llm_curator_enabled": True,
        "llm_curator_model": "gpt-5.3-mini",
    })
    p.initialize("s", hermes_home=tmp_path, agent_identity="hermes")
    p._embedder = FakeEmbedder()

    result = json.loads(p.handle_tool_call("agent_recall_curate", {"text": "Please extract memories", "default_visibility": "shared"}))
    assert result["success"] is True
    assert result["stored"] == 1
    found = json.loads(p.handle_tool_call("agent_recall_search", {"query": "curated durable"}))
    assert found["results"][0]["content"] == "Curated durable fact"
    assert found["results"][0]["visibility"] == "shared"
