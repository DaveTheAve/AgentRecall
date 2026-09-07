from __future__ import annotations

import json
import subprocess

import pytest
from conftest import FakeEmbedder, load_provider_module

import agent_recall_curator as curator_module
from agent_recall_core import AgentIdentity, AgentRecallCore
from agent_recall_curator import (
    CURATION_SCHEMA,
    ChatCompletionsCurator,
    CodexCliCurator,
    build_curation_prompt,
    parse_curation_json,
)


def test_parse_curation_json_accepts_fenced_json():
    text = 'Here you go:\n```json\n{"memories":[{"content":"User prefers concise output"}]}\n```'
    assert parse_curation_json(text) == [{
        "content": "User prefers concise output",
        "visibility": "agent",
        "tags": [],
        "importance": 0.5,
        "confidence": 0.8,
    }]


def test_parse_curation_json_rejects_bad_optional_fields():
    text = json.dumps({"memories": [{
        "content": "  Durable fact  ",
        "visibility": "public",
        "tags": "bad",
        "importance": 7,
        "confidence": "bad",
    }]})
    assert parse_curation_json(text) == []


@pytest.mark.parametrize("visibility", [[], {}])
def test_parse_curation_json_rejects_unhashable_visibility_without_raising(visibility):
    text = json.dumps({"memories": [{"content": "Durable fact", "visibility": visibility}]})

    assert parse_curation_json(text) == []


def test_parse_curation_json_rejects_oversized_response_and_bounds_candidate_count():
    oversized = json.dumps({"memories": [{"content": "x" * 140_000}]})
    many = json.dumps({"memories": [{"content": f"durable fact {index}"} for index in range(100)]})

    assert parse_curation_json(oversized) == []
    parsed = parse_curation_json(many)
    assert len(parsed) == 32
    assert parsed[-1]["content"] == "durable fact 31"


def test_parse_curation_json_rejects_oversized_content_and_optional_fields():
    text = json.dumps(
        {
            "memories": [
                {"content": "x" * 1201},
                {
                    "content": "Bounded durable fact",
                    "title": "t" * 201,
                    "summary": "s" * 501,
                    "category": "c" * 81,
                    "canonical_key_hint": "h" * 257,
                    "tags": [f"tag-{index}" for index in range(20)] + ["z" * 65],
                    "metadata": {"untrusted": "model-controlled"},
                },
            ]
        }
    )

    parsed = parse_curation_json(text)

    assert parsed == []


def test_parse_curation_json_rejects_non_finite_or_unrepresentable_scores():
    non_finite = parse_curation_json(
        '{"memories":[{"content":"Durable fact","importance":NaN,"confidence":Infinity}]}'
    )
    unrepresentable = parse_curation_json(
        json.dumps({"memories": [{"content": "Durable fact", "confidence": 10**400}]})
    )

    assert non_finite == []
    assert unrepresentable == []


def test_parse_curation_json_keeps_small_metadata_and_omits_oversized_metadata():
    small_metadata = {"source": "session-end", "turns": 4}
    oversized_metadata = {"note": "x" * 4097}
    parsed = parse_curation_json(
        json.dumps(
            {
                "memories": [
                    {"content": "Small metadata fact", "metadata": small_metadata},
                    {"content": "Oversized metadata fact", "metadata": oversized_metadata},
                ]
            }
        )
    )

    assert parsed[0]["metadata"] == small_metadata
    assert "metadata" not in parsed[1]


def test_parse_curation_json_omits_metadata_with_too_many_properties():
    small_metadata = {"source": "session-end", "turns": 4}
    too_many_properties = {
        f"key-{index}": "v"
        for index in range(curator_module.MAX_CURATED_METADATA_PROPERTIES + 1)
    }

    parsed = parse_curation_json(
        json.dumps(
            {
                "memories": [
                    {"content": "Small metadata fact", "metadata": small_metadata},
                    {"content": "Property-heavy metadata fact", "metadata": too_many_properties},
                ]
            }
        )
    )

    assert parsed[0]["metadata"] == small_metadata
    assert "metadata" not in parsed[1]


def test_curation_schema_mirrors_all_runtime_collection_and_string_bounds():
    memories_schema = CURATION_SCHEMA["properties"]["memories"]
    item_schema = memories_schema["items"]
    properties = item_schema["properties"]

    assert memories_schema["maxItems"] == curator_module.MAX_CURATED_MEMORIES
    assert properties["content"]["maxLength"] == curator_module.MAX_CURATED_CONTENT_CHARS
    for field, limit in curator_module._OPTIONAL_STRING_LIMITS.items():
        assert properties[field]["maxLength"] == limit
    assert properties["tags"]["maxItems"] == curator_module.MAX_CURATED_TAGS
    assert properties["tags"]["items"]["maxLength"] == curator_module._MAX_CURATED_TAG_CHARS
    assert properties["metadata"]["maxProperties"] == curator_module.MAX_CURATED_METADATA_PROPERTIES


def test_parser_returns_non_authoritative_canonical_key_hint():
    text = json.dumps(
        {
            "memories": [
                {
                    "content": "The user prefers concise release summaries.",
                    "category": "user_pref",
                    "canonical_key_hint": "preferences.release-summary",
                }
            ]
        }
    )

    assert parse_curation_json(text)[0]["canonical_key_hint"] == "preferences.release-summary"
    item_schema = CURATION_SCHEMA["properties"]["memories"]["items"]
    assert item_schema["properties"]["canonical_key_hint"]["type"] == "string"
    prompt = build_curation_prompt("Remember concise release summaries")
    assert "stable topical hint" in prompt
    assert "NOT authoritative" in prompt


def test_generic_curate_cannot_persist_or_overwrite_with_model_keys(tmp_path):
    candidate = {
        "content": "The user prefers concise release summaries.",
        "category": "user_pref",
        "canonical_key": "manual.existing.memory",
        "canonical_key_hint": "manual.existing.memory",
    }

    class HintingCurator:
        def curate(self, text, *, default_visibility):
            return [{**candidate, "visibility": default_visibility}]

    core = AgentRecallCore(
        {
            "db_path": str(tmp_path / "curator-hint.db"),
            "embedding_base_url": "",
            "embedding_model": "fake",
            "llm_curator_enabled": True,
        },
        AgentIdentity("workspace", "agent", "session"),
        curator_factory=HintingCurator,
    )
    core.embedder = FakeEmbedder()
    manual = core.remember(
        {
            "content": "A manually keyed fact must remain unchanged.",
            "canonical_key": "manual.existing.memory",
        }
    )

    result = core.curate({"text": "Remember concise release summaries"})
    curated = core.get_memory(result["results"][0]["id"])["memory"]
    original = core.get_memory(manual["id"])["memory"]

    assert result["results"][0]["action"] == "added"
    assert curated["canonical_key"] == ""
    assert original["content"] == "A manually keyed fact must remain unchanged."
    core.close()


def test_chat_completions_curator_calls_openai_compatible_endpoint(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self, limit):
            seen["read_limit"] = limit
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
    assert seen["read_limit"] == curator_module.MAX_CURATION_TRANSPORT_BYTES + 1


def test_chat_completions_curator_rejects_oversized_transport_before_json_parse(monkeypatch):
    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self, limit):
            assert limit == curator_module.MAX_CURATION_TRANSPORT_BYTES + 1
            return b"x" * limit

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: OversizedResponse())

    def forbidden_json_loads(value):
        raise AssertionError("oversized transport reached outer JSON parsing")

    monkeypatch.setattr(curator_module.json, "loads", forbidden_json_loads)
    curator = ChatCompletionsCurator(base_url="https://example.invalid/v1", model="model")

    with pytest.raises(RuntimeError, match="curator response exceeds transport limit"):
        curator.curate("Remember bounded responses")


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


def test_codex_curator_sends_prompt_over_stdin_and_reads_bounded_output(monkeypatch):
    prompt_marker = "complete-transcript-derived-prompt-marker"
    seen = {}

    class SuccessfulProcess:
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        output_path = cmd[cmd.index("--output-last-message") + 1]
        curator_module.Path(output_path).write_text(
            json.dumps({"memories": [{"content": "Bounded durable fact"}]}),
            encoding="utf-8",
        )
        return SuccessfulProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    curator = CodexCliCurator(command="codex", model="model")

    result = curator.curate(prompt_marker)

    assert result[0]["content"] == "Bounded durable fact"
    assert seen["cmd"][-1] == "-"
    # Approval is a root option, not an `exec` option in supported Codex CLI.
    assert seen["cmd"].index("--ask-for-approval") < seen["cmd"].index("exec")
    assert all(prompt_marker not in str(argument) for argument in seen["cmd"])
    assert prompt_marker in seen["kwargs"]["input"]
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["stdout"] is subprocess.DEVNULL
    assert seen["kwargs"]["stderr"] is subprocess.DEVNULL
    assert "capture_output" not in seen["kwargs"]


def test_codex_curator_rejects_oversized_output_before_read_all_or_parse(monkeypatch):
    prompt_marker = "sensitive-oversized-prompt-marker"
    output_marker = "sensitive-oversized-output-marker"
    seen = {}

    class SuccessfulProcess:
        returncode = 0

    def fake_run(cmd, **kwargs):
        output_path = cmd[cmd.index("--output-last-message") + 1]
        seen["output_path"] = output_path
        curator_module.Path(output_path).write_bytes(
            output_marker.encode("utf-8")
            + b"x" * (curator_module.MAX_CURATION_TRANSPORT_BYTES + 1)
        )
        return SuccessfulProcess()

    def forbidden_read_text(*args, **kwargs):
        raise AssertionError("oversized output was read without a byte bound")

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("oversized output reached the parser")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(curator_module.Path, "read_text", forbidden_read_text)
    monkeypatch.setattr(curator_module, "parse_curation_json", forbidden_parse)
    curator = CodexCliCurator(command="codex", model="model")

    with pytest.raises(RuntimeError) as raised:
        curator.curate(prompt_marker)

    message = str(raised.value)
    assert message == "codex curator output unavailable or invalid"
    assert prompt_marker not in message
    assert output_marker not in message
    assert seen["output_path"] not in message


def test_codex_curator_rejects_success_without_output_file(monkeypatch):
    prompt_marker = "sensitive-missing-output-prompt-marker"
    stdout_marker = "sensitive-captured-stdout-marker"

    class SuccessfulProcess:
        returncode = 0
        stdout = json.dumps({"memories": [{"content": stdout_marker}]})

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SuccessfulProcess())
    curator = CodexCliCurator(command="codex", model="model")

    with pytest.raises(RuntimeError) as raised:
        curator.curate(prompt_marker)

    message = str(raised.value)
    assert message == "codex curator output unavailable or invalid"
    assert prompt_marker not in message
    assert stdout_marker not in message


def test_codex_curator_rejects_output_symlink(monkeypatch, tmp_path):
    prompt_marker = "sensitive-symlink-prompt-marker"
    target = tmp_path / "sensitive-symlink-target.json"
    target.write_text('{"memories":[{"content":"must not be read"}]}', encoding="utf-8")

    class SuccessfulProcess:
        returncode = 0

    def fake_run(cmd, **kwargs):
        output_path = curator_module.Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.symlink_to(target)
        return SuccessfulProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    curator = CodexCliCurator(command="codex", model="model")

    with pytest.raises(RuntimeError) as raised:
        curator.curate(prompt_marker)

    message = str(raised.value)
    assert message == "codex curator output unavailable or invalid"
    assert prompt_marker not in message
    assert str(target) not in message


def test_codex_curator_rejects_non_regular_output(monkeypatch):
    prompt_marker = "sensitive-non-regular-prompt-marker"

    class SuccessfulProcess:
        returncode = 0

    def fake_run(cmd, **kwargs):
        output_path = curator_module.Path(cmd[cmd.index("--output-last-message") + 1])
        output_path.mkdir()
        return SuccessfulProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    curator = CodexCliCurator(command="codex", model="model")

    with pytest.raises(RuntimeError) as raised:
        curator.curate(prompt_marker)

    message = str(raised.value)
    assert message == "codex curator output unavailable or invalid"
    assert prompt_marker not in message


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
