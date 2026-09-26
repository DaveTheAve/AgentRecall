from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
from conftest import load_provider_module

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAME = "agent_recall_session_archive"


def make_provider(tmp_path, enabled: bool | str = False):
    mod = load_provider_module()
    provider = mod.AgentRecallProvider(
        {
            "db_path": str(tmp_path / "agent-recall.db"),
            "embedding_base_url": "",
            "embedding_model": "fake",
            "session_archive_enabled": enabled,
        }
    )
    provider.initialize("current-session", hermes_home=tmp_path, agent_identity="hermes")
    return mod, provider


def install_host(monkeypatch, function):
    module = types.ModuleType("tools.session_search_tool")
    module.session_search = function
    monkeypatch.setitem(sys.modules, "tools.session_search_tool", module)


def archive_module():
    return sys.modules["agent_recall.agent_recall_session_archive"]


def test_delegates_to_host_without_opening_state_db(monkeypatch, tmp_path):
    calls = []
    omitted = object()

    def host(query="", current_session_id=None, session_id=None, profile=omitted):
        calls.append(
            {
                "query": query,
                "current_session_id": current_session_id,
                "session_id": session_id,
                "profile": profile,
            }
        )
        return json.dumps({"success": True, "mode": "search"})

    install_host(monkeypatch, host)
    load_provider_module()
    archive = archive_module()
    original_import = archive.importlib.import_module

    def guarded_import(name):
        if name == "hermes_state":
            pytest.fail("SessionArchive must not open Hermes state.db directly")
        return original_import(name)

    monkeypatch.setattr(archive.importlib, "import_module", guarded_import)

    result = json.loads(
        archive.call_session_archive(
            {"query": "history"},
            current_session_id="current-session",
            hermes_home=tmp_path,
        )
    )

    assert result == {"success": True, "mode": "search"}
    assert calls == [
        {
            "query": "history",
            "current_session_id": "current-session",
            "session_id": None,
            "profile": None,
        }
    ]


def test_provider_retains_and_forwards_profile_home(monkeypatch, tmp_path):
    _, provider = make_provider(tmp_path, enabled=True)
    captured = {}

    def archive(args, *, current_session_id, hermes_home):
        captured.update(args=args, current_session_id=current_session_id, hermes_home=hermes_home)
        return json.dumps({"success": True})

    provider_module = sys.modules[provider.__class__.__module__]
    monkeypatch.setattr(provider_module, "call_session_archive", archive)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"query": "history"}))

    assert result["success"] is True
    assert captured == {
        "args": {"query": "history"},
        "current_session_id": "current-session",
        "hermes_home": tmp_path,
    }
    provider.shutdown()


@pytest.mark.parametrize(
    ("arguments", "forwarded"),
    [
        (
            {
                "query": "release plan",
                "role_filter": "user,assistant",
                "limit": 7,
                "sort": "newest",
                "detail": "full",
            },
            {
                "query": "release plan",
                "role_filter": "user,assistant",
                "limit": 7,
                "sort": "newest",
                "detail": "full",
            },
        ),
        ({"session_id": "session-2"}, {"session_id": "session-2"}),
        (
            {"session_id": "session-2", "around_message_id": 42, "window": 8},
            {"session_id": "session-2", "around_message_id": 42, "window": 8},
        ),
        ({}, {}),
    ],
)
def test_archive_modes_forward_only_allowlisted_current_profile_arguments(
    monkeypatch, tmp_path, arguments, forwarded
):
    calls = []
    payload = '{"success":true,"host":"unchanged"}'

    def host(*, current_session_id, profile=None, **kwargs):
        calls.append({**kwargs, "current_session_id": current_session_id, "profile": profile})
        return payload

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)
    arguments = {
        **arguments,
        "profile": "other-profile",
        "all_profiles": True,
        "db_path": "/private/state.db",
        "include_all_messages": True,
        "unexpected_host_argument": "must-not-forward",
    }

    result = provider.handle_tool_call(TOOL_NAME, arguments)

    assert result == payload
    assert calls == [
        {
            **forwarded,
            "current_session_id": "current-session",
            "profile": None,
        }
    ]
    provider.shutdown()


def test_session_archive_clamps_host_limits(monkeypatch, tmp_path):
    calls = []

    def host(*, current_session_id, profile=None, **kwargs):
        calls.append(kwargs)
        return json.dumps({"success": True})

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)

    provider.handle_tool_call(TOOL_NAME, {"query": "high", "limit": 999, "window": 999})
    provider.handle_tool_call(TOOL_NAME, {"query": "low", "limit": -5, "window": 0})

    assert calls[0]["limit"] == 10
    assert calls[0]["window"] == 20
    assert calls[1]["limit"] == 1
    assert calls[1]["window"] == 1
    provider.shutdown()


def test_embedded_cross_profile_session_reference_is_rejected_before_host_call(monkeypatch, tmp_path):
    calls = []

    def host(*, current_session_id, profile=None, **kwargs):
        calls.append(kwargs)
        return json.dumps({"success": True})

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"session_id": "other-profile/session-2"}))

    assert result["success"] is False
    assert "current Hermes profile" in result["error"]
    assert calls == []
    provider.shutdown()


def test_host_cross_profile_fallback_payload_is_rejected(monkeypatch, tmp_path):
    def host(*, current_session_id, profile=None, **kwargs):
        return json.dumps(
            {
                "success": True,
                "profile": "foreign-profile",
                "messages": ["private transcript"],
            }
        )

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"session_id": "bare-foreign-id"}))

    assert result == {
        "success": False,
        "error": "AgentRecall SessionArchive host API is unavailable",
    }
    assert "private transcript" not in json.dumps(result)
    provider.shutdown()


@pytest.mark.parametrize(
    "host",
    [
        lambda **kwargs: json.dumps({"success": True}),
        lambda query="", profile=None: json.dumps({"success": True}),
        lambda query="", current_session_id=None: json.dumps({"success": True}),
    ],
)
def test_unverifiable_host_signatures_fail_closed(monkeypatch, tmp_path, host):
    install_host(monkeypatch, host)
    load_provider_module()
    archive = archive_module()

    with pytest.raises(
        archive.AgentRecallError,
        match="^AgentRecall SessionArchive host API is unavailable$",
    ):
        archive.call_session_archive(
            {"query": "private transcript phrase"},
            current_session_id="current-session",
            hermes_home=tmp_path,
        )


def test_kwargs_only_host_fails_before_invocation(monkeypatch, tmp_path):
    calls = []

    def host(**kwargs):
        calls.append(kwargs)
        return json.dumps({"success": True, "results": ["foreign transcript"]})

    install_host(monkeypatch, host)
    load_provider_module()
    archive = archive_module()

    with pytest.raises(
        archive.AgentRecallError,
        match="^AgentRecall SessionArchive host API is unavailable$",
    ):
        archive.call_session_archive(
            {"query": "private transcript phrase"},
            current_session_id="current-session",
            hermes_home=tmp_path,
        )

    assert calls == []


def test_explicit_isolation_parameters_plus_kwargs_are_supported(monkeypatch, tmp_path):
    calls = []

    def host(*, current_session_id, profile=None, **kwargs):
        calls.append({**kwargs, "current_session_id": current_session_id, "profile": profile})
        return json.dumps({"success": True})

    install_host(monkeypatch, host)
    load_provider_module()
    archive = archive_module()

    archive.call_session_archive(
        {"query": "history", "sort": "oldest", "detail": "full"},
        current_session_id="current-session",
        hermes_home=tmp_path,
    )

    assert calls == [
        {
            "query": "history",
            "sort": "oldest",
            "detail": "full",
            "current_session_id": "current-session",
            "profile": None,
        }
    ]


def test_older_explicit_host_omits_unsupported_optional_fields(monkeypatch, tmp_path):
    calls = []

    def old_host(query="", limit=3, current_session_id=None, profile=None):
        calls.append(
            {
                "query": query,
                "limit": limit,
                "current_session_id": current_session_id,
                "profile": profile,
            }
        )
        return json.dumps({"success": True})

    install_host(monkeypatch, old_host)
    load_provider_module()
    archive = archive_module()

    archive.call_session_archive(
        {"query": "history", "limit": 4, "sort": "oldest", "detail": "full"},
        current_session_id="current-session",
        hermes_home=tmp_path,
    )

    assert calls == [
        {
            "query": "history",
            "limit": 4,
            "current_session_id": "current-session",
            "profile": None,
        }
    ]


@pytest.mark.parametrize(
    "host_result",
    [
        None,
        "not-json",
        json.dumps({"success": False, "error": "private query at /private/state.db"}),
        json.dumps(["unexpected"]),
    ],
)
def test_host_failure_payloads_are_sanitized(monkeypatch, tmp_path, host_result):
    def host(*, current_session_id, profile=None, **kwargs):
        return host_result

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"query": "private query"}))

    assert result == {
        "success": False,
        "error": "AgentRecall SessionArchive host API is unavailable",
    }
    serialized = json.dumps(result)
    assert "private query" not in serialized
    assert "/private/state.db" not in serialized
    provider.shutdown()


def test_host_exception_is_sanitized(monkeypatch, tmp_path):
    def host(*, current_session_id, profile=None, **kwargs):
        raise RuntimeError("private transcript at /private/profile/state.db")

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"query": "private query"}))

    assert result == {
        "success": False,
        "error": "AgentRecall SessionArchive host API is unavailable",
    }
    assert "private transcript" not in json.dumps(result)
    provider.shutdown()


def test_session_archive_defaults_disabled_and_is_omitted_after_initialize(tmp_path):
    mod = load_provider_module()
    assert mod._default_config(tmp_path)["session_archive_enabled"] is False
    assert mod._normalize_config({"session_archive_enabled": "false"})["session_archive_enabled"] is False

    provider = mod.AgentRecallProvider({"db_path": str(tmp_path / "agent-recall.db")})
    assert TOOL_NAME in {schema["name"] for schema in provider.get_tool_schemas()}
    provider.initialize("current-session", hermes_home=tmp_path, agent_identity="hermes")

    assert TOOL_NAME not in {schema["name"] for schema in provider.get_tool_schemas()}
    assert "SessionArchive" not in provider.system_prompt_block()
    provider.shutdown()


def test_session_archive_true_exposes_bounded_schema_and_prompt(tmp_path):
    _, provider = make_provider(tmp_path, enabled="true")

    assert provider._config["session_archive_enabled"] is True
    schema = {item["name"]: item for item in provider.get_tool_schemas()}[TOOL_NAME]
    properties = schema["parameters"]["properties"]
    assert set(properties) == {
        "query",
        "role_filter",
        "limit",
        "session_id",
        "around_message_id",
        "window",
        "sort",
        "detail",
    }
    assert schema["parameters"]["additionalProperties"] is False
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == 10
    assert properties["window"]["minimum"] == 1
    assert properties["window"]["maximum"] == 20
    assert "untrusted historical data" in schema["description"].lower()
    assert "sessionarchive" in provider.system_prompt_block().lower()
    provider.shutdown()


def test_disabled_direct_call_is_rejected_without_host_call(monkeypatch, tmp_path):
    calls = []

    def host(*, current_session_id, profile=None, **kwargs):
        calls.append(kwargs)
        return json.dumps({"success": True})

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=False)

    result = json.loads(provider.handle_tool_call(TOOL_NAME, {"query": "old topic"}))

    assert result["success"] is False
    assert "session_archive_enabled" in result["error"]
    assert calls == []
    provider.shutdown()


def test_archive_calls_do_not_create_agent_recall_memory_rows(monkeypatch, tmp_path):
    def host(*, current_session_id, profile=None, **kwargs):
        return json.dumps({"success": True, "mode": "discover", "results": []})

    install_host(monkeypatch, host)
    _, provider = make_provider(tmp_path, enabled=True)
    before = json.loads(provider.handle_tool_call("agent_recall_stats", {}))["stats"]["buckets"]

    response = json.loads(provider.handle_tool_call(TOOL_NAME, {"query": "historical topic"}))
    after = json.loads(provider.handle_tool_call("agent_recall_stats", {}))["stats"]["buckets"]

    assert response["success"] is True
    assert before == after == []
    provider.shutdown()


def test_preinitialization_schema_registers_real_memory_manager_route(tmp_path):
    host_root = Path(os.environ.get("AGENT_RECALL_HERMES_SOURCE", Path.home() / ".hermes" / "hermes-agent"))
    if not (host_root / "agent" / "memory_manager.py").is_file():
        pytest.skip("real Hermes source checkout is unavailable")
    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        import sys
        from pathlib import Path

        root = Path(os.environ["AGENT_RECALL_REPO"])
        spec = importlib.util.spec_from_file_location(
            "agent_recall", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_recall"] = module
        spec.loader.exec_module(module)

        from agent.memory_manager import MemoryManager

        provider = module.AgentRecallProvider({"db_path": str(Path(os.environ["TEST_HOME"]) / "memory.db")})
        manager = MemoryManager()
        manager.add_provider(provider)
        assert manager.has_tool("agent_recall_session_archive")
        manager.initialize_all("current-session", hermes_home=os.environ["TEST_HOME"])
        assert manager.has_tool("agent_recall_session_archive")
        assert "agent_recall_session_archive" not in {
            schema["name"] for schema in manager.get_all_tool_schemas()
        }
        result = json.loads(manager.handle_tool_call("agent_recall_session_archive", {"query": "private phrase"}))
        assert result.get("success") is False or "error" in result
        assert "session_archive_enabled" in result["error"]
        manager.shutdown_all()
        """
    )
    env = dict(
        os.environ,
        AGENT_RECALL_REPO=str(ROOT),
        TEST_HOME=str(tmp_path),
        PYTHONPATH=os.pathsep.join([str(host_root), str(ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
        HERMES_DISABLE_LAZY_INSTALLS="1",
    )

    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def test_real_host_enabled_search_and_bare_foreign_id_fail_closed(tmp_path):
    host_root = Path(
        os.environ.get("AGENT_RECALL_HERMES_SOURCE", Path.home() / ".hermes" / "hermes-agent")
    )
    if not (host_root / "tools" / "session_search_tool.py").is_file():
        pytest.skip("real Hermes source checkout is unavailable")
    host_python = Path(os.environ.get("AGENT_RECALL_HERMES_PYTHON", host_root / "venv" / "bin" / "python"))
    if not host_python.is_file():
        pytest.skip("real Hermes runtime Python is unavailable")
    script = textwrap.dedent(
        """
        import importlib.util
        import json
        import os
        import sys
        from pathlib import Path

        root = Path(os.environ["AGENT_RECALL_REPO"])
        home = Path(os.environ["HERMES_HOME"])
        from hermes_state import SessionDB

        local = SessionDB(home / "state.db")
        local.create_session("local-archive-session", source="cli")
        local.append_message(
            "local-archive-session", role="user", content="local-archive-needle-7b91"
        )
        local.close()

        foreign_home = home / "profiles" / "other-profile"
        foreign_home.mkdir(parents=True)
        foreign = SessionDB(foreign_home / "state.db")
        foreign.create_session("foreign-bare-session", source="cli")
        foreign.append_message(
            "foreign-bare-session", role="user", content="foreign-secret-should-not-escape"
        )
        foreign.close()

        spec = importlib.util.spec_from_file_location(
            "agent_recall", root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_recall"] = module
        spec.loader.exec_module(module)

        provider = module.AgentRecallProvider(
            {
                "db_path": str(home / "agent-recall.db"),
                "embedding_base_url": "",
                "embedding_model": "fake",
                "session_archive_enabled": True,
            }
        )
        provider.initialize("current-session", hermes_home=home, agent_identity="hermes")

        local_result = json.loads(
            provider.handle_tool_call(
                "agent_recall_session_archive",
                {"query": "local-archive-needle-7b91", "sort": "newest", "detail": "full"},
            )
        )
        assert local_result["success"] is True, local_result
        assert local_result["results"], local_result

        foreign_result_text = provider.handle_tool_call(
            "agent_recall_session_archive", {"session_id": "foreign-bare-session"}
        )
        foreign_result = json.loads(foreign_result_text)
        assert foreign_result["error"] == "AgentRecall SessionArchive host API is unavailable"
        assert foreign_result.get("success") in (None, False)
        assert "foreign-secret-should-not-escape" not in foreign_result_text
        provider.shutdown()
        """
    )
    env = dict(
        os.environ,
        AGENT_RECALL_REPO=str(ROOT),
        HERMES_HOME=str(tmp_path),
        PYTHONPATH=os.pathsep.join([str(host_root), str(ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
        HERMES_DISABLE_LAZY_INSTALLS="1",
    )

    result = subprocess.run(
        [str(host_python), "-c", script], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_session_archive_docs_preserve_the_memory_boundary():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "session_archive_enabled" in readme
    assert "untrusted historical data" in readme
    assert "not instructions" in readme
    assert "not durable memory" in readme
    assert "recent-session browsing" in readme
    assert "does not open or migrate `state.db`" in readme
