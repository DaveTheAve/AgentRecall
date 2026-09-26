"""Real-host regression: manifests are generic hooks, not MemoryProvider methods."""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_mcp_extra_accepts_current_hermes_sdk():
    tomllib = pytest.importorskip("tomllib", reason="metadata check needs Python 3.11+")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirement = next(Requirement(item) for item in project["project"]["optional-dependencies"]["mcp"]
                       if Requirement(item).name == "mcp")
    assert "2.0.0" in requirement.specifier


@pytest.mark.parametrize("layout", ["source", "package"])
def test_real_host_doctor_discovery_and_lifecycle(tmp_path, layout):
    host = Path(os.environ.get("AGENT_RECALL_HERMES_SOURCE", Path.home() / ".hermes/hermes-agent"))
    if not (host / "hermes_cli/plugin_dev.py").is_file():
        pytest.skip("real Hermes checkout unavailable; set AGENT_RECALL_HERMES_SOURCE")
    python = os.environ.get("AGENT_RECALL_HERMES_PYTHON", str(host / "venv/bin/python"))
    target = ROOT if layout == "source" else ROOT / "hermes_plugin"
    script = textwrap.dedent('''
        import inspect
        import json
        import os
        from pathlib import Path
        from hermes_cli.plugin_dev import doctor_plugin
        from agent.memory_provider import MemoryProvider
        from agent.memory_manager import MemoryManager
        from plugins.memory import discover_memory_providers, load_memory_provider

        target = Path(os.environ["PLUGIN_TARGET"])
        report = doctor_plugin(target)
        assert report.ok, report.format_text()
        assert not report.findings, report.format_text()
        assert report.manifest.provides_hooks == []
        assert report.registered_hooks == ()
        home = Path(os.environ["HERMES_HOME"])
        (home / "plugins").mkdir()
        (home / "plugins/agent-recall").symlink_to(target, target_is_directory=True)
        assert any(name == "agent-recall" and available
                   for name, _, available in discover_memory_providers())
        provider = load_memory_provider("agent-recall")
        assert isinstance(provider, MemoryProvider)
        assert Path(inspect.getfile(type(provider))).resolve().is_relative_to(Path(os.environ["REPO"]))
        provider.save_config({"db_path": str(home / "memory.db"), "embedding_base_url": "",
                             "llm_curator_enabled": False, "auto_capture_turns": True,
                             "auto_capture_compression_checkpoints": True}, home)
        manager = MemoryManager()
        manager.add_provider(provider)
        manager.initialize_all("compat-session", hermes_home=home, agent_identity="compat-agent")
        try:
            manager.sync_all("turnneedle permanent compatibility example", "Acknowledged.", session_id="compat-session")
            assert manager.flush_pending(timeout=10)
            # The provider also owns a tracked writer; join rather than sleeping.
            for thread in provider._core._sync_threads:
                thread.join(timeout=10)
                assert not thread.is_alive()
            assert "checkpoint id=" in manager.on_pre_compress([
                {"role": "user", "content": "checkpointneedle compatibility checkpoint with enough context."}])
            manager.on_memory_write("add", "memory", "mirrorneedle compatibility preference")
            for query in ("turnneedle", "checkpointneedle", "mirrorneedle"):
                result = json.loads(manager.handle_tool_call("agent_recall_search", {"query": query}))
                assert result["count"] >= 1, (query, result)
                assert any(query in row["content"] for row in result["results"]), result
        finally:
            manager.shutdown_all()
        print(report.format_text())
        print("discovery + MemoryManager lifecycle callbacks preserved")
    ''')
    env = dict(os.environ, HERMES_HOME=str(tmp_path), HERMES_ENABLE_PROJECT_PLUGINS="0",
               PLUGIN_TARGET=str(target), REPO=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               HERMES_DISABLE_LAZY_INSTALLS="1",
               PYTHONPATH=os.pathsep.join([str(host), str(ROOT)]))
    result = subprocess.run([python, "-c", script], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
